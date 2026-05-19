from skimage.feature import hog
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from lightglue import SuperPoint
import matplotlib.cm as cm

import base64
import json

import struct

# from enum import Enum

# class methods(Enum):
#     DAISY = 0
#     RESNET_MULTI = 1
#     RESNET = 2
#     SIFT = 3


device = "cuda" if torch.cuda.is_available() else "cpu"

daisy = cv2.xfeatures2d.DAISY_create(
    radius=5,
    q_radius=3,
    q_theta=4,
    q_hist=8,
    norm=cv2.xfeatures2d.DAISY_NRM_FULL,
    interpolation=False,
    use_orientation=False
)

sift = cv2.SIFT_create()


_MODEL_INSTANCE = None
last_model_name = ""
def get_matching_model(model_name=''):
    global _MODEL_INSTANCE
    

    if model_name != model_name or (_MODEL_INSTANCE is None):
        if model_name == 'roma':
            from roma_match import roma_outdoor
            _MODEL_INSTANCE =  roma_outdoor(device=device)
        elif model_name == 'dino':
            _MODEL_INSTANCE = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').eval().to(device)
        elif model_name == 'superpoint':
            _MODEL_INSTANCE =  SuperPoint(max_num_keypoints=None).eval().to(device)
        elif model_name == 'resnet_multiscale':
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            layer0 = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool
            ).eval().to(device)
            layer1 = backbone.layer1.eval().to(device)   # H/4
            layer2 = backbone.layer2.eval().to(device)   # H/8
            layer3 = backbone.layer3.eval().to(device)   # H/16
            proj = nn.Conv2d(64 + 128 + 256, 104, kernel_size=1).to(device).eval()
            _MODEL_INSTANCE = (layer0, layer1, layer2, layer3, proj)
        elif model_name == 'resnet':

            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

            _MODEL_INSTANCE = torch.nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2   # <-- STOP HERE (key change)
            ).eval().to(device)
        elif model_name == 'resnet_full':
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            _MODEL_INSTANCE = torch.nn.Sequential(
                *list(backbone.children())[:-2]
            ).eval().to(device)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    last_model_name = model_name
    return _MODEL_INSTANCE

def extract_hog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    feat = hog(
        gray,
        pixels_per_cell=(4,4),
        cells_per_block=(1,1),
        feature_vector=False
    )

    # reshape to H x W x C
    h, w = gray.shape
    feat = feat.reshape(h//4, w//4, -1)

    feat = cv2.resize(feat, (w, h))

    return feat[:, :, :104]


def extract_gradients(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)

    # orientation bins
    bins = 8
    h, w = gray.shape
    desc = np.zeros((h, w, bins), dtype=np.float32)

    for b in range(bins):
        mask = (ang >= (b*np.pi/bins)) & (ang < ((b+1)*np.pi/bins))
        desc[:, :, b] = mag * mask

    # smooth spatially
    desc = cv2.GaussianBlur(desc, (5,5), 0)

    # expand to 104 dims
    desc = np.repeat(desc, 13, axis=2)[:, :, :104]

    return desc


def extract_dense_sift(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    # dense grid of keypoints
    keypoints = [
        cv2.KeyPoint(x, y, 4)
        for y in range(h)
        for x in range(w)
    ]

    _, descriptors = sift.compute(gray, keypoints)
    # (H*W, 128)

    descriptors = descriptors.reshape(h, w, 128)

    # normalize (important)
    # descriptors /= (np.linalg.norm(descriptors, axis=2, keepdims=True) + 1e-6)

    return descriptors[:, :, :104]  # match your pipeline

# ---- Dummy feature extractor (replace with RoMa/SuperGlue backbone) ----
def extract_daisy(img_exg, cloud_ratio=0.0):
    keypoints = []
    h, w = img_exg.shape[:2]

    # Dense grid: one keypoint per pixel (like your C++ loop)
    for y in range(h):
        for x in range(w):
            keypoints.append(cv2.KeyPoint(float(x), float(y), 1))

    descriptors = daisy.compute(img_exg, keypoints)[1]
    # shape: (H*W, 104)

    # reshape to H x W x 104
    descriptors = descriptors.reshape(h, w, -1)

    return descriptors.astype(np.float32)


def decode_image(b64):
    data = base64.b64decode(b64)
    np_arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def extract_resnet(img, model_name='resnet'):
    H, W, _ = img.shape
    model = get_matching_model(model_name)
    # BGR → RGB (important!)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # normalize
    x = torch.from_numpy(img).float() / 255.0
    x = x.permute(2, 0, 1)[None].to(device)  # (1,3,H,W)

    with torch.no_grad():
        feat = model(x)  # (1,512,H/32,W/32)

    feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

    feat = feat[0].permute(1, 2, 0).cpu().numpy()  # H x W x 512

    # reduce to 104 dims
    feat = feat[:, :, :104]

    return feat.astype(np.float32)



def extract_resnet_multiscale(img):
    H, W, _ = img.shape
    model = get_matching_model('resnet_multiscale')
    layer0, layer1, layer2, layer3, proj = model
    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # to tensor
    x = torch.from_numpy(img).float() / 255.0
    x = x.permute(2, 0, 1)[None].to(device)  # (1,3,H,W)

    with torch.no_grad():
        f0 = layer0(x)
        f1 = layer1(f0)   # (1,64,H/4,W/4)
        f2 = layer2(f1)   # (1,128,H/8,W/8)
        f3 = layer3(f2)   # (1,256,H/16,W/16)

        # upsample both to full res
        f1_up = F.interpolate(f1, size=(H, W), mode="bilinear", align_corners=False)
        f2_up = F.interpolate(f2, size=(H, W), mode="bilinear", align_corners=False)
        f3_up = F.interpolate(f3, size=(H, W), mode="bilinear", align_corners=False)

        # concatenate
        feat = torch.cat([f1_up, f2_up, f3_up], dim=1)  # (1,448,H,W)

        # reduce to 104 channels
        feat = proj(feat)  # (1,104,H,W)

    # to numpy H x W x C
    feat = feat[0].permute(1, 2, 0).cpu().numpy()

    return feat.astype(np.float32)

def extract_dino(img):
    MAX_SIZE = 512
    model = get_matching_model('dino')
    H0, W0 = img.shape[:2]

    # resize
    scale = MAX_SIZE / max(H0, W0)
    if scale < 1:
        img = cv2.resize(img, (int(W0*scale), int(H0*scale)))

    H, W = img.shape[:2]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = torch.from_numpy(img).float() / 255.0
    x = x.permute(2,0,1)[None].to(device)

    with torch.no_grad():
        feat = model.get_intermediate_layers(x, n=1)[0]

    feat = feat[:, 1:]  # remove CLS

    # patch size = 8
    feat = feat.reshape(1, H//8, W//8, -1).permute(0,3,1,2)

    feat = torch.nn.functional.interpolate(
        feat, size=(H, W), mode="bilinear", align_corners=False
    )

    feat = feat[0].permute(1,2,0).cpu().numpy()

    # resize back to original image size
    if scale < 1:
        feat = cv2.resize(feat, (W0, H0))

    return feat[:, :, :104]

# def extract_dino_freeze(img):
#     H, W, _ = img.shape

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     x = torch.from_numpy(img).float() / 255.0
#     x = x.permute(2,0,1)[None].to(device)  # (1,3,H,W)

#     with torch.no_grad():
#         feat = model.get_intermediate_layers(x, n=1)[0]  # tokens

#     feat = feat[:, 1:]  # remove CLS
#     feat = feat.reshape(1, H//8, W//8, -1).permute(0,3,1,2)

#     feat = F.interpolate(feat, size=(H,W), mode="bilinear", align_corners=False)
#     feat = feat[0].permute(1,2,0).cpu().numpy()

#     return feat[:, :, :104]

def extract_superpoint(img):
    model = get_matching_model('superpoint')
    H, W, _ = img.shape

    # BGR → RGB (important!)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # normalize
    x = torch.from_numpy(img).float() / 255.0
    x = x.permute(2, 0, 1)[None].to(device)  # (1,3,H,W)

    with torch.no_grad():
        feats = model.extract(x)

    # ---- dense descriptors ----
    print("Feats keys: ", feats.keys())
    desc = feats["descriptors"]  # (1, 256, H/8, W/8)
    desc = F.interpolate(desc, size=(H, W), mode="bilinear", align_corners=False)

    desc = desc[0].permute(1, 2, 0).cpu().numpy()  # H x W x 256

    # ---- reduce to 104 dims ----
    desc = desc[:, :, :104]   # simplest (fastest)

    return desc.astype(np.float32)
    