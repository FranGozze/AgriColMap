from skimage.feature import hog
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from sklearn.decomposition import PCA

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
            _MODEL_INSTANCE = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').eval().to(device)
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

def compress_features_to_104(feat_src, feat_tgt):
    """
    Compresses dense descriptors from 384 channels to 104 channels using PCA.
    Expects arrays of shape (H, W, 384)
    """
    H, W, C = feat_src.shape
    
    # Flatten spatial dimensions
    feat_src_flat = feat_src.reshape(-1, C)
    feat_tgt_flat = feat_tgt.reshape(-1, C)
    
    # Combine to learn the joint feature space
    combined = np.vstack([feat_src_flat, feat_tgt_flat])
    
    # Fit PCA to reduce from 384 -> 104
    pca = PCA(n_components=104, random_state=42)
    combined_compressed = pca.fit_transform(combined)
    
    # Separate back and reshape to original spatial sizes
    feat_src_104 = combined_compressed[:H*W].reshape(H, W, 104)
    feat_tgt_104 = combined_compressed[H*W:].reshape(H, W, 104)
    
    # Re-normalize to unit length for the matching layers
    feat_src_104 /= (np.linalg.norm(feat_src_104, axis=-1, keepdims=True) + 1e-8)
    feat_tgt_104 /= (np.linalg.norm(feat_tgt_104, axis=-1, keepdims=True) + 1e-8)
    
    return feat_src_104, feat_tgt_104


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
    print("DINO raw feature shape:", feat.shape)
    return feat


def extract_dino_single_image(img):
    """
    Extracts DINO features from a single image and reduces channels to 104
    without destroying values or using multi-image dependencies.
    """
    PATCH_SIZE = 8
    MAX_SIZE = 512
    model = get_matching_model('dino')
    # model.patch_embed.proj.stride = (4, 4)
    H0, W0 = img.shape[:2]

    # 1. Handle scaling down if image is too large
    scale = MAX_SIZE / max(H0, W0)
    if scale < 1:
        img = cv2.resize(img, (int(W0 * scale), int(H0 * scale)))

    # 2. Force dimensions to be strict multiples of the ViT patch size (8)
    H, W = img.shape[:2]
    H_new = (H // PATCH_SIZE) * PATCH_SIZE
    W_new = (W // PATCH_SIZE) * PATCH_SIZE
    if H != H_new or W != W_new:
        img = cv2.resize(img, (W_new, H_new))
        H, W = H_new, W_new

    # 3. Convert format and move to GPU
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img_rgb).float() / 255.0
    x = x.permute(2, 0, 1)[None].to(device)

    # 4. Extract raw features (No normalization layer)
    with torch.no_grad():
        # This was originally
        # feat = model.get_intermediate_layers(x, n=12)[1]

        all_layers = model.get_intermediate_layers(x, n=12)
        
        # Grab a shallow layer (index 1), a lower-mid layer (index 4), and a mid layer (index 7)
        # We skip the very last semantic layers to stay DAISY-compatible
        layer_shallow = all_layers[1][:, 1:] # Drop CLS
        layer_mid1    = all_layers[4][:, 1:]
        layer_mid2    = all_layers[7][:, 1:]
        
        # Interleave channels from all three layers to capture multi-scale context
        # Shape becomes [1, Patches, 384 * 3] -> [1, Patches, 1152]
        feat = torch.cat([layer_shallow, layer_mid1, layer_mid2], dim=-1)

        
    
    # Remove the CLS token
    # feat = feat[:, 1:]  

    # 5. Reshape back into a spatial grid mapping
    num_patches_h = H // PATCH_SIZE
    num_patches_w = W // PATCH_SIZE
    feat = feat.reshape(1, num_patches_h, num_patches_w, -1).permute(0, 3, 1, 2)

    # 6. Upsample feature map back to processed image size
    # Bilinear avoids generating artificial negative values that Bicubic can introduce
    feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
    
    # Convert tensor to numpy array: shape becomes (H, W, 384)
    feat = feat[0].permute(1, 2, 0).cpu().numpy()

    # 7. CRITICAL: Channel reduction to 104 via strided slicing
    # Instead of taking the first 104 channels [:104], we skip by 3s (::3) 
    # to sample features evenly across the whole 384 spectrum.
    print("DINO raw feature shape:", feat[:, :, ::3].shape)
    feat_104 = feat[:, :, ::3][:, :, :104]  # Now shape is (H, W, 104)

    # 7. CRITICAL MATING CONSTRAINTS FOR DAISY REPLACEMENT:
    # A) Convert to strictly non-negative values (Mimicking gradient magnitudes)
    feat_104 = np.abs(feat_104) 

    # # B) Min-Max scale each channel individually to range [0, 1] 
    # # This forces DINO channels to behave exactly like DAISY gradient bins.
    # f_min = feat_104.min(axis=(0, 1), keepdims=True)
    # f_max = feat_104.max(axis=(0, 1), keepdims=True)
    # feat_104 = (feat_104 - f_min) / (f_max - f_min + 1e-8)

    # B) Local L2 Normalization per-pixel across the channel axis
    local_norm = np.linalg.norm(feat_104, axis=-1, keepdims=True)
    feat_104 = feat_104 / (local_norm + 1e-8)
    
    # C) Clip values to 0.2 (Stops illumination/gradient spikes from dominating your matching space)
    feat_104 = np.clip(feat_104, 0, 0.2)
    
    # D) Final re-normalization so the descriptor vector equals unit length 1.0
    local_norm = np.linalg.norm(feat_104, axis=-1, keepdims=True)
    feat_104 = feat_104 / (local_norm + 1e-8)


    # 8. Scale back to match original image input dimensions
    if scale < 1 or H0 != H or W0 != W:
        feat_104 = cv2.resize(feat_104, (W0, H0), interpolation=cv2.INTER_LINEAR)
    
    
    return feat_104


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


def extract_gabor(img,
                  num_orientations=8,
                  sigmas=(2, 4, 8),
                  lambdas=(4, 8, 16)):
    """
    Returns:
        H x W x C float32
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray /= 255.0

    features = []

    for sigma in sigmas:
        for theta in np.linspace(0, np.pi, num_orientations, endpoint=False):
            for lambd in lambdas:

                kernel = cv2.getGaborKernel(
                    (21, 21),
                    sigma,
                    theta,
                    lambd,
                    gamma=0.5,
                    psi=0,
                    ktype=cv2.CV_32F
                )

                response = cv2.filter2D(gray, cv2.CV_32F, kernel)
                features.append(response)

    feat = np.stack(features, axis=2)

    norm = np.linalg.norm(feat, axis=2, keepdims=True)
    feat /= (norm + 1e-8)

    return feat.astype(np.float32)

def extract_dense_hog(img,
                      cell_size=4,
                      num_bins=9):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag, ang = cv2.cartToPolar(gx, gy)

    h, w = gray.shape

    feat = np.zeros((h, w, num_bins), dtype=np.float32)

    bin_width = 2 * np.pi / num_bins

    for b in range(num_bins):
        low = b * bin_width
        high = (b + 1) * bin_width

        mask = ((ang >= low) & (ang < high))
        feat[:, :, b] = mag * mask

    for c in range(num_bins):
        feat[:, :, c] = cv2.GaussianBlur(
            feat[:, :, c],
            (cell_size * 2 + 1, cell_size * 2 + 1),
            0
        )

    norm = np.linalg.norm(feat, axis=2, keepdims=True)
    feat /= (norm + 1e-8)

    return feat.astype(np.float32)

def extract_gabor_only(img):
    return extract_gabor(img, num_orientations=13, sigmas=(2,4,8,12), lambdas=(4,8))

def extract_hog(img):
    feat = extract_dense_hog(img, num_bins=104)
    return feat.astype(np.float32)

def extract_gabor_hog(img):
    gabor_feat = extract_gabor(img)
    hog_feat = extract_dense_hog(img, num_bins=32)

    feat = np.concatenate([gabor_feat, hog_feat], axis=2)

    # norm = np.linalg.norm(feat, axis=2, keepdims=True)
    # feat /= (norm + 1e-8)

    return feat.astype(np.float32)

def extract_gabor_hog2(img):
    gabor_feat = extract_gabor(img, num_orientations=13, sigmas=(2,4,8,12), lambdas=(4,8))
    hog_feat = extract_dense_hog(img, num_bins=32)

    feat = np.concatenate([gabor_feat, hog_feat], axis=2)

    # norm = np.linalg.norm(feat, axis=2, keepdims=True)
    # feat /= (norm + 1e-8)

    return feat.astype(np.float32)

def extract_lss(img,
                patch_radius=2,
                search_radius=5):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    h, w = gray.shape

    channels = []

    center = gray

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):

            if dx == 0 and dy == 0:
                continue

            shifted = np.roll(center, (dy, dx), axis=(0, 1))

            diff = (center - shifted) ** 2

            sim = cv2.GaussianBlur(
                diff,
                (2 * patch_radius + 1,
                 2 * patch_radius + 1),
                0
            )

            sim = np.exp(-sim / (sim.mean() + 1e-6))

            channels.append(sim)

    feat = np.stack(channels, axis=2)

    norm = np.linalg.norm(feat, axis=2, keepdims=True)
    feat /= (norm + 1e-8)
    feat = feat

    return feat.astype(np.float32)

def extract_gabor_lss(img):
    gabor_feat = extract_gabor(img)
    lss_feat = extract_lss(img, search_radius=3)

    feat = np.concatenate([gabor_feat, lss_feat], axis=2)

    # norm = np.linalg.norm(feat, axis=2, keepdims=True)
    # feat /= (norm + 1e-8)
    feat = feat[:, :, :104]
    return feat.astype(np.float32)

def extract_gabor_hog2_lss(img):
    gabor_feat = extract_gabor(img, num_orientations=13, sigmas=(2,4,8,12), lambdas=(4,8))
    hog_feat = extract_dense_hog(img, num_bins=32)
    lss_feat = extract_lss(img)

    feat = np.concatenate([gabor_feat, hog_feat, lss_feat], axis=2)

    # norm = np.linalg.norm(feat, axis=2, keepdims=True)
    # feat /= (norm + 1e-8)

    return feat.astype(np.float32)

def extract_daisy_gabor(img):
    daisy_feat = extract_daisy(img)
    gabor_feat = extract_gabor(img)

    feat = np.concatenate([daisy_feat, gabor_feat], axis=2)

    # norm = np.linalg.norm(feat, axis=2, keepdims=True)
    # feat /= (norm + 1e-8)

    return feat.astype(np.float32)

def extract_daisy_hog(img):
    daisy_feat = extract_daisy(img)
    hog_feat = extract_dense_hog(img, num_bins=32)

    feat = np.concatenate([daisy_feat, hog_feat], axis=2)

    # norm = np.linalg.norm(feat, axis=2, keepdims=True)
    # feat /= (norm + 1e-8)

    return feat.astype(np.float32)

def extract_daisy_gabor_hog(img):
    daisy_feat = extract_daisy(img)
    gabor_feat = extract_gabor(img)
    hog_feat = extract_dense_hog(img, num_bins=32)

    feat = np.concatenate([daisy_feat, gabor_feat, hog_feat], axis=2)

    # norm = np.linalg.norm(feat, axis=2, keepdims=True)
    # feat /= (norm + 1e-8)

    return feat.astype(np.float32)

def extract_lbp(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    lbp = np.zeros((h, w), dtype=np.uint8)

    center = gray[1:-1, 1:-1]

    offsets = [
        (-1,-1), (-1,0), (-1,1),
        (0,1),
        (1,1), (1,0), (1,-1),
        (0,-1)
    ]

    for bit, (dy, dx) in enumerate(offsets):
        neighbor = gray[
            1+dy:h-1+dy,
            1+dx:w-1+dx
        ]

        lbp[1:-1,1:-1] |= np.uint8(neighbor > center) << bit

    feat = np.zeros((h, w, 256), dtype=np.float32)

    for i in range(256):
        feat[:, :, i] = (lbp == i)

    return feat