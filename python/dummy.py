import zmq
import numpy as np
import cv2
import base64
import json

from supeglue_match import superglue_match
from roma_match import roma_match

import struct

# ---- Replace this with RoMa / SuperGlue ----
def dummy_match(img1, img2):
    # Fake matcher (replace with real model)
    h, w = img1.shape[:2]
    pts1 = np.array([[10, 10], [100, 100]], dtype=float)
    pts2 = np.array([[12, 12], [98, 102]], dtype=float)
    return pts1, pts2

# -------------------------------------------

daisy = cv2.xfeatures2d.DAISY_create(
    radius=5,
    q_radius=3,
    q_theta=4,
    q_hist=8,
    norm=cv2.xfeatures2d.DAISY_NRM_FULL,
    interpolation=False,
    use_orientation=False
)


# ---- Dummy feature extractor (replace with RoMa/SuperGlue backbone) ----
def extract_features(img_exg, cloud_ratio=0.0):
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

# -----------------------------------------------------------------------

def decode_image(b64):
    data = base64.b64decode(b64)
    np_arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

print("Python matcher server running...")



img_counter = 0


while True:
    message = socket.recv_json()

    img_exg = decode_image(message["img_exg"])
    img_elev = decode_image(message["img_elev"])
    cloudRatio = message["cloud_ratio"]
    # cv2.imwrite(f"received_img_{img_counter}.jpg", img_exg)
    
    # print("Extracted features: ", feat_exg)

    H, W = img_exg.shape[:2]
    feat_elev = np.zeros((H, W, 33), dtype=np.float32)

    # ---- OPTIONAL: convert to uint8 here (faster) ----
    feat_exg = np.clip(feat_exg * 255, 0, 255).astype(np.uint8)
    feat_elev = feat_elev.astype(np.uint8)

    H, W, C1 = feat_exg.shape
    _, _, C2 = feat_elev.shape

    header = struct.pack("6i", H, W, C1, H, W, C2)

    socket.send_multipart([
        header,
        feat_exg.tobytes(),
        feat_elev.tobytes()
    ])

    print("Processed a matching request. Features shape:", feat_exg.shape, feat_elev.shape)
    img_counter += 1