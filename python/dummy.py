import zmq
import numpy as np
import cv2
import base64
import json

import struct

import utils

# ---- Replace this with RoMa / SuperGlue ----
def dummy_match(img1, img2):
    # Fake matcher (replace with real model)
    h, w = img1.shape[:2]
    pts1 = np.array([[10, 10], [100, 100]], dtype=float)
    pts2 = np.array([[12, 12], [98, 102]], dtype=float)
    return pts1, pts2

# -------------------------------------------


# -----------------------------------------------------------------------



context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

print("Python matcher server running...")


img_counter = 0


while True:
# if True:
    message = socket.recv_json()

    img_exg = utils.decode_image(message["img_exg"])
    img_elev = utils.decode_image(message["img_elev"])
    cloudRatio = message["cloud_ratio"]
    # cv2.imwrite(f"imgs/received_img_{img_counter}.jpg", img_exg)
    # img_exg = cv2.imread("test_img_exg.jpg", cv2.IMREAD_COLOR)  # For testing without ZMQ

    # feat_exg = dino_interface.extract_features(img_exg)    
    feat_exg = utils.extract_dense_sift(img_exg)
    # feat_exg = resnet_interface.extract_multiscale(img_exg)
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