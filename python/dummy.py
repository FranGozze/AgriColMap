import zmq
import numpy as np
import cv2
import base64
import json

import struct

import utils_methods as utils

# ---- Replace this with RoMa / SuperGlue ----
def dummy_match(img1, img2):
    # Fake matcher (replace with real model)
    h, w = img1.shape[:2]
    pts1 = np.array([[10, 10], [100, 100]], dtype=float)
    pts2 = np.array([[12, 12], [98, 102]], dtype=float)
    return pts1, pts2

# -------------------------------------------


# -----------------------------------------------------------------------



# List of extract functions:
# - extract_daisy
# - extract_superpoint      -- Doesn't work!!
# - extract_resnet_multiscale
# - extract_resnet (single scale, for testing)
# - extract_dino            -- Doesn't work!!
# - extract_gradients       -- Has been tested and works, but the results are bad
# - extract_hog             -- Doesn't work!!
# - extract_dense_sift


methods = [utils.extract_daisy, utils.extract_resnet, utils.extract_dense_sift, utils.extract_dino_single_image,
    utils.extract_gabor_hog, utils.extract_lss, utils.extract_gabor_lss, utils.extract_hog, utils.extract_gabor_only, utils.extract_gabor_hog2, utils.extract_gabor_hog2_lss, utils.extract_resnet_multiscale,
    utils.extract_daisy_gabor, utils.extract_daisy_hog, utils.extract_daisy_gabor_hog, utils.extract_fpfh
    ]

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
    id_method = message["id_method"]
    # cv2.imwrite(f"imgs/received_img_{img_counter}_exg.jpg", img_exg)
    # cv2.imwrite(f"imgs/received_img_{img_counter}_elev.jpg", img_elev)
    # img_exg = cv2.imread("imgs/received_img_0_exg.jpg", cv2.IMREAD_COLOR)  # For testing without ZMQ
    # id_method = 3  # For testing without ZMQ
    print(f"method: {id_method}")
    # if id_method < 0 or id_method >= len(methods):
    #     print(f"Invalid method ID: {id_method}. Using default method (0).")
    #     id_method = 0
    if id_method < 15:
        feat_exg = methods[id_method](img_exg)
        # print("Extracted features: ", feat_exg)

        H, W = img_exg.shape[:2]
        feat_elev = np.zeros((H, W, 33), dtype=np.float32)
    elif id_method >= 15 and id_method <= 17:
        feat_exg = methods[0](img_exg)
        if id_method == 15:
            feat_elev = methods[8](img_elev)
        elif id_method == 16:
            feat_elev = methods[7](img_elev)
        elif id_method == 17:
            feat_elev = methods[4](img_elev)
    elif id_method >= 18 and id_method <= 20:
        feat_exg = methods[0](img_exg)
        feat_elev = methods[id_method - 6](img_elev)
    elif id_method >= 21 and id_method <= 23:
        feat_exg = methods[id_method - 9](img_exg)
        feat_elev = methods[id_method - 9](img_elev)
    elif id_method == 24:
        feat_exg = methods[0](img_exg)
        feat_elev = methods[15](img_elev, cloudRatio)


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