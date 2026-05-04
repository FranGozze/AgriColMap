import zmq
import numpy as np
import cv2
import base64
import json

from supeglue_match import superglue_match
from roma_match import roma_match

# ---- Replace this with RoMa / SuperGlue ----
def dummy_match(img1, img2):
    # Fake matcher (replace with real model)
    h, w = img1.shape[:2]
    pts1 = np.array([[10, 10], [100, 100]], dtype=float)
    pts2 = np.array([[12, 12], [98, 102]], dtype=float)
    return pts1, pts2

# -------------------------------------------

# ---- Dummy feature extractor (replace with RoMa/SuperGlue backbone) ----
def extract_features(img_exg, img_elev, cloud_ratio):
    H, W = img_exg.shape[:2]

    # Replace with real model
    feat_exg = np.random.rand(H, W, 104).astype(np.float32)
    feat_elev = np.random.rand(H, W, 33).astype(np.float32)

    return feat_exg, feat_elev
# -----------------------------------------------------------------------

def decode_image(b64):
    data = base64.b64decode(b64)
    np_arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

print("Python matcher server running...")

# matches, kp1, kp2 = superglue_match("received_img1_exg.jpg", "received_img2_exg.jpg", output="superglue_matches_exg.jpg")
# matches, kp1, kp2 = roma_match("received_img1_exg.jpg", "received_img2_exg.jpg", output="roma_matches_exg.jpg")
# print("Test run of SuperGlue matcher. Matched points:", len(matches), "Matches:", matches)



img_counter = 0

while True:
    message = socket.recv_json()

    img_exg = decode_image(message["img_exg"])
    img_elev = decode_image(message["img_elev"])
    cloudRatio = message["cloud_ratio"]

    print("Received matching request. Image shapes:", img_exg.shape, img_elev.shape, "C:", cloudRatio)

    # cv2.imwrite(f"received_img_{img_counter}.jpg", img)
    # cv2.imwrite(f"received_img2_{img_counter}.jpg", img2)
    # matches, kp1, kp2 = superglue_match(img, img2)
    # matches, kp1, kp2 = roma_match(img, img2, "cpu", output=f"roma_matches_{img_counter}.jpg")
    # pts1, pts2 = dummy_match(img, img2)

    features_exg, features_elev = extract_features(img_exg, img_elev, cloudRatio)

    response = {
    "shape_exg": [img_exg.shape[0], img_exg.shape[1], 104],
    "data_exg": features_exg.flatten().tolist(),
    "shape_elev": [img_elev.shape[0], img_elev.shape[1], 33],
    "data_elev": features_elev.flatten().tolist()
    }

    socket.send_json(response)
    print("Processed a matching request. Features shape:", features_exg.shape, features_elev.shape)
    img_counter += 1