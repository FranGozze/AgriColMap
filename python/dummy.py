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

    img1 = decode_image(message["img1"])
    img2 = decode_image(message["img2"])
    print("Received matching request. Image shapes:", img1.shape, img2.shape)

    cv2.imwrite(f"received_img1_{img_counter}.jpg", img1)
    cv2.imwrite(f"received_img2_{img_counter}.jpg", img2)
    # matches, kp1, kp2 = superglue_match(img1, img2)
    matches, kp1, kp2 = roma_match(img1, img2, "cpu", output=f"roma_matches_{img_counter}.jpg")
    # pts1, pts2 = dummy_match(img1, img2)

    response = {
        "pts1": kp1.tolist(),
        "pts2": kp2.tolist()
    }

    socket.send_json(response)
    print("Processed a matching request. Matched points:", len(matches), "Mathes:" , matches)
    img_counter += 1