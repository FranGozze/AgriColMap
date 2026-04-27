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

# context = zmq.Context()
# socket = context.socket(zmq.REP)
# socket.bind("tcp://*:5555")

print("Python matcher server running...")

matches, kp1, kp2 = roma_match("received_img1.jpg", "received_img2.jpg", "gpu")
print("Test run of RoMa matcher. Matched points:", len(matches), "Matches:", matches)

# while True:
#     message = socket.recv_json()

#     img1 = decode_image(message["img1"])
#     img2 = decode_image(message["img2"])
#     print("Received matching request. Image shapes:", img1.shape, img2.shape)

#     cv2.imwrite("received_img1.jpg", img1)
#     cv2.imwrite("received_img2.jpg", img2)
#     # matches, kp1, kp2 = superglue_match(img1, img2)
#     matches, kp1, kp2 = roma_match(img1, img2)
#     # pts1, pts2 = dummy_match(img1, img2)

#     # response = {
#     #     "pts1": pts1.tolist(),
#     #     "pts2": pts2.tolist()
#     # }

#     # socket.send_json(response)
#     print("Processed a matching request. Matched points:", len(matches), "Mathes:" , matches)