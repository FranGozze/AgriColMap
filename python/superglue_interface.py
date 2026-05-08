# from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
# from SuperGluePretrainedNetwork.models.superglue import SuperGlue
# from SuperGluePretrainedNetwork.models.utils import (frame2tensor, make_matching_plot)
# from SuperGluePretrainedNetwork.models.matching import Matching
import torch
import torch.nn.functional as F
import cv2

import matplotlib.cm as cm

# THIS is to avoid loading the SuperGlue model multiple times if the function is called repeatedly


_MODEL_INSTANCE = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def get_superpoint_model():
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        # This only runs the very first time the function is called
        # config = {
        #     'nms_radius': 2,
        #     'keypoint_threshold': 0.001,
        #     'max_keypoints': 2048,
        # }

        
        _MODEL_INSTANCE =  SuperPoint(max_num_keypoints=None).eval().to(device)
    return _MODEL_INSTANCE


def superglue_match(img1, img2, device='cpu', output=None):
    """
    Compute matches between two images using SuperPoint + SuperGlue.

    Returns
    -------
    matches : list(cv2.DMatch)
    pts1 : Nx2 numpy array
    pts2 : Nx2 numpy array
    """

    

    

    # superpoint = SuperPoint(config['superpoint']).eval().to(device)
    # superglue = SuperGlue(config['superglue']).eval().to(device)

    if isinstance(img1, str):
        img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    if isinstance(img2, str):
        img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

    cv2.imwrite("debug_img1.jpg", img1)
    cv2.imwrite("debug_img2.jpg", img2)

    img1_tensor = frame2tensor(img1, device)
    img2_tensor = frame2tensor(img2, device)
    
    matching = get_matching_model(device)
    # 4. Perform Matching
    pred = matching({'image0': img1_tensor, 'image1': img2_tensor})
    
    # Extract data to CPU/Numpy
    confidence = pred['matching_scores0'][0].detach().numpy()
    pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    valid = matches > -1
    mkpts0 = kpts0[valid]           # Coordinates in Image 1
    mkpts1 = kpts1[matches[valid]]  # Corresponding coordinates in Image 2
    
    
    color = cm.jet(confidence[valid])
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0))
    ]

    make_matching_plot(img1, img2, kpts0, kpts1, mkpts0, mkpts1, color, text, output)

    kp1 = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in mkpts0]
    kp2 = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in mkpts1]
    
    # Create dummy DMatch objects so OpenCV visualization functions work
    # Since SuperGlue already matched them 1-to-1, match index i maps to i
    good_matches = [cv2.DMatch(_imgIdx=0, _queryIdx=i, _trainIdx=i, _distance=0) 
                    for i in range(len(kp1))]
    
    return good_matches, kp1, kp2

