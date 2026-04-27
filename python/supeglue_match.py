from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from SuperGluePretrainedNetwork.models.superglue import SuperGlue
from SuperGluePretrainedNetwork.models.utils import (frame2tensor, make_matching_plot_fast)
from SuperGluePretrainedNetwork.models.matching import Matching
import torch


# THIS is to avoid loading the SuperGlue model multiple times if the function is called repeatedly

_MODEL_INSTANCE = None

def get_matching_model(device='cpu'):
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        config = {
        'superpoint': {
            'nms_radius': 2,
            'keypoint_threshold': 0.001,
            'max_keypoints': 2048,
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 50,
            'match_threshold': 0.15,
        }
    }
        # This only runs the very first time the function is called
        _MODEL_INSTANCE = Matching(config).eval().to(device)
    return _MODEL_INSTANCE

def superglue_match(img1, img2, device='cpu'):
    """
    Compute matches between two images using SuperPoint + SuperGlue.

    Returns
    -------
    pts1 : Nx2 numpy array
    pts2 : Nx2 numpy array
    matches : list(cv2.DMatch)
    """

    

    

    # superpoint = SuperPoint(config['superpoint']).eval().to(device)
    # superglue = SuperGlue(config['superglue']).eval().to(device)

    # --- Convert to grayscale ---
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1_tensor = frame2tensor(img1, device)
    img2_tensor = frame2tensor(img2, device)
    
    matching = get_matching_model(device)
    # 4. Perform Matching
    pred = matching({'image0': img1_tensor, 'image1': img2_tensor})
    
    # Extract data to CPU/Numpy
    pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    valid = matches > -1
    mkpts0 = kpts0[valid]           # Coordinates in Image 1
    mkpts1 = kpts1[matches[valid]]  # Corresponding coordinates in Image 2
    
    kp1 = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in mkpts0]
    kp2 = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in mkpts1]
    
    # Create dummy DMatch objects so OpenCV visualization functions work
    # Since SuperGlue already matched them 1-to-1, match index i maps to i
    good_matches = [cv2.DMatch(_imgIdx=0, _queryIdx=i, _trainIdx=i, _distance=0) 
                    for i in range(len(kp1))]
    
    return good_matches, kp1, kp2

