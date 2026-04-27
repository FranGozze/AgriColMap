from romatch import roma_outdoor

def roma_match(img1, img2, device='cpu'):
    """
    Compute matches between two images using RoMa.

    Returns
    -------
    pts1 : Nx2 numpy array
    pts2 : Nx2 numpy array
    matches : list(cv2.DMatch)
    """
    roma_model = roma_outdoor(device=device)
    # Match
    warp, certainty = roma_model.match(img1, img2, device=device)
    # Sample matches for estimation
    matches, certainty = roma_model.sample(warp, certainty)
    # Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
    kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

    return matches, kptsA, kptsB