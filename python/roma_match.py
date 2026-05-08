from romatch import roma_outdoor
from PIL import Image
from romatch.utils.utils import tensor_to_pil
import cv2
import numpy as np

_MODEL_INSTANCE = None



def show_matches(img1, img2, pts1, pts2, output=None):
    # Create a visualization of the matches
    draw_img = cv2.drawMatches(img1, [cv2.KeyPoint(int(pt[0]), int(pt[1]),1) for pt in pts1],
                               img2, [cv2.KeyPoint(int(pt[0]), int(pt[1]),1) for pt in pts2],
                               [cv2.DMatch(i, i, 0) for i in range(len(pts1))], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if output:
        cv2.imwrite(output, draw_img)
    else:
        cv2.imshow("Matches", draw_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def roma_match(img1, img2, device='cpu', output=None):
    """
    Compute matches between two images using RoMa.

    Returns
    -------
    pts1 : Nx2 numpy array
    pts2 : Nx2 numpy array
    matches : list(cv2.DMatch)
    """

    roma_model = get_matching_model(device)

    if isinstance(img1, str):
        W_A, H_A = Image.open(img1).size
        W_B, H_B = Image.open(img2).size
    else: # If img1 and img2 are already loaded as numpy arrays, get their shapes
        img1_converted = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_converted = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # 3. Convert NumPy array to PIL Image
        img1 = Image.fromarray(img1_converted)
        img2 = Image.fromarray(img2_converted)
        W_A, H_A = img1.size
        W_B, H_B = img2.size
    # Match
    warp, certainty = roma_model.match(img1, img2, device=device)
    # Sample matches for estimation
    matches, certainty = roma_model.sample(warp, certainty)
    # Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
    kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

    # print("RoMa matched to pixel coordinates: ", roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B))
    matched_pointA, matched_pointB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

    show_matches(np.array(img1), np.array(img2), matched_pointA, matched_pointB, output) 
    # H, W = roma_model.get_output_resolution()

    # im1 = Image.open(img1).resize((W, H))
    # im2 = Image.open(img2).resize((W, H))
    # warp, certainty = roma_model.match(im1, im2, device=device)

    # x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    # x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)

    # im2_transfer_rgb = F.grid_sample(
    # x2[None], warp[:, :, :W, 2:], mode="bilinear", align_corners=False
    # )[0]
    # im1_transfer_rgb = F.grid_sample(
    # x1[None], warp[:, :, W:, :2], mode="bilinear", align_corners=False
    # )[0]
    # warp_im = torch.cat((im2_transfer_rgb,im1_transfer_rgb),dim=2)
    # white_im = torch.ones((H,2*W),device=device)
    # vis_im = certainty * warp_im + (1 - certainty) * white_im
    # tensor_to_pil(vis_im, unnormalize=False).save(output)

    return matches, matched_pointA, matched_pointB