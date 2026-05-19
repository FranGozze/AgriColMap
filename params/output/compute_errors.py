from pathlib import Path
import numpy as np
from output_utils import parse_text_list, scale_from_matrix, scale_matrix, compute_angle, print_metrics

import argparse


def extract_from_file(data):
    A = np.array([
        [data[0], data[1], data[2]],
        [data[4], data[5], data[6]],
        [data[8], data[9], data[10]],
    ], dtype=float)
    t = np.array([data[3], data[7], data[11]], dtype=float)
    s = np.array([data[12], data[13]], dtype=float)
    sn = np.array([data[14], data[15]], dtype=float)
    tn = np.array([data[16], data[17]], dtype=float)
    yn = float(data[18])
    return s, A, t, tn, yn, sn


def extract_from_gt_file(data):
    A = np.array([
        [data[0], data[1], data[2]],
        [data[4], data[5], data[6]],
        [data[8], data[9], data[10]],
    ], dtype=float)
    t = np.array([data[3], data[7], data[11]], dtype=float)
    s_init = np.array([data[12], data[13], 1.0], dtype=float)
    s = scale_from_matrix(A)
    A = scale_matrix(A, s)
    return s, s_init, A, t


    

def main():

    parser = argparse.ArgumentParser(description='Computes registration success rate and error metrics from affine result files.')    
    parser.add_argument('--gt',  default="20180524-mavic-ugv-soybean-eschikon-row5_AffineGroundTruth.txt", help='Ground truth file')
    parser.add_argument('--results',  default="20180524-mavic-ugv-soybean-eschikon-row5", help='Folder containing result files to process')
    parser.add_argument('-o', '--output', help='name of output file plot')    

    args = parser.parse_args()

    

    root = Path(__file__).resolve().parent
    gt_path = root / args.gt
    results_folder = root / args.results
    # row5_path = root / "20180524-mavic-ugv-soybean-eschikon-row5_AffineGroundTruth.txt"
    # list_path = root / "file_list.txt"

    row5_data = np.loadtxt(gt_path)
    # file_list = parse_text_list(results_folder)
    file_list = [f.name for f in results_folder.iterdir() if f.is_file()]

    print(f"Found {len(file_list)} files in results folder.")

    s_gt5, s_init5, Aff_gt5, t_gt5 = extract_from_gt_file(row5_data)

    counter = 0
    succ_number = 0

    grouped_by = {}

    for file_name in file_list:        

        current_path = results_folder / file_name
        if not current_path.exists():
            print(f"Skipping missing file: {file_name}")
            continue
        partitions = str(current_path).split("_")
        idx = partitions[2]
        scale_noise_magnitude = partitions[3]
        transl_noise_magnitude = partitions[4]
        yaw_noise_magnitude = partitions[5]

        curr_file = np.loadtxt(current_path)
        s, Aff, t, tn, yn, sn = extract_from_file(curr_file)

        s = scale_from_matrix(Aff)
        Aff = scale_matrix(Aff, s)
        yn_rad = yn * (3.14 / 180.0)
        Rnorm = np.array([
            [np.cos(yn_rad), -np.sin(yn_rad), 0.0],
            [np.sin(yn_rad), np.cos(yn_rad), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)
        Affn = Rnorm @ Aff

        # if np.linalg.norm(sn - 1.0) < 0.19 or np.linalg.norm(sn - 1.0) > 0.21 or yn > 6 or yn < 4:
        #     print(f"Skipping file {file_name} due to invalid scale or yaw: sn={sn}, yn={yn}. sn should be close to 1.0 and yn should be around 5 degrees.")
        #     print(f"linalg.norm(sn - 1.0) = {np.linalg.norm(sn - 1.0)}")
        #     continue

        diff_Aff = np.linalg.solve(Affn, Aff_gt5)
        s_scl = np.array([s[0] * sn[0], s[1] * sn[1]], dtype=float)

        angle_err = max(0.005, compute_angle(diff_Aff))
        scale_err = max(0.005, np.linalg.norm(s_scl - s_gt5[:2]))
        transl_err = max(0.005, np.linalg.norm(t - t_gt5))

        if abs(angle_err) <= 0.2 and abs(scale_err) <= 2.5 and abs(transl_err) <= 0.1:
            succ_number += 1
            # print(f"Successful registration case: {file_name}")
            # print(f"transl_err: {transl_err:.4f}, angle_err: {angle_err}, scale_err: {(scale_err*100):.4f} % ")
            if scale_noise_magnitude not in grouped_by:
                grouped_by[scale_noise_magnitude] = {}
            method = partitions[6].split(".")[0]
            if method not in grouped_by[scale_noise_magnitude]:
                grouped_by[scale_noise_magnitude][method] = {
                    "transl_err": [],
                    "angle_err": [],
                    "scale_err": []
                }
            grouped_by[scale_noise_magnitude][method]["transl_err"].append(transl_err)
            grouped_by[scale_noise_magnitude][method]["angle_err"].append(angle_err)
            grouped_by[scale_noise_magnitude][method]["scale_err"].append(scale_err)
            
            
        else:
            print(f"Failed registration case: {file_name}")
            # print("Aff:", Aff)
            # print("Aff_gt5:", Aff_gt5)
            print(f" transl_err: {transl_err}, angle_err: {angle_err}, scale_err: {scale_err}")

        counter += 1

    print(f"Processed: {counter}")
    if counter > 0:
        print_metrics(grouped_by)
        print(f"Success ratio: {succ_number / counter:.6f}")
    else:
        print("No valid cases processed.")


if __name__ == "__main__":
    main()
