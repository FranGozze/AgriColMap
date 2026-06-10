from pathlib import Path
import numpy as np
from output_utils import parse_text_list, scale_from_matrix, scale_matrix, compute_angle, print_metrics

import argparse

debug = False
def printDebugInfo(message):
    if debug:
        print(message)

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


def process_folder(gt_path, results_folder, rowNumber=None, transl_noise_filter=None, scale_noise_filter=None, yaw_noise_filter=None):
    row5_data = np.loadtxt(gt_path)
    # file_list = parse_text_list(results_folder)
    file_list = [f.name for f in results_folder.iterdir() if f.is_file()]

    print(f"Found {len(file_list)} files in results folder.")

    s_gt5, s_init5, Aff_gt5, t_gt5 = extract_from_gt_file(row5_data)

    if rowNumber is None:
        rowNumber = gt_path.split("_")[0][-1]  # Extract row number from GT file name
    counter = 0
    succ_number = 0

    grouped_by = {}



    for file_name in file_list:        

        current_path = results_folder / file_name
        if not current_path.exists():
            print(f"Skipping missing file: {file_name}")
            continue
        partitions = str(file_name).split("_")
        # print(f"Processing file: {file_name} with partitions: {partitions}")
        scale_noise_magnitude = partitions[2]
        transl_noise_magnitude = partitions[3]
        yaw_noise_magnitude = partitions[4]
        method = partitions[5]
        idx = partitions[6].split(".")[0]

        if transl_noise_filter is not None and float(transl_noise_magnitude) != float(transl_noise_filter):
            printDebugInfo(f"Skipping file {file_name} due to translational noise filter: {transl_noise_filter}")
            continue

        if scale_noise_filter is not None and float(scale_noise_magnitude) != float(scale_noise_filter):
            printDebugInfo(f"Skipping file {file_name} due to scale noise filter: {scale_noise_filter}")
            continue

        if yaw_noise_filter is not None and float(yaw_noise_magnitude) != float(yaw_noise_filter):
            printDebugInfo(f"Skipping file {file_name} due to yaw noise filter: {yaw_noise_filter}")
            continue

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
        #     print(f"Skipping file {scale_noise_magnitude} {method} linalg.norm(sn - 1.0) = {np.linalg.norm(sn - 1.0)}, yn={yn}.")
        #     continue

        diff_Aff = np.linalg.solve(Affn, Aff_gt5)
        s_scl = np.array([s[0] * sn[0], s[1] * sn[1]], dtype=float)

        angle_err = max(0.005, compute_angle(diff_Aff))
        scale_err = max(0.005, np.linalg.norm(s_scl - s_gt5[:2]))
        transl_err = max(0.005, np.linalg.norm(t - t_gt5))

        if abs(transl_err) <= 0.1 and abs(angle_err) <= 0.2 and abs(scale_err) <= 2.5:
        # if True:
            succ_number += 1
            # print(f"Successful registration case: {file_name}")
            # print(f"transl_err: {transl_err:.4f}, angle_err: {angle_err}, scale_err: {(scale_err*100):.4f} % ")
            if scale_noise_magnitude not in grouped_by:
                grouped_by[scale_noise_magnitude] = {}
            if transl_noise_magnitude not in grouped_by[scale_noise_magnitude]:
                grouped_by[scale_noise_magnitude][transl_noise_magnitude] = {}
            if yaw_noise_magnitude not in grouped_by[scale_noise_magnitude][transl_noise_magnitude]:
                grouped_by[scale_noise_magnitude][transl_noise_magnitude][yaw_noise_magnitude] = {}
            if method not in grouped_by[scale_noise_magnitude][transl_noise_magnitude][yaw_noise_magnitude]:
                grouped_by[scale_noise_magnitude][transl_noise_magnitude][yaw_noise_magnitude][method] = {
                    "transl_err": [],
                    "angle_err": [],
                    "scale_err": []
                }
            grouped_by[scale_noise_magnitude][transl_noise_magnitude][yaw_noise_magnitude][method]["transl_err"].append(abs(transl_err))
            grouped_by[scale_noise_magnitude][transl_noise_magnitude][yaw_noise_magnitude][method]["angle_err"].append(abs(angle_err))
            grouped_by[scale_noise_magnitude][transl_noise_magnitude][yaw_noise_magnitude][method]["scale_err"].append(abs(scale_err))
            
            
        else:            
            # print("Aff:", Aff)            
            # print("Aff_gt5:", Aff_gt5)
            fail_reasons = f" transl_err: {transl_err}" if abs(transl_err) > 0.05 else ""
            fail_reasons += f" angle_err: {angle_err}" if abs(angle_err) > 0.1 else ""
            fail_reasons += f" scale_err: {scale_err}" if abs(scale_err) > 2.5 else ""
            printDebugInfo(f"Failed registration case: {file_name} ({fail_reasons})")

        counter += 1

    print(f"Processed: {counter}")
    if succ_number > 0:
        print_metrics(rowNumber,grouped_by)
        print(f"Success ratio: {succ_number / counter * 100:.6f} %. Failed cases: {counter - succ_number}.")
        print(f"Max transl err: {max([max(grouped_by[scale][transl][yaw][method]['transl_err']) for scale in grouped_by for transl in grouped_by[scale] for yaw in grouped_by[scale][transl] for method in grouped_by[scale][transl][yaw]])}")
        print(f"Max angle err: {max([max(grouped_by[scale][transl][yaw][method]['angle_err']) for scale in grouped_by for transl in grouped_by[scale] for yaw in grouped_by[scale][transl] for method in grouped_by[scale][transl][yaw]])}")
        print(f"Max scale err: {max([max(grouped_by[scale][transl][yaw][method]['scale_err']) for scale in grouped_by for transl in grouped_by[scale] for yaw in grouped_by[scale][transl] for method in grouped_by[scale][transl][yaw]])}")
    else:
        print("No valid cases processed.")

def main():

    parser = argparse.ArgumentParser(description='Computes registration success rate and error metrics from affine result files.')    
    parser.add_argument('--gt',  default="20180524-mavic-ugv-soybean-eschikon-row3_AffineGroundTruth.txt", help='Ground truth file')
    parser.add_argument('--results',  default="20180524-mavic-ugv-soybean-eschikon-row3", help='Folder containing result files to process')
    parser.add_argument('-c', '--complete', action="store_true", help="show all the soybean rows")
    parser.add_argument('-o', '--output', help='name of output file plot')
    parser.add_argument('--show_plots', action="store_true", help='TODO: whether to show the plots after processing')
    parser.add_argument('--save_plots', action="store_true", help='TODO: whether to save the plots after processing')
    parser.add_argument('-t', '--filter_transl_noise',  default=None, help='filter out cases without translational noise selected')
    parser.add_argument('-s', '--filter_scale_noise',  default=None, help='filter out cases without scale noise selected')
    parser.add_argument('-y', '--filter_yaw_noise',  default=None, help='filter out cases without yaw noise selected')
    parser.add_argument('-d', '--debug', action="store_true", help='enable debug output')

    args = parser.parse_args()
    global debug 
    debug = args.debug
    root = Path(__file__).resolve().parent
    if args.complete:
        for x in [3,4,5]:
            gt_file = f"20180524-mavic-ugv-soybean-eschikon-row{x}_AffineGroundTruth.txt"
            results_folder = root / f"20180524-mavic-ugv-soybean-eschikon-row{x}"
            print(f"\nProcessing row {x}...")
            process_folder(gt_file, results_folder, rowNumber=x, transl_noise_filter=args.filter_transl_noise, scale_noise_filter=args.filter_scale_noise, yaw_noise_filter=args.filter_yaw_noise)
    else:
        
        rowNumber = args.gt.split("_")[0][-1]  # Extract row number from GT file name
        gt_path = root / args.gt
        results_folder = root / args.results
        print(f"Processing GT: {gt_path} with results from folder: {results_folder}...")
        process_folder(gt_path, results_folder, rowNumber=rowNumber, transl_noise_filter=args.filter_transl_noise, scale_noise_filter=args.filter_scale_noise, yaw_noise_filter=args.filter_yaw_noise)
        # row5_path = root / "20180524-mavic-ugv-soybean-eschikon-row5_AffineGroundTruth.txt"
        # list_path = root / "file_list.txt"

    


if __name__ == "__main__":
    main()
