from pathlib import Path
import numpy as np
from output_utils import parse_text_list, scale_from_matrix, print_metrics
import argparse


def parse_affine_result(data):
    A = np.array([
        [data[0], data[1], data[2]],
        [data[4], data[5], data[6]],
        [data[8], data[9], data[10]],
    ], dtype=float)
    t = np.array([data[3], data[7], data[11]], dtype=float)
    return A, t


def compute_metrics(A, t, xy_err, yaw_err, xy_scl_err, scale_mag_noise):
    scl_x = np.linalg.norm(A[:, 0])
    scl_y = np.linalg.norm(A[:, 1])
    scl_z = np.linalg.norm(A[:, 2])
    R = np.array([
        [A[0, 0] / scl_x, A[0, 1] / scl_y, A[0, 2] / scl_z],
        [A[1, 0] / scl_x, A[1, 1] / scl_y, A[1, 2] / scl_z],
        [A[2, 0] / scl_x, A[2, 1] / scl_y, A[2, 2] / scl_z],
    ], dtype=float)
    tr = np.trace(R)
    rot_err = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    trans_err = np.linalg.norm(t[:2])
    scl_err = np.sqrt(scl_x * scl_x + scl_y * scl_y)
    xy_err_mag = np.linalg.norm(xy_err)
    xy_scl_err_mag = np.linalg.norm(xy_scl_err)
    return rot_err, trans_err, scl_err, xy_err_mag, yaw_err, xy_scl_err_mag, scale_mag_noise


def main():
    parser = argparse.ArgumentParser(description='Computes registration success rate and error metrics from affine result files.')    
    parser.add_argument('--gt',  default="20180524-mavic-ugv-soybean-eschikon-row3_AffineGroundTruth.txt", help='Ground truth file')
    parser.add_argument('--results',  default="20180524-mavic-ugv-soybean-eschikon-row3", help='Folder containing result files to process')
    parser.add_argument('-o', '--output', help='name of output folder')    

    args = parser.parse_args()

    

    root = Path(__file__).resolve().parent
    results_folder = root / args.results
    output_folder = root / args.output if args.output else results_folder
    output_folder.mkdir(parents=True, exist_ok=True)
    gta = np.loadtxt(root / args.gt)
    input_list = [f.name for f in results_folder.iterdir() if f.is_file()]
    # input_list = parse_text_list(root / "row3_CPD_list.txt")

    result = []
    grouped_by = {}
    for entry in input_list:
        if not entry:
            continue
        print(f"Processing file: {entry}")
        file_path = results_folder / entry
        if not file_path.exists():
            print(f"Missing file {file_path}, skipping")
            continue
        partitions = str(entry).split("_")
        idx = partitions[2]
        scale_noise_magnitude = float(partitions[3])
        transl_noise_magnitude = partitions[4]
        yaw_noise_magnitude = partitions[5]

        curr_file = np.loadtxt(file_path)
        A, t = parse_affine_result(curr_file)
        # xy_err = np.array([curr_file[14], curr_file[15]], dtype=float)        
        # yaw_err = float(curr_file[17])        
        # xy_scl_err = np.array([curr_file[18] - 1.0, curr_file[19] - 1.0], dtype=float)
        xy_err = np.array([curr_file[16], curr_file[17]], dtype=float)
        yaw_err = float(curr_file[18])
        xy_scl_err = np.array([curr_file[14] - 1.0, curr_file[15] - 1.0], dtype=float)
        result_metrics = compute_metrics(A, t, xy_err, yaw_err, xy_scl_err, scale_noise_magnitude)

        result.append(result_metrics)




        
        if scale_noise_magnitude not in grouped_by:
            grouped_by[scale_noise_magnitude] = {}
        method = partitions[6].split(".")[0]
        if method not in grouped_by[scale_noise_magnitude]:
            grouped_by[scale_noise_magnitude][method] = {
                "transl_err": [],
                "angle_err": [],
                "scale_err": []
            }
        grouped_by[scale_noise_magnitude][method]["transl_err"].append(result_metrics[1])
        grouped_by[scale_noise_magnitude][method]["angle_err"].append(result_metrics[0])
        grouped_by[scale_noise_magnitude][method]["scale_err"].append(result_metrics[2])

    result = np.vstack(result) if result else np.zeros((0, 7), dtype=float)
    np.savetxt(output_folder / "soybean_row3_cpd_comparison_result.csv", result, delimiter=',', header='rot_err,trans_err,scl_err,xy_err_mag,yaw_err,xy_scl_err_mag,scale_mag_noise', comments='', fmt='%.6f')
    np.save(output_folder / "soybean_row3_cpd_comparison_result.npy", result)
    print(f"Computed {result.shape[0]} rows and saved results to CSV and NPY.")
    print_metrics(grouped_by)


if __name__ == '__main__':
    main()
