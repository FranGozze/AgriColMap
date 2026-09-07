from pathlib import Path
import ast
import csv
import numpy as np
from output_utils import parse_text_list, scale_from_matrix, scale_matrix, compute_angle, print_metrics, print_metrics_latex

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



def parse_args():
    parser = argparse.ArgumentParser(description="Compress results")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory containing the results.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to the output CSV file where compressed results will be saved.",
    )
    parser.add_argument(
        "-a", "--add",
        action="store_true",
        help="Add the compressed results to the existing CSV file."
    )
    return parser.parse_args()

def add_to_grouped_data(grouped_data, method, scale_noise_magnitude, transl_noise_magnitude, yaw_noise_magnitude, idx, content):
    if method not in grouped_data:
        grouped_data[method] = {}
    if scale_noise_magnitude not in grouped_data[method]:
        grouped_data[method][scale_noise_magnitude] = {}
    if transl_noise_magnitude not in grouped_data[method][scale_noise_magnitude]:
        grouped_data[method][scale_noise_magnitude][transl_noise_magnitude] = {}
    if yaw_noise_magnitude not in grouped_data[method][scale_noise_magnitude][transl_noise_magnitude]:
        grouped_data[method][scale_noise_magnitude][transl_noise_magnitude][yaw_noise_magnitude] = {}
    grouped_data[method][scale_noise_magnitude][transl_noise_magnitude][yaw_noise_magnitude][idx] = content

def compress_results(input_dir, output_csv, add=False):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")


    file_list = [f.name for f in input_dir.iterdir() if f.is_file()]    
    csv_data = []
    grouped_data = {}
    if add and Path(output_csv).exists():
        # The old writer stored each dictionary using its repr(), which is not
        # a valid CSV row.  Read both that legacy format and regular CSV files.
        existing_data = []
        with open(output_csv, newline="", encoding="utf-8") as file:
            first_line = file.readline()
            file.seek(0)
            if first_line.startswith("{'method':"):
                print(f"Legacy format detected in {output_csv}. Reading as literal_eval.")
                existing_data = [ast.literal_eval(line) for line in file if line.strip()]
            else:
                print(f"CSV format detected in {output_csv}. Reading as CSV.")
                existing_data = list(csv.DictReader(file))
        for row in existing_data:
            # method,scale_noise_magnitude,transl_noise_magnitude,yaw_noise_magnitude,idx
            add_to_grouped_data(grouped_data, row["method"], row["scale_noise_magnitude"], row["transl_noise_magnitude"], row["yaw_noise_magnitude"], row["idx"], row["content"])

        print(f"Existing data loaded from {output_csv}. Entries: \n{grouped_data}")

    # Scale the results from matrix
    # for file_name in file_list:        
    #             grouped_data[row["method"]][row["scale_noise_magnitude"]][row["transl_noise_magnitude"]] = {}
    #         if row["yaw_noise_magnitude"] not in grouped_data[row["method"]][row["scale_noise_magnitude"]][row["transl_noise_magnitude"]]:
    #             grouped_data[row["method"]][row["scale_noise_magnitude"]][row["transl_noise_magnitude"]][row["yaw_noise_magnitude"]] = {}
    #         if row["idx"] not in grouped_data[row["method"]][row["scale_noise_magnitude"]][row["transl_noise_magnitude"]][row["yaw_noise_magnitude"]]:
    #             grouped_data[row["method"]][row["scale_noise_magnitude"]][row["transl_noise_magnitude"]][row["yaw_noise_magnitude"]][row["idx"]] = row["content"]
        
    #     print(f"Existing data loaded from {output_csv}. Entries: \n{grouped_data}")

    # Scale the results from matrix
    for file_name in file_list:        

        current_path = input_dir / file_name
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
        curr_file = np.loadtxt(current_path)
        content = "(" + ", ".join(map(str, curr_file)) + ")"
        # s, Aff, t, tn, yn, sn = extract_from_file(curr_file)
        if (method in grouped_data and
            scale_noise_magnitude in grouped_data[method] and
            transl_noise_magnitude in grouped_data[method][scale_noise_magnitude] and
            yaw_noise_magnitude in grouped_data[method][scale_noise_magnitude][transl_noise_magnitude] and
            idx in grouped_data[method][scale_noise_magnitude][transl_noise_magnitude][yaw_noise_magnitude]):
            print(f"Skipping existing entry for method: {method}, scale: {scale_noise_magnitude}, transl: {transl_noise_magnitude}, yaw: {yaw_noise_magnitude}, idx: {idx}")
            continue
        add_to_grouped_data(grouped_data, method, scale_noise_magnitude, transl_noise_magnitude, yaw_noise_magnitude, idx, content)
        
        csv_data.append({
            "method": method,
            "scale_noise_magnitude": scale_noise_magnitude,
            "transl_noise_magnitude": transl_noise_magnitude,
            "yaw_noise_magnitude": yaw_noise_magnitude,
            "idx": idx,
            "content": content,
        })
    

    # Prepare data for CSV output.  Do not concatenate a NumPy scalar with a
    # list of dictionaries; genfromtxt returns a zero-dimensional array for a
    # one-row file.
    headers = ["method", "scale_noise_magnitude", "transl_noise_magnitude",
               "yaw_noise_magnitude", "idx", "content"]
    output_path = Path(output_csv)
    rows = existing_data + csv_data if add and output_path.exists() else csv_data
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
        print(f"Compressed results written to {output_path}. Total entries: {len(rows)}")

    

def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    compress_results(root / args.input_dir, root / args.output_csv, args.add)

if __name__ == "__main__":
    main()
