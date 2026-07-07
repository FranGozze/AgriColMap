import numpy as np
from pathlib import Path


def load_numeric_file(path):
    return np.loadtxt(path, dtype=float)


def parse_text_list(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def scale_from_matrix(A):
    A = np.asarray(A, dtype=float)
    s1 = np.linalg.norm(A[:, 0])
    s2 = np.linalg.norm(A[:, 1])
    s3 = np.linalg.norm(A[:, 2])
    return np.array([s1, s2, s3], dtype=float)


def scale_matrix(A, s):
    A = np.asarray(A, dtype=float)
    s = np.asarray(s, dtype=float)
    B = A.copy()
    B[:, 0] = A[:, 0] / s[0]
    B[:, 1] = A[:, 1] / s[1]
    B[:, 2] = A[:, 2] / s[2]
    return B

def print_metrics(rowNumber, group):
    print("\nRegistration Error Metrics (averaged over successful cases):")
    print("Scale_mag trans_mag yaw_mag row succesRate  Method   Transl Err  Angle Err  Scale Err")
    
    for scale in sorted(group.keys()):
        for transl_noise in sorted(group[scale].keys()):
            for yaw_noise in sorted(group[scale][transl_noise].keys()):
                for method in sorted(group[scale][transl_noise][yaw_noise].keys()):
                    samples = len(group[scale][transl_noise][yaw_noise][method]['transl_err'])
                    if samples > 0:
                        avg_transl_err = np.mean(group[scale][transl_noise][yaw_noise][method]['transl_err'])
                        avg_angle_err = np.mean(group[scale][transl_noise][yaw_noise][method]['angle_err'])
                        avg_scale_err = np.mean(group[scale][transl_noise][yaw_noise][method]['scale_err'])
                        success_rate = samples / group[scale][transl_noise][yaw_noise][method]['count'] * 100
                        print(f"{float(scale):.2f}%,      {(float(transl_noise)*2.5):.2f} m,  {float(yaw_noise):.2f}º,  {rowNumber},      {success_rate:.1f}%,   {method},   {float(avg_transl_err):.3f}m,     {float(avg_angle_err):.3f}º,    {(float(avg_scale_err)*100):.2f} %")
    print("\n")


def print_metrics_latex(row_number, group, label="Registration Error Metrics"):
    print(f"% {label}")
    print("\\begin{tabular}{c|r|l|l|r|r}")
    print("\\toprule")
    print("Scale Noise & Succes rate &  Method & Transl Err (m) & Angle Err (deg) & Scale Err (\\%) \\\\ ")

    for scale in sorted(group.keys()):
        for transl_noise in sorted(group[scale].keys()):
            for yaw_noise in sorted(group[scale][transl_noise].keys()):
                print("\\midrule")
                print("\\multirow{" + str(len(group[scale][transl_noise][yaw_noise].keys())) + "}{*}{" + f"{float(scale)*100:.2f} \%"  + "}")
                for method in sorted(group[scale][transl_noise][yaw_noise].keys()):
                    avg_transl_err = np.mean(group[scale][transl_noise][yaw_noise][method]['transl_err'])
                    avg_angle_err = np.mean(group[scale][transl_noise][yaw_noise][method]['angle_err'])
                    avg_scale_err = np.mean(group[scale][transl_noise][yaw_noise][method]['scale_err'])
                    samples = len(group[scale][transl_noise][yaw_noise][method]['transl_err'])
                    # TODO: compute success rate and print it instead of samples/6
                    success_rate = samples / group[scale][transl_noise][yaw_noise][method]['count'] * 100
                    print(
                        f"& {success_rate:.1f} \% & {method} & {float(avg_transl_err):.3f} & {float(avg_angle_err):.3f} & {(float(avg_scale_err)):.3f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\n")


def compute_angle(A):
    A = np.asarray(A, dtype=float)
    value = (np.trace(A) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)
    return np.arccos(value)


def get_affine_from_vector(data):
    data = np.asarray(data, dtype=float).flatten()
    A = np.array([
        [data[0], data[1], data[2]],
        [data[4], data[5], data[6]],
        [data[8], data[9], data[10]],
    ], dtype=float)
    t = np.array([data[3], data[7], data[11]], dtype=float)
    return A, t


def rotation_error_from_affine(A):
    A = np.asarray(A, dtype=float)
    s = scale_from_matrix(A)
    R = A.copy()
    R[:, 0] /= s[0]
    R[:, 1] /= s[1]
    R[:, 2] /= s[2]
    tr = np.trace(R)
    return compute_angle(R)


def ensure_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)
