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


def banned_method(method):
    return method.lower() == "unkwown" or method.lower() == "unkwon" or "soil" in method.lower() or "lss" in method.lower() or method.lower() == "resnet" or "2" in method
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
                        print(f"{float(scale):.2f}%,      {(float(transl_noise)*2.5):.2f} m,  {float(yaw_noise):.2f}º,  {rowNumber},    {samples} / {group[scale][transl_noise][yaw_noise][method]['count']}   {method},   {float(avg_transl_err):.3f}m,     {float(avg_angle_err):.3f}º,    {(float(avg_scale_err)*100):.2f} %")
    print("\n")


def print_metrics_latex(row_number, group, label="Registration Error Metrics", save_path=None):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / f"row_{row_number}_metrics.tex", "w", encoding="utf-8") as f:
        f.write(f"%File: row_{row_number}_metrics.tex\n")
        f.write("\\begin{table*}[t] \n \\centering\n")
        f.write("\\begin{tabular}{c|r|l|l|r|r}\n")
        f.write("\\toprule\n")
        f.write("Scale Noise & Method & Succes rate  & Transl Err (m) & Angle Err (deg) & Scale Err (\\%) \\\\ \n")

        for scale in sorted(group.keys()):
            for transl_noise in sorted(group[scale].keys()):
                for yaw_noise in sorted(group[scale][transl_noise].keys()):
                    f.write("\\midrule\n")
                    f.write("\\multirow{" + str(len(group[scale][transl_noise][yaw_noise].keys())) + "}{*}{" + f"{float(scale)*100:.2f} \%"  + "}\n")
                    for method in sorted(group[scale][transl_noise][yaw_noise].keys()):
                        if banned_method(method):
                            continue
                        samples = len(group[scale][transl_noise][yaw_noise][method]['transl_err'])
                        if samples > 0:
                            avg_transl_err = np.mean(group[scale][transl_noise][yaw_noise][method]['transl_err'])
                            avg_angle_err = np.mean(group[scale][transl_noise][yaw_noise][method]['angle_err'])
                            avg_scale_err = np.mean(group[scale][transl_noise][yaw_noise][method]['scale_err'])
                            success_rate = samples / group[scale][transl_noise][yaw_noise][method]['count'] * 100
                            f.write(
                                f"& {success_rate:.1f} \% & {method} & {float(avg_transl_err):.3f} & {float(avg_angle_err):.3f} & {(float(avg_scale_err)):.3f} \\\\ \n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
        f.write("\n")


def print_metrics_heatmap(row_number, group, label="Registration Success Rate", save_path=None):
    
    def color(value, mean, vmin, vmax):
        delta1 = mean - vmin
        delta2 = vmax - vmin
        assert delta1 > 0 and delta2 > 0, f"Invalid range: mean={mean}, vmin={vmin}, vmax={vmax}"
        A = (0.5 * delta2 - delta1) / (delta1 * delta2 * (delta1 - delta2))
        B = (1 - A * (delta2**2)) / delta2
        x = A * ((value - vmin)**2) + B * (value - vmin)
        assert 0.0 <= x <= 1.0, f"Invalid color value: {x} for value={value}, mean={mean}, vmin={vmin}, vmax={vmax}"
        return x
    def success_cell_color(value, mean, vmin, vmax):
        x = color(value, mean, vmin, vmax)
        return f"{1-x:.3f} {x:.3f} 0.000"
    def error_cell_color(value, mean, vmin, vmax):
        x = color(value, mean, vmin, vmax)
        return f"{x:.3f} {1-x:.3f} 0.000"
    def error_data(rows, index):
        _mean = np.mean([r[index] for r in rows]) if rows else 0.0
        _min = min([r[index] for r in rows]) if rows else 0.0
        _max = max([r[index] for r in rows]) if rows else 1.0
        return _mean, _min, _max
    def print_rows(rows):
        mean_success_rate = np.mean([r[0] for r in rows]) if rows else 0.0
        min_success_rate = min([r[0] for r in rows]) if rows else 0.0
        max_success_rate = max([r[0] for r in rows]) if rows else 100.0
        
        mean_transl_err, min_transl_err, max_transl_err = error_data(rows, 2)
        
        mean_angle_err, min_angle_err, max_angle_err = error_data(rows, 3)
        
        mean_scale_err, min_scale_err, max_scale_err = error_data(rows, 4)

        print(f"% transl_err: mean={mean_transl_err:.3f}, min={min_transl_err:.3f}, max={max_transl_err:.3f}")
        result = []        
        for success_rate, method, avg_transl_err, avg_angle_err, avg_scale_err in rows:                
            result.append(
                f"& {method} & \\cellcolor[rgb]{{{success_cell_color(success_rate, mean_success_rate, min_success_rate, max_success_rate)}}} {success_rate:.1f} \%"
                + f" & \\cellcolor[rgb]{{{error_cell_color(avg_transl_err, mean_transl_err, min_transl_err, max_transl_err)}}}{float(avg_transl_err):.3f}"
                + f" & \\cellcolor[rgb]{{{error_cell_color(avg_angle_err, mean_angle_err, min_angle_err, max_angle_err)}}}{float(avg_angle_err):.3f}"
                + f" & \\cellcolor[rgb]{{{error_cell_color(avg_scale_err, mean_scale_err, min_scale_err, max_scale_err)}}}{(float(avg_scale_err)):.3f} \\\\"
            )
        return "\n".join(result)
        # return ""


    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / f"row_{row_number}_metrics_heatmap.tex", "w", encoding="utf-8") as f:
        f.write(f"%File: row_{row_number}_metrics_heatmap.tex\n")
        f.write("\\begin{table*}[t] \n \\centering\n")
        f.write("\\begin{tabular}{c|r|l|l|r|r}\n")
        f.write("\\toprule\n")
        f.write("Scale Noise & Method & Succes rate  & Transl Err (m) & Angle Err (deg) & Scale Err (\\%) \\\\ \n")
        for scale in sorted(group.keys()):
            for transl_noise in sorted(group[scale].keys()):
                for yaw_noise in sorted(group[scale][transl_noise].keys()):
                    f.write("\\midrule\n")
                    f.write("\\multirow{" + str(len(group[scale][transl_noise][yaw_noise].keys())) + "}{*}{" + f"{float(scale)*100:.2f} \%"  + "}\n")
                    results = []
                    for method in sorted(group[scale][transl_noise][yaw_noise].keys()):
                        if banned_method(method):
                            continue
                        samples = len(group[scale][transl_noise][yaw_noise][method]['transl_err'])
                        if samples > 0:
                            avg_transl_err = np.mean(group[scale][transl_noise][yaw_noise][method]['transl_err'])
                            avg_angle_err = np.mean(group[scale][transl_noise][yaw_noise][method]['angle_err'])
                            avg_scale_err = np.mean(group[scale][transl_noise][yaw_noise][method]['scale_err'])
                            success_rate = samples / group[scale][transl_noise][yaw_noise][method]['count'] * 100
                            results.append((success_rate, method, avg_transl_err, avg_angle_err, avg_scale_err))
                    f.write(print_rows(results))

        f.write("\n\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
        


def plot_metrics(row_number, group, title="Registration Error Metrics", save_path=None):
    import matplotlib.pyplot as plt

    scales = sorted(group.keys())
    transl_noises = sorted(group[scales[0]].keys())
    yaw_noises = sorted(group[scales[0]][transl_noises[0]].keys())
    methods = sorted(group[scales[0]][transl_noises[0]][yaw_noises[0]].keys())
    _max = 0
    
    for scale in scales:
        for transl_noise in transl_noises:
            for yaw_noise in yaw_noises:
               
                plt.figure(figsize=(20, 8))
                counter = 0
                for method in methods:
                    if banned_method(method):
                        continue
                    samples = len(group[scale][transl_noise][yaw_noise][method]['transl_err'])
                    if samples > 0:
                        avg_transl_err = np.mean(group[scale][transl_noise][yaw_noise][method]['transl_err'])
                        avg_angle_err = np.mean(group[scale][transl_noise][yaw_noise][method]['angle_err'])
                        avg_scale_err = np.mean(group[scale][transl_noise][yaw_noise][method]['scale_err'])
                        success_rate = samples / group[scale][transl_noise][yaw_noise][method]['count'] * 100
                        _max = max(_max, success_rate + 5)
                        plt.bar(f"{counter}", success_rate, label=f"{counter}-{method}")
                        counter += 1

                plt.title(f"{title} - Scale: {float(scale)*100:.2f}%, Transl Noise: {float(transl_noise):.2f} m, Yaw Noise: {float(yaw_noise):.2f}º")
                plt.ylabel("Success Rate (%)")
                plt.ylim(0, _max)
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.grid(axis='y')
                if save_path is None:
                    plt.show()
                else:                
                    save_path = Path(save_path)
                    save_path.mkdir(parents=True, exist_ok=True)
                    plt.savefig(save_path / f"row_{row_number}_metrics_scale_{scale}_transl_{transl_noise}_yaw_{yaw_noise}.png", bbox_inches='tight')



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
