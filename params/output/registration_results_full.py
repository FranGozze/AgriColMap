from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from output_utils import ensure_folder


def save_plot(x, series, title, output_file):
    fig, ax = plt.subplots(figsize=(16, 9))
    for values, label, style in series:
        ax.plot(x, values, style, linewidth=3, label=label)
    ax.grid(True)
    if 'deltaT' in output_file.name:
        ax.set_xlabel('ΔT [m]')
        ax.set_xlim(0.02, 5.5)
    else:
        ax.set_xlabel('ΔΨ [rad]')
        ax.set_xlim(0.02, 0.285)
    ax.set_ylabel('Correct Registration Rate [%]')
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.tick_params(axis='both', labelsize=14)
    fig.tight_layout()
    fig.savefig(output_file, transparent=True)
    plt.close(fig)
    print(f'Saved plot to {output_file}')


def main():
    root = Path(__file__).resolve().parent
    plot_dir = root / 'plots'
    ensure_folder(plot_dir)

    scale_T = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5])
    scale_yaw = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25])

    vector_soybean = np.array([1.00, 1.00, 0.99, 0.99, 1.00, 1.00, 1.00, 0.99, 0.99, 1.00, 1.00, 0.98, 0.99, 1.00, 1.00, 0.99, 0.98, 0.97, 0.98, 0.95, 0.96, 0.97, 0.96, 0.00, 0.00])
    vector_sug10 = np.array([0.97, 1.00, 0.98, 0.96, 0.98, 0.99, 1.00, 0.98, 0.97, 0.98, 1.00, 0.99, 0.99, 0.96, 0.97, 0.95, 0.96, 0.97, 0.96, 0.94, 0.98, 0.96, 0.95, 0.00, 0.00])
    vector_sug20 = np.array([0.95, 0.95, 0.93, 0.97, 0.96, 0.94, 0.99, 0.98, 0.93, 0.94, 0.93, 0.98, 0.97, 0.99, 0.96, 0.95, 0.94, 0.98, 0.96, 0.96, 0.99, 0.96, 0.94, 0.00, 0.00])
    vector_ww = np.array([1.00, 0.99, 0.96, 0.98, 0.96, 0.99, 0.99, 1.00, 0.97, 0.98, 0.99, 0.98, 0.96, 0.95, 0.96, 0.99, 1.00, 0.95, 0.99, 0.98, 0.98, 0.93, 0.97, 0.00, 0.00])
    vector_CPD = np.array([0.96, 0.83, 0.34, 0.12, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    vector_ICP = np.array([0.97, 0.54, 0.07, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    vector_GoICP = np.array([0.97, 0.98, 0.96, 0.92, 0.98, 0.91, 0.79, 0.43, 0.21, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    vector_SURF = np.array([0.83, 0.86, 0.85, 0.87, 0.84, 0.86, 0.88, 0.83, 0.84, 0.88, 0.85, 0.86, 0.88, 0.85, 0.84, 0.85, 0.86, 0.84, 0.82, 0.83, 0.83, 0.82, 0.84, 0.00, 0.00])
    vector_ORB = np.array([0.85, 0.85, 0.83, 0.86, 0.89, 0.86, 0.83, 0.82, 0.84, 0.85, 0.86, 0.84, 0.89, 0.84, 0.84, 0.83, 0.82, 0.81, 0.85, 0.79, 0.85, 0.81, 0.82, 0.00, 0.00])
    vector_FAST_BRIEF = np.array([0.84, 0.86, 0.87, 0.83, 0.81, 0.83, 0.85, 0.84, 0.86, 0.87, 0.88, 0.89, 0.85, 0.83, 0.81, 0.82, 0.87, 0.83, 0.85, 0.85, 0.89, 0.83, 0.85, 0.00, 0.00])

    save_plot(
        scale_T,
        [
            (vector_soybean, 'AgriColMap_sb', 'g'),
            (vector_sug10, 'AgriColMap_sg10', 'g'),
            (vector_sug20, 'AgriColMap_sg20', 'r'),
            (vector_ww, 'AgriColMap_ww', 'm'),
            (vector_CPD, 'CPD', 'g--'),
            (vector_ICP, 'ICP', 'r--'),
            (vector_GoICP, 'GoICP', 'c'),
            (vector_SURF, 'SURF', 'b--o'),
            (vector_ORB, 'ORB', 'r--^'),
            (vector_FAST_BRIEF, 'FAST-BRIEF', 'g--p'),
        ],
        'Success Registration Rate at 0% Scale Error',
        plot_dir / 'noScale_deltaT_goicp_full.pdf'
    )

    vector_ICP_y = np.array([0.96, 0.98, 0.96, 0.94, 0.95, 0.93, 0.76, 0.46, 0.23, 0.11, 0.02, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00])
    vector_sug10_y = np.array([0.99, 0.95, 0.97, 0.92, 0.94, 0.98, 0.94, 0.95, 0.98, 0.97, 0.96, 0.94, 0.86, 0.14, 0.04, 0.00, 0.00])
    vector_sug20_y = np.array([0.99, 0.93, 0.98, 0.96, 0.97, 0.99, 0.94, 0.96, 0.97, 0.95, 0.93, 0.99, 0.92, 0.20, 0.02, 0.00, 0.00])
    vector_ww_y = np.array([1.00, 0.95, 0.95, 0.98, 0.98, 0.96, 0.97, 0.93, 0.99, 0.99, 0.95, 1.00, 0.89, 0.26, 0.01, 0.00, 0.00])
    vector_soybean_y = np.array([1.00, 1.00, 0.99, 1.00, 0.99, 0.99, 1.00, 0.99, 0.98, 0.99, 1.00, 0.99, 0.98, 0.30, 0.07, 0.00, 0.00])
    vector_CPD_y = np.array([0.96, 0.98, 0.96, 0.94, 0.95, 0.93, 0.92, 0.87, 0.74, 0.54, 0.32, 0.21, 0.12, 0.05, 0.01, 0.00, 0.00])
    vector_Go_ICP_y = np.array([0.96, 0.97, 0.97, 0.97, 0.95, 0.62, 0.13, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
    vector_SURF_y = np.array([0.88, 0.86, 0.85, 0.81, 0.56, 0.21, 0.03, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
    vector_ORB_y = np.array([0.86, 0.88, 0.89, 0.24, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
    vector_FAST_BRIEF_y = np.array([0.88, 0.85, 0.86, 0.37, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])

    save_plot(
        scale_yaw,
        [
            (vector_soybean_y, 'AgriColMap_sb', 'g'),
            (vector_sug10_y, 'AgriColMap_sg10', 'g'),
            (vector_sug20_y, 'AgriColMap_sg20', 'r'),
            (vector_ww_y, 'AgriColMap_ww', 'm'),
            (vector_CPD_y, 'CPD', 'g--'),
            (vector_ICP_y, 'ICP', 'r--'),
            (vector_Go_ICP_y, 'GoICP', 'c'),
            (vector_SURF_y, 'SURF', 'b--o'),
            (vector_ORB_y, 'ORB', 'r--^'),
            (vector_FAST_BRIEF_y, 'FAST-BRIEF', 'g--p'),
        ],
        'Success Registration Rate at 0% Scale Error',
        plot_dir / 'noScale_deltaYaw_goicp_full.pdf'
    )


if __name__ == '__main__':
    main()
