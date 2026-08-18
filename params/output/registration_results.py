from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from output_utils import ensure_folder


def main():
    root = Path(__file__).resolve().parent
    plot_dir = root / "plots"
    ensure_folder(plot_dir)

    scale_yaw = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25])

    vector_y_ss = np.array([1.00, 1.00, 0.99, 1.00, 1.00, 0.99, 0.99, 0.99, 1.00, 0.99, 1.00, 1.00, 0.98, 0.83, 0.17, 0.02, 0.00])
    vector_CPD_y_ss = np.array([0.96, 0.98, 0.96, 0.94, 0.95, 0.85, 0.71, 0.65, 0.32, 0.09, 0.02, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00])
    vector_ICP_y_ss = np.array([0.33, 0.11, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(scale_yaw, vector_y_ss, 'g', linewidth=3, label='Ours')
    ax.plot(scale_yaw, vector_CPD_y_ss, linewidth=3, label='CPD')
    ax.plot(scale_yaw, vector_ICP_y_ss, 'r', linewidth=3, label='ICP')
    ax.grid(True)
    ax.set_xlabel(r'$ 	riangle oldsymbol{	heta}~[	ext{rad}]$')
    ax.set_ylabel('Correct Registration Rate [%]')
    ax.set_xlim(0.02, 0.25)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.tick_params(axis='both', labelsize=14)
    fig.tight_layout()

    output_path = plot_dir / 'SmallScale_deltaYaw.pdf'
    fig.savefig(output_path, transparent=True)
    plt.close(fig)
    print(f'Saved plot to {output_path}')


if __name__ == '__main__':
    main()
