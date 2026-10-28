"""Plot the saved original wrist trajectory (no approximation yet)."""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def plot_3d_path(positions, name, out):
    """Plot the complete path with its start and end positions."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*positions.T, linewidth=0.7, label='Original')
    ax.scatter(*positions[0], color='green', label='Start')
    ax.scatter(*positions[-1], color='red', label='End')
    ax.set(
        xlabel='X (m)',
        ylabel='Y (m)',
        zlabel='Z (m)',
        title=f'Original {name} trajectory',
    )

    ax.set_box_aspect(np.maximum(np.ptp(positions, axis=0), 1e-6))
    ax.legend()

    fig.tight_layout()
    fig.savefig(out / 'original_trajectory_3d.png', dpi=160)
    plt.close(fig)


def plot_positions(times, positions, name, out):
    """Plot each coordinate against time."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, axis in enumerate(axes):
        axis.plot(times, positions[:, i], linewidth=0.8)
        axis.set_ylabel(f'{"XYZ"[i]} (m)')
        axis.grid(alpha=0.25)
    axes[0].set_title(f'Original {name} position (global coordinates)')
    axes[-1].set_xlabel('Time (s)')

    fig.tight_layout()
    fig.savefig(out / 'original_position.png', dpi=160)
    plt.close(fig)


def main():
    with np.load(ROOT / 'results/wrist_trajectory.npz') as data:
        positions = data['positions']
        times = data['timestamps']
        name = str(data['joint_name'].item()).replace('_', ' ')

    out = ROOT / 'results/figures'
    out.mkdir(parents=True, exist_ok=True)
    plot_3d_path(positions, name, out)
    plot_positions(times, positions, name, out)

    print(f'Saved original trajectory figures to {out}')


if __name__ == '__main__':
    main()
