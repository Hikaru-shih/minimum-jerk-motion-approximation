"""Approximate the wrist using fixed sampled goals and overlapping MJ displacements."""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOAL_INTERVAL = 0.5
OVERLAP = 0.25


def minimum_jerk(times, activation, completion):
    """Unit displacement, held at zero before activation and one after completion."""
    if completion <= activation:
        raise ValueError('Completion must follow activation')

    u = np.clip((times - activation) / (completion - activation), 0.0, 1.0)

    return u**3 * (10.0 + u * (-15.0 + 6.0 * u))


def reconstruct(times, initial, goals, activation, completion):
    """Sum trajectory displacements from the initial wrist position."""
    displacements = np.diff(np.vstack([initial, goals]), axis=0)
    result = np.broadcast_to(initial, (len(times), 3)).copy()

    for displacement, a, b in zip(displacements, activation, completion):
        result += minimum_jerk(times, a, b)[:, None] * displacement

    return result


def select_goals(original, times, fps):
    """Sample positions and expand their intervals to overlap."""


    step = round(GOAL_INTERVAL * fps)
    indices = np.unique(np.r_[np.arange(0, len(times), step), len(times) - 1])
    sample_times = times[indices]
    goals = original[indices[1:]]



    activation = np.maximum(times[0], sample_times[:-1] - OVERLAP / 2)
    completion = np.minimum(times[-1], sample_times[1:] + OVERLAP / 2)

    return indices, sample_times, goals, activation, completion


def plot_comparisons(
    times, original, approx, original_velocity, approx_velocity, out
):
    """Save full position and velocity comparisons."""
    for filename, ref, fitted, unit, title in [
        ('position_comparison.png', original, approx, 'm', 'Right wrist position'),
        (
            'velocity_comparison.png',
            original_velocity,
            approx_velocity,
            'm/s',
            'Right wrist velocity (same finite difference, no filtering)',
        ),
    ]:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        for j, ax in enumerate(axes):
            ax.plot(times, ref[:, j], label='Original', linewidth=0.9)
            ax.plot(
                times,
                fitted[:, j],
                label='Minimum jerk',
                linewidth=0.9,
                linestyle='--',
            )
            ax.set_ylabel(f'{"XYZ"[j]} ({unit})')
            ax.grid(alpha=0.25)

        axes[0].set_title(title)
        axes[0].legend()
        axes[-1].set_xlabel('Time (s)')
        fig.tight_layout()
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)


def plot_position_detail(times, original, approx, sample_times, goals, out):
    """Show 17-23 seconds from the same reconstruction."""

    mask = (times >= 17) & (times <= 23)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(times[mask], original[mask, j], label='Original')
        ax.plot(times[mask], approx[mask, j], '--', label='Minimum jerk')
        selected = (sample_times[1:] >= 17) & (sample_times[1:] <= 23)
        ax.scatter(
            sample_times[1:][selected],
            goals[selected, j],
            s=15,
            color='black',
            label='Sampled goal (source time)',
        )
        ax.set_ylabel(f'{"XYZ"[j]} (m)')
        ax.grid(alpha=0.25)
    axes[0].set_title('Position comparison: 17-23 s (same reconstruction)')
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(out / 'position_comparison_detail.png', dpi=160)
    plt.close(fig)


def main():

    with np.load(ROOT / 'results/wrist_trajectory.npz') as data:
        original = data['positions'].astype(np.float64)
        times = data['timestamps']
        fps = float(data['frame_rate'])

    if len(times) < 3 or np.any(np.diff(times) <= 0):
        raise ValueError('Expected at least three ordered timestamps')


    indices, sample_times, goals, activation, completion = select_goals(
        original, times, fps
    )


    approx = reconstruct(times, original[0], goals, activation, completion)

    if not np.isfinite(approx).all():
        raise ValueError('Non-finite reconstruction')

    np.testing.assert_allclose(approx[0], original[0], atol=1e-10)
    np.testing.assert_allclose(approx[-1], original[-1], atol=1e-10)


    original_velocity = np.gradient(original, times, axis=0, edge_order=2)
    approx_velocity = np.gradient(approx, times, axis=0, edge_order=2)


    np.savez_compressed(
        ROOT / 'results/goals.npz',
        initial_position=original[0],
        goal_positions=goals,
        source_frame_indices=indices[1:],
        source_sample_times=sample_times[1:],
        activation_times=activation,
        completion_times=completion,
        goal_interval=GOAL_INTERVAL,
        overlap=OVERLAP,
    )
    np.savez_compressed(
        ROOT / 'results/approximation.npz',
        timestamps=times,
        positions=approx,
        original_velocity=original_velocity,
        approximated_velocity=approx_velocity,
    )


    out = ROOT / 'results/figures'
    out.mkdir(exist_ok=True)
    plot_comparisons(times, original, approx, original_velocity, approx_velocity, out)
    plot_position_detail(times, original, approx, sample_times, goals, out)

    print(f'Goals: {len(goals)} (initial position stored separately)')
    print(f'Interval: {GOAL_INTERVAL} s; overlap: {OVERLAP} s')
    print(f'Reconstructed samples: {approx.shape}; endpoints verified')
    print(f'Figures saved to {out}')


if __name__ == '__main__':
    main()
