import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def motion_scores(positions, times):
    positions = np.asarray(positions, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 3 or positions.shape != (len(times), 3):
        raise ValueError('Expected at least three timestamps and matching XYZ positions')
    if not np.isfinite(positions).all() or not np.isfinite(times).all():
        raise ValueError('Positions and timestamps must be finite')
    if np.any(np.diff(times) <= 0):
        raise ValueError('Timestamps must increase strictly')
    velocity = np.gradient(positions, times, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, times, axis=0, edge_order=2)
    return np.linalg.norm(acceleration, axis=1), velocity, acceleration


def local_peaks(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError('Scores must be a finite one-dimensional array')
    peaks = []
    index = 1
    while index < len(scores) - 1:
        end = index
        while end + 1 < len(scores) and scores[end + 1] == scores[index]:
            end += 1
        if (end < len(scores) - 1 and scores[index] > 0
                and scores[index] > scores[index - 1] and scores[end] > scores[end + 1]):
            peaks.append(index)
        index = end + 1
    return np.array(peaks, dtype=int)


def select_motion_goals(scores, times, max_goals=102, min_gap_s=0.1):
    if not isinstance(max_goals, int) or max_goals < 1:
        raise ValueError('Maximum goal count must be a positive integer')
    if not np.isfinite(min_gap_s) or min_gap_s <= 0:
        raise ValueError('Minimum gap must be positive and finite')
    if len(times) != len(scores) or np.any(np.diff(times) <= 0):
        raise ValueError('Scores need matching strictly increasing timestamps')
    if times[-1] - times[0] < min_gap_s - 1e-12:
        raise ValueError('Initial and final samples violate the minimum gap')
    peaks = local_peaks(scores)
    ranked = peaks[np.lexsort((peaks, -scores[peaks]))]
    selected = [0, len(times) - 1]
    order = []
    for frame in ranked:
        if len(selected) - 1 >= max_goals:
            break
        if np.all(np.abs(times[selected] - times[frame]) >= min_gap_s - 1e-12):
            selected.append(int(frame))
            order.append(int(frame))
    return np.sort(selected), ranked, np.array(order, dtype=int)


def plot_candidates(times, positions, scores, ranked, selected, output, window=None):
    mask = np.ones(len(times), dtype=bool)
    if window is not None:
        mask = (times >= window[0]) & (times <= window[1])
    selected_in_view = selected[mask[selected]]
    peaks_in_view = ranked[mask[ranked]]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for coordinate, axis in enumerate(axes[:3]):
        axis.plot(times[mask], positions[mask, coordinate], linewidth=0.8, label='Original')
        axis.scatter(times[selected_in_view], positions[selected_in_view, coordinate],
                     color='tab:orange', s=16, label='Selected source samples', zorder=3)
        axis.set_ylabel(f'{"XYZ"[coordinate]} (m)')
    axes[3].plot(times[mask], scores[mask], linewidth=0.8, label='Acceleration magnitude')
    axes[3].scatter(times[peaks_in_view], scores[peaks_in_view], s=8, color='gray',
                    label='Candidate peaks')
    axes[3].scatter(times[selected_in_view], scores[selected_in_view], s=20,
                    color='tab:orange', label='Selected (endpoints included)', zorder=3)
    axes[0].set_title('Motion-change selection diagnostic: raw data, no smoothing')
    axes[0].legend(fontsize=8)
    axes[3].set_ylabel('Acceleration magnitude (m/s^2)')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-goals', type=int, default=102)
    parser.add_argument('--min-gap', type=float, default=0.1)
    args = parser.parse_args()
    with np.load(ROOT / 'results/wrist_trajectory.npz') as data:
        positions = data['positions'].astype(np.float64)
        times = data['timestamps']
    scores, velocity, acceleration = motion_scores(positions, times)
    selected, ranked, order = select_motion_goals(scores, times, args.max_goals, args.min_gap)
    output = ROOT / 'results/followup/motion_change_diagnostic'
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'candidates.npz', timestamps=times, scores=scores,
                        velocity=velocity, acceleration=acceleration,
                        ranked_peak_indices=ranked, selected_indices=selected,
                        addition_order=order)
    selected_frames = set(order.tolist())
    with (output / 'candidates.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['rank', 'frame', 'time_s', 'acceleration_magnitude_m_s2',
                         'selected_at_requested_budget'])
        for rank, frame in enumerate(ranked, start=1):
            writer.writerow([rank, int(frame), float(times[frame]), float(scores[frame]),
                             int(frame) in selected_frames])
    summary = {
        'purpose': 'Candidate inspection only; no reconstruction or error comparison yet',
        'motivation': 'Choose goals from changes in the original motion rather than fixed intervals',
        'score': 'Magnitude of acceleration vector; rate of change of velocity, not jerk',
        'smoothing': 'None',
        'derivative': 'numpy.gradient twice, original timestamps, edge_order=2',
        'candidate_rule': 'Positive interior local maxima; earliest frame of exact plateau',
        'score_threshold': None,
        'ranking': 'Descending score, earlier frame breaks ties',
        'requested_max_goals': args.max_goals,
        'reached_goals': len(selected) - 1,
        'min_source_time_gap_s': args.min_gap,
        'candidate_peak_count': len(ranked),
        'goal_count_convention': 'Excludes initial position; includes final position',
        'selection_uses_reconstruction_error': False,
        'top_candidates': [
            {'frame': int(i), 'time_s': float(times[i]), 'score_m_s2': float(scores[i])}
            for i in ranked[:10]
        ],
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    plot_candidates(times, positions, scores, ranked, selected, output / 'candidates_full.png')
    plot_candidates(times, positions, scores, ranked, selected, output / 'candidates_detail.png',
                    window=(17, 23))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
