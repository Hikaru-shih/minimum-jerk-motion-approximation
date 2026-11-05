import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
WINDOWS = ((8.0, 12.0), (17.0, 23.0), (31.0, 33.0))


def speed_profile(positions, times):
    positions = np.asarray(positions, dtype=float)
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 3 or positions.shape != (len(times), 3):
        raise ValueError('Expected at least three timestamps and positions (N, 3)')
    if not np.isfinite(positions).all() or not np.isfinite(times).all():
        raise ValueError('Positions and times must be finite')
    if np.any(np.diff(times) <= 0):
        raise ValueError('Timestamps must increase strictly')
    velocity = np.gradient(positions, times, axis=0, edge_order=2)
    return velocity, np.linalg.norm(velocity, axis=1)


def local_minimum_runs(speed):
    speed = np.asarray(speed, dtype=float)
    if speed.ndim != 1 or len(speed) < 3:
        raise ValueError('Expected at least three speed samples')
    if not np.isfinite(speed).all() or np.any(speed < 0):
        raise ValueError('Speed must be finite and nonnegative')
    minima = []
    start = 1
    while start < len(speed) - 1:
        end = start
        while end + 1 < len(speed) and speed[end + 1] == speed[start]:
            end += 1
        if (end < len(speed) - 1 and speed[start] < speed[start - 1]
                and speed[end] < speed[end + 1]):
            minima.append((start, end))
        start = end + 1
    return np.asarray(minima, dtype=int).reshape(-1, 2)


def plot_candidates(times, positions, speed, candidate_frames, path, window=None):
    mask = np.ones(len(times), dtype=bool)
    if window is not None:
        mask = (times >= window[0]) & (times <= window[1])
    selected = candidate_frames[mask[candidate_frames]]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for coordinate, axis in enumerate(axes[:3]):
        axis.plot(times[mask], positions[mask, coordinate], color='tab:blue', linewidth=0.9)
        axis.scatter(times[selected], positions[selected, coordinate], color='tab:orange',
                     s=9 if window is not None else 4, zorder=3)
        axis.set_ylabel(f'{"XYZ"[coordinate]} (m)')
    axes[3].plot(times[mask], speed[mask], color='black', linewidth=0.9, label='Raw speed')
    axes[3].scatter(times[selected], speed[selected], color='tab:orange',
                    s=14 if window is not None else 5, zorder=3, label='Local minima (candidates)')
    axes[3].set_ylabel('Speed (m/s)')
    axes[3].set_ylim(bottom=0)
    axes[3].legend(loc='upper right', fontsize=9)
    axes[3].set_xlabel('Time (s)')
    scope = 'full clip' if window is None else f'{window[0]:g}-{window[1]:g} s'
    axes[0].set_title(f'Step 3: {scope}; {len(selected)} raw minima; no smoothing or filtering')
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    source_path = PROJECT_ROOT / 'results/wrist_trajectory.npz'
    output = ROOT / 'results/step3_speed_minima'
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source_path) as source:
        positions = source['positions'].astype(float)
        times = source['timestamps'].copy()
    velocity, speed = speed_profile(positions, times)
    runs = local_minimum_runs(speed)
    frames = runs[:, 0]
    spacing = np.diff(times[frames])
    rows = []
    for number, (start, end) in enumerate(runs):
        rows.append({
            'frame': int(start), 'time_s': float(times[start]),
            'plateau_end_frame': int(end), 'plateau_end_time_s': float(times[end]),
            'speed_m_s': float(speed[start]),
            'left_speed_m_s': float(speed[start - 1]),
            'right_speed_m_s': float(speed[end + 1]),
            'next_candidate_gap_s': float(spacing[number]) if number < len(spacing) else '',
            'x_m': float(positions[start, 0]),
            'y_m': float(positions[start, 1]),
            'z_m': float(positions[start, 2]),
        })
    with (output / 'candidates.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'frame', 'time_s', 'plateau_end_frame', 'plateau_end_time_s', 'speed_m_s',
            'left_speed_m_s', 'right_speed_m_s', 'next_candidate_gap_s', 'x_m', 'y_m', 'z_m',
        ])
        writer.writeheader()
        writer.writerows(rows)
    window_records = []
    for start, end in WINDOWS:
        mask = (times >= start) & (times <= end)
        selected = frames[(times[frames] >= start) & (times[frames] <= end)]
        record = {
            'window_s': [start, end],
            'candidate_count': len(selected),
            'speed_min_m_s': float(speed[mask].min()),
            'speed_max_m_s': float(speed[mask].max()),
            'coordinate_ranges_m': np.ptp(positions[mask], axis=0).tolist(),
        }
        window_records.append(record)
        plot_candidates(times, positions, speed, frames,
                        output / f'candidates_{int(start):02d}_{int(end):02d}.png', (start, end))
    summary = {
        'scope': 'Unfiltered speed-minimum candidates, not validated movement boundaries',
        'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'frames': len(times),
        'speed_definition': 'Euclidean norm of XYZ velocity, not a signed velocity component',
        'derivative': 'numpy.gradient on all original timestamps, axis=0, edge_order=2',
        'smoothing': None,
        'speed_threshold': None,
        'minimum_time_gap': None,
        'candidate_rule': 'Interior run strictly lower than both neighboring samples; exact plateau uses earliest frame',
        'endpoints': 'Not detected as interior minima; endpoint treatment remains a separate segmentation decision',
        'candidate_count': len(frames),
        'flat_minimum_runs': int(np.sum(runs[:, 1] > runs[:, 0])),
        'candidate_speed_quantiles_m_s': dict(zip(
            ['minimum', 'q25', 'median', 'q75', 'maximum'],
            np.quantile(speed[frames], [0, 0.25, 0.5, 0.75, 1]).tolist()
        )) if len(frames) else None,
        'candidate_spacing_quantiles_s': dict(zip(
            ['minimum', 'q25', 'median', 'q75', 'maximum'],
            np.quantile(spacing, [0, 0.25, 0.5, 0.75, 1]).tolist()
        )) if len(spacing) else None,
        'gaps_below_0_1_s': int(np.sum(spacing < 0.1 - 1e-12)),
        'gap_reference_note': '0.1 s is descriptive only, recalling the earlier experiment; no spacing filter is applied',
        'display_windows_reason': 'Previously inspected steady, transition, and rapid-fluctuation regions; not a representative motion sample',
        'windows': window_records,
        'limitations': [
            'A local speed minimum need not be a stop or movement boundary',
            'Raw differentiation can emphasize small variations; their physical origin is not established',
            'Global wrist speed depends on the selected coordinate frame and body motion',
            'No reconstruction fitting, timing optimization, or goal-count decision is performed',
        ],
    }
    (output / 'summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8'
    )
    np.savez_compressed(output / 'candidates.npz', timestamps=times, positions=positions,
                        velocity=velocity, speed=speed, candidate_frames=frames, minimum_runs=runs)
    plot_candidates(times, positions, speed, frames, output / 'candidates_full.png')
    print('Raw local speed minima:', len(frames))
    print('Candidate speed quantiles (m/s):', summary['candidate_speed_quantiles_m_s'])
    print('Candidate spacing quantiles (s):', summary['candidate_spacing_quantiles_s'])
    print('Gaps below 0.1 s (not filtered):', summary['gaps_below_0_1_s'], '/', len(spacing))
    for window in window_records:
        print('Window:', window)
    print('Candidates have not been validated as boundaries. Saved:', output.relative_to(ROOT))


if __name__ == '__main__':
    main()
