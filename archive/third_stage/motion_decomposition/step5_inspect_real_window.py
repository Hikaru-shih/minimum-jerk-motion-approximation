import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .step3_speed_minima import speed_profile

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]


def main():
    source_path = PROJECT_ROOT / 'results/wrist_trajectory.npz'
    output = ROOT / 'results/step5_real_window'
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    with np.load(source_path) as source:
        times = source['timestamps'].astype(float)
        positions = source['positions'].astype(float)
    velocity, speed = speed_profile(positions, times)
    frames = np.flatnonzero((times >= 18.0) & (times <= 21.0))
    t, p, v, s = times[frames], positions[frames], velocity[frames], speed[frames]
    assert len(frames) == 361
    assert np.all(np.diff(frames) == 1)
    assert np.isfinite(p).all() and np.isfinite(v).all()
    assert np.allclose(np.diff(t), 1 / 120, atol=1e-12, rtol=0)
    cropped_velocity, _ = speed_profile(p, t)
    np.savez_compressed(output / 'window.npz', source_frames=frames,
                        timestamps=t, relative_timestamps=t-t[0], positions=p,
                        velocity_from_full_clip=v, speed_from_full_clip=s)
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True)
    for c in range(3):
        axes[c, 0].plot(t, p[:, c], color='tab:blue')
        axes[c, 0].scatter(t[[0, -1]], p[[0, -1], c], color='tab:red', zorder=3)
        axes[c, 0].set_ylabel(f'{"XYZ"[c]} position (m)')
        axes[c, 1].plot(t, v[:, c], color='tab:orange')
        axes[c, 1].axhline(0, color='gray', linewidth=0.6)
        axes[c, 1].set_ylabel(f'{"XYZ"[c]} velocity (m/s)')
    axes[3, 0].plot(t, np.linalg.norm(p-p[0], axis=1), color='tab:blue')
    axes[3, 0].set_ylabel('Distance from start (m)')
    axes[3, 1].plot(t, s, color='black')
    axes[3, 1].scatter(t[[0, -1]], s[[0, -1]], color='tab:red', zorder=3)
    axes[3, 1].set_ylabel('Speed (m/s)')
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel('Original clip time (s)')
    fig.suptitle('Real wrist window: 18-21 s; original samples, no smoothing\n'
                 'Velocity differentiated on full clip before cropping; red = crop endpoints')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / 'position_velocity.png', dpi=160)
    plt.close(fig)
    summary = {
        'source': 'results/wrist_trajectory.npz', 'source_sha256': source_hash,
        'window_s': [float(t[0]), float(t[-1])],
        'source_frame_indices_inclusive': [int(frames[0]), int(frames[-1])],
        'samples': len(t), 'duration_s': float(t[-1]-t[0]),
        'coordinate_frame': 'Original global XYZ in meters; no recentering',
        'selection_reason': 'Previously inspected region with relatively steady portions around a clear position change; debugging window, not representative validation',
        'velocity_definition': 'numpy.gradient on full original clip, edge_order=2, then crop',
        'start_position_m': p[0].tolist(), 'end_position_m': p[-1].tolist(),
        'endpoint_speeds_m_s': s[[0, -1]].tolist(),
        'speed_range_m_s': [float(s.min()), float(s.max())],
        'coordinate_ranges_m': np.ptp(p, axis=0).tolist(),
        'crop_only_derivative_max_endpoint_difference_m_s': float(np.max(np.linalg.norm(cropped_velocity[[0,-1]]-v[[0,-1]], axis=1))),
        'limitations': [
            'Crop endpoints are not validated movement boundaries or zero-speed states',
            'Completed minimum-jerk components imply zero endpoint velocity; fixed positions alone do not remove this mismatch',
            'No smoothing, component count selection, reconstruction, timing optimization, or penalty applied',
            'One selected short window cannot validate the complete clip or movement-unit interpretation',
        ],
    }
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
