import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from .fixed_times import design_matrix, fit_fixed_endpoints

ROOT = Path(__file__).resolve().parents[1]


def component_velocities(times, activation, completion, displacements):
    design_matrix(times, activation, completion)
    times = np.asarray(times, dtype=float)
    activation = np.asarray(activation, dtype=float)
    completion = np.asarray(completion, dtype=float)
    displacements = np.asarray(displacements, dtype=float)
    if displacements.shape != (len(activation), 3) or not np.isfinite(displacements).all():
        raise ValueError('Expected finite displacements (K, 3)')
    duration = completion - activation
    u = np.clip((times[:, None] - activation) / duration, 0, 1)
    rate = 30 * u**2 * (1 - u)**2 / duration
    return rate[:, :, None] * displacements[None, :, :]


def integrate(times, values):
    return float(np.sum(np.diff(times) * (values[:-1] + values[1:]) / 2))


def composition_metrics(times, velocities):
    times = np.asarray(times)
    lengths = [integrate(times, np.linalg.norm(velocities[:, i], axis=1))
               for i in range(velocities.shape[1])]
    total_length = integrate(times, np.linalg.norm(velocities.sum(axis=1), axis=1))
    component_length = sum(lengths)
    result = {
        'component_path_lengths_m': lengths,
        'sum_component_path_lengths_m': component_length,
        'combined_path_length_m': total_length,
        'relative_component_length_excess': (
            (component_length - total_length) / component_length if component_length else 0.0
        ),
    }
    if velocities.shape[1] == 2:
        dot = np.sum(velocities[:, 0] * velocities[:, 1], axis=1)
        result['opposing_velocity_sample_count'] = int(np.sum(dot < 0))
    return result


def main():
    source = ROOT / 'results/step4_known_timing'
    output = ROOT / 'results/step4_timing_diagnostic'
    output.mkdir(parents=True, exist_ok=True)
    summary_path = source / 'summary.json'
    fits_path = source / 'fits.npz'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    failed = next(row for row in summary['initializations'] if row['initialization'] == 2)
    with np.load(fits_path) as saved:
        times = saved['timestamps'].copy()
        original = saved['original'].copy()
    initial, final = original[0], original[-1]
    true_start = np.asarray(summary['known_activation'])
    true_end = np.asarray(summary['known_completion'])
    true_d = np.asarray(summary['known_displacements'])
    bad_start = np.asarray(failed['activation'])
    bad_end = np.asarray(failed['completion'])
    bad_d = np.asarray(failed['displacements'])
    known_velocity = component_velocities(times, true_start, true_end, true_d)
    bad_velocity = component_velocities(times, bad_start, bad_end, bad_d)
    diagnostics = {}
    for name, start, end, displacement, velocity in [
        ('known', true_start, true_end, true_d, known_velocity),
        ('initialization_2', bad_start, bad_end, bad_d, bad_velocity),
    ]:
        cosine = np.dot(displacement[0], displacement[1]) / (
            np.linalg.norm(displacement[0]) * np.linalg.norm(displacement[1])
        )
        diagnostics[name] = {
            **composition_metrics(times, velocity),
            'duration_s': (end - start).tolist(),
            'overlap_s': float(max(0, min(end) - max(start))),
            'displacement_direction_angle_degrees': float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
        }

    path_rows = []
    for fraction in np.linspace(0, 1, 201):
        start = (1 - fraction) * bad_start + fraction * true_start
        end = (1 - fraction) * bad_end + fraction * true_end
        fitted = fit_fixed_endpoints(times, original, initial, final, start, end)
        path_rows.append({'fraction_toward_known_times': float(fraction),
                          'position_rmse_m': fitted['position_rmse_m'],
                          'initial_error_m': fitted['initial_error_m'],
                          'final_error_m': fitted['final_error_m']})
    with (output / 'path_to_known_times.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(path_rows[0]))
        writer.writeheader()
        writer.writerows(path_rows)
    np.testing.assert_allclose(path_rows[0]['position_rmse_m'], failed['fitted_position_rmse_m'], atol=1e-12)
    assert path_rows[-1]['position_rmse_m'] < 1e-12
    assert max(row['final_error_m'] for row in path_rows) < 1e-12
    peak = max(path_rows, key=lambda row: row['position_rmse_m'])
    report = {
        'scope': 'Diagnosis only; no changes to the timing optimizer or cancellation penalty',
        'source_sha256': {'summary': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                          'fits': hashlib.sha256(fits_path.read_bytes()).hexdigest()},
        'optimizer_stop_message': failed['optimizer_message'],
        'optimizer_optimality': failed['optimizer_optimality'],
        'failed_run_position_rmse_m': failed['fitted_position_rmse_m'],
        'component_diagnostics': diagnostics,
        'path_definition': 'Linear interpolation of activation/completion times from run 2 to known times; refit displacements at every point',
        'path_samples': len(path_rows),
        'path_first_positive_fraction': path_rows[1],
        'path_maximum_rmse': peak,
        'path_end': path_rows[-1],
        'interpretation_limits': [
            'Known times are used only for this diagnostic, never passed to the optimizer',
            'Length excess includes effects of nonparallel directions, not just opposing directions',
            'The known decomposition also has opposing velocity contributions',
            'One sampled interpolation path does not characterize all directions or prove a strict local minimum',
            'This is not evidence that a cancellation penalty resolves the failure',
        ],
    }
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True, sharey='row')
    for column, velocities in enumerate((known_velocity, bad_velocity)):
        for coordinate, name in enumerate('XYZ'):
            axis = axes[coordinate, column]
            axis.plot(times, velocities[:, 0, coordinate], color='tab:blue', label='Component 1')
            axis.plot(times, velocities[:, 1, coordinate], color='tab:orange', label='Component 2')
            axis.plot(times, velocities.sum(axis=1)[:, coordinate], '--', color='black', label='Combined')
            axis.set_ylabel(f'{name} velocity (m/s)')
            axis.grid(alpha=0.2)
        axes[-1, column].set_xlabel('Time (s)')
    axes[0, 0].set_title('Known decomposition')
    axes[0, 1].set_title('Initialization 2 result')
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Analytic component velocities: different components can produce similar combined motion')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output / 'component_velocities.png', dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 4))
    axis.plot([row['fraction_toward_known_times'] for row in path_rows],
              [row['position_rmse_m'] * 100 for row in path_rows])
    axis.set_xlabel('Fraction from run 2 times toward known times')
    axis.set_ylabel('Position RMSE (cm)')
    axis.set_title('A sampled path to the known solution; displacements refitted at every point')
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / 'path_to_known_times.png', dpi=180)
    plt.close(fig)
    print('Component diagnostics:', json.dumps(diagnostics, indent=2))
    print('Path RMSE start / first step / maximum / end (m):',
          path_rows[0]['position_rmse_m'], path_rows[1]['position_rmse_m'],
          peak['position_rmse_m'], path_rows[-1]['position_rmse_m'])
    print('Peak path fraction:', peak['fraction_toward_known_times'])
    print('Saved:', output.relative_to(ROOT))


if __name__ == '__main__':
    main()
