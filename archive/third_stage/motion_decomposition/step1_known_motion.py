import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from .fixed_times import fit_displacements

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]


def generate_motion(times, initial, displacements, activation, completion):
    positions = np.tile(initial, (len(times), 1)).astype(float)
    for displacement, start, end in zip(displacements, activation, completion):
        u = np.clip((times - start) / (end - start), 0.0, 1.0)
        progress = 10 * u**3 - 15 * u**4 + 6 * u**5
        positions += progress[:, None] * displacement
    return positions


def main():
    output = ROOT / 'results/step1_known_motion'
    output.mkdir(parents=True, exist_ok=True)
    times = np.arange(409) / 120.0
    initial = np.array([0.20, -0.10, 0.40])
    known_displacements = np.array([[0.30, 0.10, -0.05], [-0.12, 0.20, 0.08]])
    activation = np.array([0.20, 1.20])
    completion = np.array([2.20, 3.20])
    original = generate_motion(
        times, initial, known_displacements, activation, completion
    )
    fitted = fit_displacements(times, original, initial, activation, completion)
    known_goals = initial + np.cumsum(known_displacements, axis=0)

    np.testing.assert_allclose(fitted['displacements'], known_displacements, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fitted['positions'], original, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fitted['goals'], known_goals, rtol=0, atol=1e-12)
    assert fitted['rank'] == 2

    identical_activation = np.array([0.20, 0.20])
    identical_completion = np.array([2.20, 2.20])
    ambiguous_original = generate_motion(
        times, initial, known_displacements, identical_activation, identical_completion
    )
    ambiguous = fit_displacements(
        times, ambiguous_original, initial, identical_activation, identical_completion
    )
    alternative = known_displacements + np.array([[0.05, 0.03, -0.02], [-0.05, -0.03, 0.02]])
    alternative_positions = generate_motion(
        times, initial, alternative, identical_activation, identical_completion
    )
    np.testing.assert_allclose(alternative_positions, ambiguous_original, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ambiguous['positions'], ambiguous_original, rtol=0, atol=1e-12)
    assert ambiguous['rank'] == 1
    assert not np.allclose(ambiguous['displacements'], known_displacements)

    summary = {
        'scope': 'Synthetic position-only least squares with fixed times and fixed initial position',
        'units': 'seconds and meters',
        'endpoint_constraint': 'None; endpoints are recovered here because the data follow the exact model',
        'noise': 'None',
        'identifiable_case': {
            'frames': len(times),
            'initial': initial.tolist(),
            'activation': activation.tolist(),
            'completion': completion.tolist(),
            'known_displacements': known_displacements.tolist(),
            'fitted_displacements': fitted['displacements'].tolist(),
            'known_goals': known_goals.tolist(),
            'fitted_goals': fitted['goals'].tolist(),
            'rank': fitted['rank'],
            'singular_values': fitted['singular_values'].tolist(),
            'condition_number': fitted['condition_number'],
            'position_rmse_m': fitted['position_rmse_m'],
            'maximum_displacement_component_error_m': float(np.max(np.abs(
                fitted['displacements'] - known_displacements
            ))),
        },
        'nonidentifiable_case': {
            'activation': identical_activation.tolist(),
            'completion': identical_completion.tolist(),
            'known_displacements': known_displacements.tolist(),
            'alternative_displacements': alternative.tolist(),
            'fitted_displacements': ambiguous['displacements'].tolist(),
            'rank': ambiguous['rank'],
            'unique_displacements': False,
            'position_rmse_m': ambiguous['position_rmse_m'],
            'explanation': 'Identical basis columns determine only the sum of the two displacements',
        },
    }
    (output / 'summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8'
    )
    np.savez_compressed(
        output / 'trajectories.npz', timestamps=times, initial=initial,
        activation=activation, completion=completion,
        original=original, fitted=fitted['positions'], basis=fitted['basis'],
        known_displacements=known_displacements, fitted_displacements=fitted['displacements'],
        ambiguous_original=ambiguous_original, ambiguous_fitted=ambiguous['positions'],
    )
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    for i in range(2):
        axes[0].plot(times, fitted['basis'][:, i], label=f'Progress s{i + 1}')
    axes[0].set_ylabel('Progress')
    axes[0].legend()
    for coordinate, axis in enumerate(axes[1:]):
        axis.plot(times, original[:, coordinate], color='black', label='Known trajectory')
        axis.plot(times, fitted['positions'][:, coordinate], '--', color='tab:orange', label='Least squares')
        axis.set_ylabel(f'{"XYZ"[coordinate]} (m)')
    axes[1].legend()
    axes[0].set_title('Step 1: known motion with two overlapping displacements')
    axes[-1].set_xlabel('Time (s)')
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / 'known_motion_fit.png', dpi=180)
    plt.close(fig)

    print('Fixed initial position; no endpoint constraint; no noise.')
    print('Basis matrix shape:', fitted['basis'].shape)
    print('Known displacements (m):\n', known_displacements)
    print('Fitted displacements (m):\n', fitted['displacements'])
    print('Recovered goals (m):\n', fitted['goals'])
    print('Basis rank:', fitted['rank'], '/ 2')
    print('Condition number:', fitted['condition_number'])
    print('Position RMSE (m):', fitted['position_rmse_m'])
    print('Identical-time case: rank', ambiguous['rank'], '/ 2; displacement solution is not unique.')
    print('Identical-time fitted displacements (m):\n', ambiguous['displacements'])
    print('Both cases passed the expected checks. Outputs:', output.relative_to(ROOT))


if __name__ == '__main__':
    main()
