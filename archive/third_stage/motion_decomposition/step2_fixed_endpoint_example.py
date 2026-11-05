import json
from pathlib import Path

import numpy as np

from .fixed_times import fit_fixed_endpoints
from .step1_known_motion import generate_motion

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]


def main():
    times = np.array([0, 1, 2, 3], dtype=float)
    initial, final = np.array([2, 0, 0]), np.array([8, 0, 0])
    positions = np.array([[2, 0, 0], [5, 0, 0], [7, 0, 0], [8, 0, 0]])
    result = fit_fixed_endpoints(times, positions, initial, final, [0, 1], [2, 3])
    np.testing.assert_allclose(result['displacements'], [[5.0, 0, 0], [1.0, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(result['positions'][:, 0], [2, 4.5, 7.5, 8], atol=1e-12)
    np.testing.assert_allclose(result['free_basis'].T @ (result['positions'] - positions), 0, atol=1e-12)

    known_times = np.arange(409) / 120.0
    known_initial = np.array([0.2, -0.1, 0.4])
    known_d = np.array([[0.3, 0.1, -0.05], [-0.12, 0.2, 0.08]])
    known_final = known_initial + known_d.sum(axis=0)
    known_positions = generate_motion(known_times, known_initial, known_d, [0.2, 1.2], [2.2, 3.2])
    known = fit_fixed_endpoints(known_times, known_positions, known_initial, known_final,
                              [0.2, 1.2], [2.2, 3.2])
    np.testing.assert_allclose(known['displacements'], known_d, rtol=0, atol=1e-12)
    for fitted in (result, known):
        assert fitted['initial_error_m'] < 1e-12
        assert fitted['final_error_m'] < 1e-12

    output = ROOT / 'results/step2_fixed_endpoint_example'
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        'scope': 'Synthetic fixed-endpoint checks only; not the replacement real-data experiment',
        'method': 'Eliminate last displacement using the endpoint equality; solve all effects jointly',
        'hand_example': {
            'initial': initial.tolist(), 'final': final.tolist(),
            'timestamps': times.tolist(), 'observed_x': positions[:, 0].tolist(),
            'fitted_x': result['positions'][:, 0].tolist(),
            'fitted_displacements': result['displacements'].tolist(),
            'position_rmse_m': result['position_rmse_m'],
            'initial_error_m': result['initial_error_m'],
            'final_error_m': result['final_error_m'],
        },
        'known_3d_example': {
            'known_displacements': known_d.tolist(),
            'fitted_displacements': known['displacements'].tolist(),
            'position_rmse_m': known['position_rmse_m'],
            'initial_error_m': known['initial_error_m'],
            'final_error_m': known['final_error_m'],
            'free_rank': known['free_rank'], 'free_columns': known['free_columns'],
        },
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Hand example: initial X=2, final X=8; d1+d2=6.')
    print('Observed X:', positions[:, 0])
    print('Fitted X:', result['positions'][:, 0])
    print('Fitted displacements:', result['displacements'][:, 0])
    print('Endpoint errors (m):', result['initial_error_m'], result['final_error_m'])
    print('Known 3D displacement recovery RMSE (m):', known['position_rmse_m'])
    print('Checks passed. Saved:', output.relative_to(ROOT))


if __name__ == '__main__':
    main()
