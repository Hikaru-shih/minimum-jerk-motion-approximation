"""Check the MJ formula, displacement sum, and saved wrist extraction.

Run in the project's WSL environment: python3 src/verify_results.py
This reads existing outputs; it does not regenerate or modify them.
"""

from unittest.mock import patch

import numpy as np
import torch
from human_body_prior.body_model.body_model import BodyModel

from approximate_wrist import minimum_jerk, reconstruct
from extract_wrist import (
    MODEL,
    ROOT,
    WRIST_INDEX,
    body_model_module,
    compatible_lbs,
    extract_positions,
    load_motion,
)


def check_minimum_jerk():
    """Check known values, time normalization, and endpoint derivatives."""

    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    actual = minimum_jerk(times, 2.0, 4.0)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)



    h = 0.001
    values = minimum_jerk(np.array([0, h, 2*h, 3*h]), 0.0, 1.0)
    end_values = minimum_jerk(np.array([1, 1-h, 1-2*h, 1-3*h]), 0.0, 1.0)
    velocities = []
    accelerations = []
    for samples in (values, end_values):
        velocities.append((-3*samples[0] + 4*samples[1] - samples[2]) / (2*h))
        accelerations.append(
            (2*samples[0] - 5*samples[1] + 4*samples[2] - samples[3]) / h**2
        )
    velocities[1] *= -1  # End samples are ordered backwards in time.
    np.testing.assert_allclose(velocities, 0, rtol=0, atol=3e-5)
    np.testing.assert_allclose(accelerations, 0, rtol=0, atol=1e-3)
    print(f'PASS MJ known values: {actual.tolist()}')
    print(f'  Endpoint velocities (finite differences): {velocities}')
    print(f'  Endpoint accelerations (finite differences): {accelerations}')


def check_displacement_sum():
    """Use literal, hand-calculated coordinates in an overlapping example."""
    initial = np.array([1.0, 2.0, 3.0])
    goals = np.array([[3.0, 2.0, 3.0], [3.0, 6.0, 3.0]])
    activation = np.array([0.0, 1.0])
    completion = np.array([2.0, 3.0])
    times = np.array([-1.0, 1.0, 1.5, 2.0, 4.0])
    # At t=1.5: first progress is s(0.75)=0.896484375,
    # second is s(0.25)=0.103515625. Displacements are (2,0,0), (0,4,0).
    expected = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 2.0, 3.0],
        [2.79296875, 2.4140625, 3.0],
        [3.0, 4.0, 3.0],
        [3.0, 6.0, 3.0],
    ])
    actual = reconstruct(times, initial, goals, activation, completion)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    print('PASS displacement sum: five hand-calculated XYZ positions match')
    print(f'  Overlap at t=1.5 s: {actual[2].tolist()}')


def check_wrist_extraction():
    """Compare individual frames, a fresh batch, and the saved full-clip output."""
    poses, trans, betas, fps, _ = load_motion()

    indices = np.array([0, 31, 32, 3000, len(poses) - 1])
    batched = extract_positions(poses[indices], trans[indices], betas)
    model = BodyModel(bm_path=str(MODEL), model_type='smplh', num_betas=16)
    singles = []
    with torch.no_grad(), patch.object(body_model_module, 'lbs', compatible_lbs):
        for index in indices:
            pose = torch.as_tensor(poses[index:index+1], dtype=torch.float32)
            body = model(
                root_orient=pose[:, :3],
                pose_body=pose[:, 3:66],
                pose_hand=pose[:, 66:156],
                betas=torch.as_tensor(betas[None, :16], dtype=torch.float32),
                trans=torch.as_tensor(trans[index:index+1], dtype=torch.float32),
            )
            singles.append(body.Jtr[0, WRIST_INDEX].cpu().numpy())
    singles = np.array(singles)
    with np.load(ROOT / 'results/wrist_trajectory.npz') as saved:
        positions = saved['positions']
        np.testing.assert_equal(positions.shape, (len(poses), 3))
        np.testing.assert_allclose(
            saved['timestamps'], np.arange(len(poses)) / fps, rtol=0, atol=1e-12
        )
        if not np.isfinite(positions).all():
            raise AssertionError('Non-finite saved positions')
        stored_samples = positions[indices]
    # Model arithmetic is float32: allow one micrometer of absolute discrepancy.
    np.testing.assert_allclose(batched, singles, rtol=0, atol=1e-6)
    np.testing.assert_allclose(stored_samples, singles, rtol=0, atol=1e-6)
    print(f'PASS wrist frames: {indices.tolist()}')
    print(f'  Max batch vs single difference: {np.max(np.abs(batched-singles)):.3e} m')
    print(f'  Max saved vs single difference: {np.max(np.abs(stored_samples-singles)):.3e} m')
    print('PASS full saved trajectory: shape, finite coordinates, and timestamps')


def main():
    check_minimum_jerk()
    check_displacement_sum()
    check_wrist_extraction()
    print('All three verification groups passed.')


if __name__ == '__main__':
    main()
