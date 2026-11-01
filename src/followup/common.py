import numpy as np

from ..approximate_wrist import reconstruct


class NoEligibleGoal(ValueError):
    """No source frame satisfies the required separation."""


def trajectory_errors(original, approximated, times):
    original = np.asarray(original, dtype=np.float64)
    approximated = np.asarray(approximated, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    if times.ndim != 1 or len(times) < 3:
        raise ValueError('Expected at least three timestamps')
    if original.shape != (len(times), 3) or approximated.shape != original.shape:
        raise ValueError('Both trajectories must have shape (number of timestamps, 3)')
    if not all(np.isfinite(array).all() for array in (original, approximated, times)):
        raise ValueError('Trajectories and timestamps must be finite')
    if np.any(np.diff(times) <= 0):
        raise ValueError('Timestamps must increase strictly')

    position_error = np.linalg.norm(approximated - original, axis=1)
    original_velocity = np.gradient(original, times, axis=0, edge_order=2)
    approximated_velocity = np.gradient(approximated, times, axis=0, edge_order=2)
    velocity_error = np.linalg.norm(approximated_velocity - original_velocity, axis=1)

    metrics = {
        'position_rmse_m': float(np.sqrt(np.mean(position_error**2))),
        'velocity_rmse_m_s': float(np.sqrt(np.mean(velocity_error**2))),
        'max_position_error_m': float(np.max(position_error)),
    }
    return metrics, position_error


def overlap_times(sample_times, overlap_fraction=0.25):
    sample_times = np.asarray(sample_times, dtype=np.float64)
    if sample_times.ndim != 1 or len(sample_times) < 2:
        raise ValueError('Expected initial and final source timestamps')
    if not np.isfinite(sample_times).all() or np.any(np.diff(sample_times) <= 0):
        raise ValueError('Source timestamps must be finite and strictly increasing')
    if not 0 <= overlap_fraction < 0.5:
        raise ValueError('Overlap fraction must be in [0, 0.5)')

    intervals = np.diff(sample_times)
    extensions = overlap_fraction * np.minimum(intervals[:-1], intervals[1:])
    activation = sample_times[:-1].copy()
    completion = sample_times[1:].copy()
    activation[1:] -= extensions
    completion[:-1] += extensions
    return activation, completion


def reconstruct_samples(original, times, indices, overlap_fraction=0.25):
    indices = np.asarray(indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('Source indices must be a one-dimensional integer array')
    if len(indices) < 2 or indices[0] != 0 or indices[-1] != len(times) - 1:
        raise ValueError('Source indices must include the initial and final frames')
    if np.any(np.diff(indices) <= 0):
        raise ValueError('Source indices must increase strictly')

    activation, completion = overlap_times(times[indices], overlap_fraction)
    positions = reconstruct(times, original[0], original[indices[1:]], activation, completion)
    return positions, activation, completion


def select_error_candidate(position_error, times, selected_indices, min_gap_s):
    if not np.isfinite(min_gap_s) or min_gap_s <= 0:
        raise ValueError('Minimum source-time gap must be positive and finite')
    eligible = np.ones(len(times), dtype=bool)
    for index in selected_indices:
        eligible &= np.abs(times - times[index]) >= min_gap_s - 1e-12
    eligible[selected_indices] = False
    if not eligible.any():
        raise NoEligibleGoal('No candidate satisfies the minimum source-time gap')
    return int(np.argmax(np.where(eligible, position_error, -np.inf)))
