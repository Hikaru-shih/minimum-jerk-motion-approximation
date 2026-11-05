import numpy as np


def trajectory_errors(original, approximated, times):
    original = np.asarray(original, dtype=float)
    approximated = np.asarray(approximated, dtype=float)
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 3:
        raise ValueError('Expected at least three timestamps')
    if original.shape != (len(times), 3) or approximated.shape != original.shape:
        raise ValueError('Expected two trajectories with shape (N, 3)')
    if not all(np.isfinite(a).all() for a in (original, approximated, times)):
        raise ValueError('Trajectories and timestamps must be finite')
    if np.any(np.diff(times) <= 0):
        raise ValueError('Timestamps must increase strictly')
    position_error = np.linalg.norm(approximated - original, axis=1)
    original_velocity = np.gradient(original, times, axis=0, edge_order=2)
    approximated_velocity = np.gradient(approximated, times, axis=0, edge_order=2)
    velocity_error = approximated_velocity - original_velocity
    return {
        'position_rmse_m': float(np.sqrt(np.mean(position_error**2))),
        'velocity_rmse_m_s': float(np.sqrt(np.mean(np.sum(velocity_error**2, axis=1)))),
        'max_position_error_m': float(position_error.max()),
    }, position_error
