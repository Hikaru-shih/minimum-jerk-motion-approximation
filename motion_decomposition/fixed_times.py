import numpy as np


def design_matrix(times, activation, completion):
    times = np.asarray(times, dtype=float)
    activation = np.asarray(activation, dtype=float)
    completion = np.asarray(completion, dtype=float)

    if times.ndim != 1 or len(times) < 2:
        raise ValueError('Expected at least two timestamps')
    if activation.ndim != 1 or not len(activation):
        raise ValueError('Expected at least one activation time')
    if completion.shape != activation.shape:
        raise ValueError('Activation and completion shapes must match')
    if not all(np.isfinite(a).all() for a in (times, activation, completion)):
        raise ValueError('Times must be finite')
    if np.any(np.diff(times) <= 0):
        raise ValueError('Timestamps must increase strictly')
    if np.any(completion <= activation):
        raise ValueError('Completion must follow activation')

    u = np.clip(
        (times[:, None] - activation) / (completion - activation), 0.0, 1.0
    )
    return u**3 * (10.0 + u * (-15.0 + 6.0 * u))


def fit_displacements(times, positions, initial, activation, completion):
    basis = design_matrix(times, activation, completion)
    positions = np.asarray(positions, dtype=float)
    initial = np.asarray(initial, dtype=float)
    if positions.shape != (len(basis), 3) or initial.shape != (3,):
        raise ValueError('Expected positions (N, 3) and initial position (3,)')
    if not np.isfinite(positions).all() or not np.isfinite(initial).all():
        raise ValueError('Positions must be finite')

    target = positions - initial
    displacements, _, rank, singular_values = np.linalg.lstsq(
        basis, target, rcond=None
    )
    reconstructed = initial + basis @ displacements
    residual = reconstructed - positions
    full_rank = rank == basis.shape[1]
    return {
        'basis': basis,
        'displacements': displacements,
        'goals': initial + np.cumsum(displacements, axis=0),
        'positions': reconstructed,
        'rank': int(rank),
        'singular_values': singular_values,
        'full_column_rank': bool(full_rank),
        'condition_number': (
            float(singular_values[0] / singular_values[-1])
            if full_rank else float('inf')
        ),
        'position_rmse_m': float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
    }


def fit_fixed_endpoints(times, positions, initial, final, activation, completion):
    basis = design_matrix(times, activation, completion)
    positions = np.asarray(positions, dtype=float)
    initial = np.asarray(initial, dtype=float)
    final = np.asarray(final, dtype=float)
    if positions.shape != (len(basis), 3) or initial.shape != (3,) or final.shape != (3,):
        raise ValueError('Expected positions (N, 3) and endpoints (3,)')
    if not all(np.isfinite(a).all() for a in (positions, initial, final)):
        raise ValueError('Positions and endpoints must be finite')
    if not np.all(basis[0] == 0) or not np.all(basis[-1] == 1):
        raise ValueError('All components must be inactive at the first sample and complete at the last')

    delta = final - initial
    free_basis = basis[:, :-1] - basis[:, -1:]
    target = positions - initial - basis[:, -1:] * delta
    if free_basis.shape[1]:
        free_displacements, _, rank, singular_values = np.linalg.lstsq(
            free_basis, target, rcond=None
        )
    else:
        free_displacements = np.empty((0, 3))
        rank, singular_values = 0, np.empty(0)
    last_displacement = delta - free_displacements.sum(axis=0)
    displacements = np.vstack([free_displacements, last_displacement])
    reconstructed = initial + basis @ displacements
    full_rank = rank == free_basis.shape[1]
    return {
        'basis': basis,
        'free_basis': free_basis,
        'displacements': displacements,
        'goals': initial + np.cumsum(displacements, axis=0),
        'positions': reconstructed,
        'free_rank': int(rank),
        'free_columns': free_basis.shape[1],
        'full_column_rank': bool(full_rank),
        'singular_values': singular_values,
        'condition_number': (
            None if not len(singular_values) else
            float(singular_values[0] / singular_values[-1]) if full_rank else float('inf')
        ),
        'position_rmse_m': float(np.sqrt(np.mean(np.sum((reconstructed - positions)**2, axis=1)))),
        'initial_error_m': float(np.linalg.norm(reconstructed[0] - initial)),
        'final_error_m': float(np.linalg.norm(reconstructed[-1] - final)),
        'displacement_sum_error_m': float(np.linalg.norm(displacements.sum(axis=0) - delta)),
    }
