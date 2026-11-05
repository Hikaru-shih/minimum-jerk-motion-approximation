import numpy as np
from scipy.optimize import least_squares

from .fixed_times import fit_fixed_endpoints


def decode_times(parameters, first_time, last_time, minimum_duration):
    parameters = np.asarray(parameters, dtype=float)
    span = last_time - first_time
    if not np.isfinite(span) or not 0 < minimum_duration < span:
        raise ValueError('Minimum duration must be positive and shorter than the observation span')
    if parameters.ndim != 2 or parameters.shape[1] != 2 or not len(parameters):
        raise ValueError('Expected timing parameters (K, 2)')
    if not np.isfinite(parameters).all() or np.any((parameters < 0) | (parameters > 1)):
        raise ValueError('Timing parameters must be finite and lie in [0, 1]')
    activation = first_time + (span - minimum_duration) * parameters[:, 0]
    remaining = np.maximum(last_time - activation - minimum_duration, 0)
    completion = activation + minimum_duration + remaining * parameters[:, 1]
    completion = np.minimum(completion, last_time)
    return activation, completion


def encode_times(activation, completion, first_time, last_time, minimum_duration):
    activation = np.asarray(activation, dtype=float)
    completion = np.asarray(completion, dtype=float)
    span = last_time - first_time
    if not np.isfinite(span) or not 0 < minimum_duration < span:
        raise ValueError('Invalid minimum duration')
    if activation.ndim != 1 or not len(activation) or activation.shape != completion.shape:
        raise ValueError('Expected matching timing vectors')
    if not np.isfinite(activation).all() or not np.isfinite(completion).all():
        raise ValueError('Times must be finite')
    if (np.any(activation < first_time) or np.any(completion > last_time)
            or np.any(completion - activation < minimum_duration - 1e-12)):
        raise ValueError('Initial times violate observation bounds or minimum duration')
    remaining = np.maximum(last_time - activation - minimum_duration, 0)
    duration_fraction = np.divide(
        completion - activation - minimum_duration, remaining,
        out=np.zeros_like(remaining), where=remaining > 0,
    )
    return np.clip(np.column_stack([
        (activation - first_time) / (span - minimum_duration), duration_fraction,
    ]), 0, 1)


def fit_timing(times, positions, initial, final, activation, completion,
               minimum_duration, max_nfev=1000):
    times = np.asarray(times, dtype=float)
    initial_fit = fit_fixed_endpoints(times, positions, initial, final, activation, completion)
    parameters = encode_times(activation, completion, times[0], times[-1], minimum_duration)
    component_count = len(parameters)
    evaluations = 0

    def residual(flat_parameters):
        nonlocal evaluations
        evaluations += 1
        start, end = decode_times(
            flat_parameters.reshape(component_count, 2), times[0], times[-1], minimum_duration
        )
        fitted = fit_fixed_endpoints(times, positions, initial, final, start, end)
        return (fitted['positions'] - positions).ravel()

    optimized = least_squares(
        residual, parameters.ravel(), bounds=(0, 1), method='trf', jac='3-point',
        loss='linear', x_scale='jac', max_nfev=max_nfev,
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
    )
    start, end = decode_times(
        optimized.x.reshape(component_count, 2), times[0], times[-1], minimum_duration
    )
    fitted = fit_fixed_endpoints(times, positions, initial, final, start, end)
    order = np.argsort(start, kind='stable')
    return {
        'activation': start[order], 'completion': end[order],
        'displacements': fitted['displacements'][order],
        'positions': fitted['positions'],
        'initial_position_rmse_m': initial_fit['position_rmse_m'],
        'position_rmse_m': fitted['position_rmse_m'],
        'initial_error_m': fitted['initial_error_m'],
        'final_error_m': fitted['final_error_m'],
        'success': bool(optimized.success), 'status': int(optimized.status),
        'message': str(optimized.message), 'nfev': int(optimized.nfev),
        'residual_evaluations': evaluations, 'optimality': float(optimized.optimality),
        'free_displacement_rank': fitted['free_rank'],
        'free_displacement_columns': fitted['free_columns'],
        'timing_jacobian_rank': int(np.linalg.matrix_rank(optimized.jac)),
        'timing_parameter_count': 2 * component_count,
        'timing_jacobian_singular_values': np.linalg.svd(optimized.jac, compute_uv=False),
    }
