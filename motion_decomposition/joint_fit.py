import numpy as np
from scipy.optimize import least_squares

from .fixed_times import design_matrix, fit_fixed_endpoints
from .opposition import opposition_residuals
from .timing_fit import decode_times, encode_times


def fit_joint(times, positions, initial, final, activation, completion,
              minimum_duration, weight, length_scale, max_nfev=1000):
    if not np.isfinite(weight) or weight < 0:
        raise ValueError('Weight must be finite and nonnegative')
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError('Length scale must be finite and positive')
    base = fit_fixed_endpoints(times, positions, initial, final, activation, completion)
    times, positions = np.asarray(times, dtype=float), np.asarray(positions, dtype=float)
    initial, final = np.asarray(initial, dtype=float), np.asarray(final, dtype=float)
    parameters = encode_times(activation, completion, times[0], times[-1], minimum_duration)
    count, samples = len(parameters), len(times)
    timing_size = 2*count
    delta = final-initial
    start_vector = np.concatenate([parameters.ravel(), base['displacements'][:-1].ravel()])
    lower = np.concatenate([np.zeros(timing_size), np.full(3*(count-1), -np.inf)])
    upper = np.concatenate([np.ones(timing_size), np.full(3*(count-1), np.inf)])
    evaluations = 0

    def unpack(vector):
        a, b = decode_times(vector[:timing_size].reshape(count, 2),
                            times[0], times[-1], minimum_duration)
        free = vector[timing_size:].reshape(count-1, 3)
        d = np.vstack([free, delta-free.sum(axis=0)])
        p = initial+design_matrix(times, a, b) @ d
        return a, b, d, p

    def residual(vector):
        nonlocal evaluations
        evaluations += 1
        a, b, d, p = unpack(vector)
        opposition = opposition_residuals(times, a, b, d, length_scale)
        return np.concatenate([(p-positions).ravel()/np.sqrt(samples),
                               np.sqrt(weight/samples)*opposition])

    initial_residual = residual(start_vector)
    result = least_squares(residual, start_vector, bounds=(lower, upper),
                           method='trf', jac='3-point', x_scale='jac', loss='linear',
                           ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=max_nfev)
    a, b, d, p = unpack(result.x)
    opposition = opposition_residuals(times, a, b, d, length_scale)
    mse = float(np.mean(np.sum((p-positions)**2, axis=1)))
    penalty = float(np.sum(opposition**2)/samples)
    order = np.argsort(a, kind='stable')
    return {'activation': a[order], 'completion': b[order], 'displacements': d[order],
            'positions': p, 'position_rmse_m': np.sqrt(mse),
            'opposition_cost_m2': penalty, 'objective_m2': mse+weight*penalty,
            'initial_objective_m2': float(initial_residual @ initial_residual),
            'sum_displacement_norms_m': float(np.linalg.norm(d, axis=1).sum()),
            'initial_error_m': float(np.linalg.norm(p[0]-initial)),
            'final_error_m': float(np.linalg.norm(p[-1]-final)),
            'success': bool(result.success), 'status': int(result.status),
            'message': str(result.message), 'nfev': int(result.nfev),
            'residual_evaluations': evaluations, 'optimality': float(result.optimality),
            'joint_jacobian_rank': int(np.linalg.matrix_rank(result.jac)),
            'parameter_count': len(start_vector)}
