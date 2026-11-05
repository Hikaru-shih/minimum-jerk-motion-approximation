import numpy as np
from scipy.optimize import least_squares

from .fixed_times import fit_fixed_endpoints
from .step4_diagnose_initialization import component_velocities


def opposition_residuals(times, activation, completion, displacements, length_scale):
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError('Length scale must be finite and positive')
    times = np.asarray(times, dtype=float)
    velocities = component_velocities(times, activation, completion, displacements)
    scale = (times[-1]-times[0])**2 / length_scale
    residuals = []
    for i in range(velocities.shape[1]):
        for j in range(i+1, velocities.shape[1]):
            dot = np.sum(velocities[:, i]*velocities[:, j], axis=1)
            residuals.append(scale*np.maximum(0, -dot))
    return np.concatenate(residuals) if residuals else np.empty(0)


def fit_opposition_fixed_times(times, positions, initial, final, activation, completion,
                               weight, length_scale, max_nfev=2000):
    if not np.isfinite(weight) or weight < 0:
        raise ValueError('Weight must be finite and nonnegative')
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError('Length scale must be finite and positive')
    base = fit_fixed_endpoints(times, positions, initial, final, activation, completion)
    times, positions = np.asarray(times), np.asarray(positions)
    initial, final = np.asarray(initial), np.asarray(final)
    count, sample_count = len(activation), len(times)
    delta = final-initial

    def unpack(free):
        first = free.reshape(count-1, 3)
        return np.vstack([first, delta-first.sum(axis=0)])

    def residual(free):
        displacements = unpack(free)
        reconstructed = initial+base['basis'] @ displacements
        opposition = opposition_residuals(times, activation, completion, displacements, length_scale)
        return np.concatenate([(reconstructed-positions).ravel()/np.sqrt(sample_count),
                               np.sqrt(weight/sample_count)*opposition])

    if weight == 0 or count == 1:
        displacements = base['displacements']
        success, message, nfev = True, 'Endpoint-constrained linear solution', 0
    else:
        result = least_squares(residual, base['displacements'][:-1].ravel(),
                               jac='3-point', x_scale='jac', max_nfev=max_nfev,
                               ftol=1e-12, xtol=1e-12, gtol=1e-12)
        displacements = unpack(result.x)
        success, message, nfev = bool(result.success), str(result.message), int(result.nfev)
    reconstructed = initial+base['basis'] @ displacements
    position_mse = float(np.mean(np.sum((reconstructed-positions)**2, axis=1)))
    penalty = opposition_residuals(times, activation, completion, displacements, length_scale)
    opposition_cost = float(np.sum(penalty**2)/sample_count)
    return {'positions': reconstructed, 'displacements': displacements,
            'position_rmse_m': np.sqrt(position_mse), 'opposition_cost_m2': opposition_cost,
            'objective_m2': position_mse+weight*opposition_cost,
            'sum_displacement_norms_m': float(np.linalg.norm(displacements, axis=1).sum()),
            'initial_error_m': float(np.linalg.norm(reconstructed[0]-initial)),
            'final_error_m': float(np.linalg.norm(reconstructed[-1]-final)),
            'success': success, 'message': message, 'nfev': nfev}
