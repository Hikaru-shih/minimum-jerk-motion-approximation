import numpy as np
from .fixed_times import design_matrix


def speed_profile(positions, times):
    positions = np.asarray(positions, dtype=float)
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 3 or positions.shape != (len(times), 3):
        raise ValueError('Expected at least three timestamps and positions (N, 3)')
    if not np.isfinite(positions).all() or not np.isfinite(times).all():
        raise ValueError('Positions and times must be finite')
    if np.any(np.diff(times) <= 0):
        raise ValueError('Timestamps must increase strictly')
    velocity = np.gradient(positions, times, axis=0, edge_order=2)
    return velocity, np.linalg.norm(velocity, axis=1)


def local_minimum_runs(speed):
    speed = np.asarray(speed, dtype=float)
    if speed.ndim != 1 or len(speed) < 3:
        raise ValueError('Expected at least three speed samples')
    if not np.isfinite(speed).all() or np.any(speed < 0):
        raise ValueError('Speed must be finite and nonnegative')
    minima = []
    start = 1
    while start < len(speed) - 1:
        end = start
        while end + 1 < len(speed) and speed[end + 1] == speed[start]:
            end += 1
        if (end < len(speed) - 1 and speed[start] < speed[start - 1]
                and speed[end] < speed[end + 1]):
            minima.append((start, end))
        start = end + 1
    return np.asarray(minima, dtype=int).reshape(-1, 2)


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
