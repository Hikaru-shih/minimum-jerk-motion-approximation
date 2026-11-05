import numpy as np

from .motion_utils import local_minimum_runs, speed_profile
from .timing_fit import encode_times


def times_from_knots(knots, overlap_fraction=0.25):
    knots = np.asarray(knots, dtype=float)
    if knots.ndim != 1 or len(knots) < 2 or not np.isfinite(knots).all():
        raise ValueError('Expected finite time knots including both endpoints')
    if np.any(np.diff(knots) <= 0) or not 0 <= overlap_fraction < 0.5:
        raise ValueError('Knots must increase; overlap fraction must lie in [0, 0.5)')
    gaps = np.diff(knots)
    extension = overlap_fraction * np.minimum(gaps[:-1], gaps[1:])
    activation, completion = knots[:-1].copy(), knots[1:].copy()
    activation[1:] -= extension
    completion[:-1] += extension
    return activation, completion


def speed_seed_knots(times, positions, component_count):
    times = np.asarray(times, dtype=float)
    _, speed = speed_profile(positions, times)
    if not isinstance(component_count, int) or not 1 <= component_count < len(times):
        raise ValueError('Component count must be prescribed and smaller than the sample count')
    uniform = np.linspace(times[0], times[-1], component_count + 1)
    interval = (times[-1] - times[0]) / component_count
    frames = local_minimum_runs(speed)[:, 0]
    knots = uniform.copy()
    selections = []
    for i in range(1, component_count):
        lower, upper = uniform[i] - interval / 2, uniform[i] + interval / 2
        candidates = frames[(times[frames] >= lower) & (times[frames] < upper)]
        if len(candidates):
            frame = int(candidates[np.argmin(speed[candidates])])
            knots[i] = times[frame]
        else:
            frame = None
        selections.append({'knot': i, 'slot_start_s': float(lower), 'slot_end_s': float(upper),
                           'candidate_count': len(candidates), 'frame': frame,
                           'time_s': float(knots[i]), 'fallback_uniform': frame is None})
    return uniform, knots, selections


def make_initializations(times, positions, component_count, minimum_duration,
                         seed=23, jitters_per_seed=4, jitter_fraction=0.2,
                         overlap_fraction=0.25):
    if not isinstance(jitters_per_seed, int) or jitters_per_seed < 0:
        raise ValueError('Jitter count must be a nonnegative integer')
    if not 0 <= jitter_fraction < 0.5:
        raise ValueError('Jitter fraction must lie in [0, 0.5)')
    uniform, speed_knots, selections = speed_seed_knots(times, positions, component_count)
    rng = np.random.default_rng(seed)
    starts = []
    for name, base in [('uniform', uniform), ('speed_minimum_seed', speed_knots)]:
        gaps = np.diff(base)
        radius = jitter_fraction * np.minimum(gaps[:-1], gaps[1:])
        for repetition in range(jitters_per_seed + 1):
            knots = base.copy()
            if repetition:
                knots[1:-1] += rng.uniform(-1, 1, len(radius)) * radius
            activation, completion = times_from_knots(knots, overlap_fraction)
            try:
                encode_times(activation, completion, times[0], times[-1], minimum_duration)
                valid, reason = True, None
            except ValueError as error:
                valid, reason = False, str(error)
            starts.append({'name': f'{name}_{repetition}', 'seed_type': name,
                           'knots': knots, 'activation': activation, 'completion': completion,
                           'valid': valid, 'invalid_reason': reason})
    return starts, selections


def valid_timing_result(result, endpoint_tolerance=1e-12):
    return bool(
        result['success']
        and np.isfinite(result['position_rmse_m']) and result['position_rmse_m'] >= 0
        and 0 <= result['initial_error_m'] <= endpoint_tolerance
        and 0 <= result['final_error_m'] <= endpoint_tolerance
        and all(np.isfinite(result[key]).all()
                for key in ('positions', 'activation', 'completion', 'displacements'))
        and np.all(result['completion'] > result['activation'])
    )


def choose_best_result(results):
    eligible = [i for i, result in enumerate(results) if result is not None and valid_timing_result(result)]
    if not eligible:
        raise ValueError('No successful, finite fit satisfying endpoint tolerance')
    return min(eligible, key=lambda i: (results[i]['position_rmse_m'], i))
