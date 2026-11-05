import numpy as np


def generate_motion(times, initial, displacements, activation, completion):
    positions = np.tile(initial, (len(times), 1)).astype(float)
    for displacement, start, end in zip(displacements, activation, completion):
        u = np.clip((times - start) / (end - start), 0.0, 1.0)
        progress = 10 * u**3 - 15 * u**4 + 6 * u**5
        positions += progress[:, None] * displacement
    return positions
