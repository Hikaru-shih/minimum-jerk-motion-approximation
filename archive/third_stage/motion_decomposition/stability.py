from itertools import permutations

import numpy as np

from .step4_diagnose_initialization import component_velocities


def align_components(times, reference, candidate):
    ref = component_velocities(times, reference['activation'], reference['completion'], reference['displacements'])
    other = component_velocities(times, candidate['activation'], candidate['completion'], candidate['displacements'])
    if ref.shape != other.shape:
        raise ValueError('Comparison requires the same component count')
    count = ref.shape[1]
    if count > 6:
        raise ValueError('Exhaustive alignment is limited to six components')
    cost = np.mean(np.sum((ref[:,:,None,:]-other[:,None,:,:])**2,axis=3),axis=0)
    order = min(permutations(range(count)), key=lambda p: (sum(cost[i,p[i]] for i in range(count)),p))
    return np.asarray(order), float(sum(cost[i,order[i]] for i in range(count)))


def compare_aligned(times, reference, candidate):
    order, cost = align_components(times,reference,candidate)
    d_difference = np.asarray(candidate['displacements'])[order]-reference['displacements']
    a_difference = np.asarray(candidate['activation'])[order]-reference['activation']
    b_difference = np.asarray(candidate['completion'])[order]-reference['completion']
    trajectory_difference = np.asarray(candidate['positions'])-reference['positions']
    return {'alignment':order,'velocity_signature_rmse_m_s':np.sqrt(cost),
            'maximum_activation_difference_s':float(np.max(np.abs(a_difference))),
            'maximum_completion_difference_s':float(np.max(np.abs(b_difference))),
            'maximum_displacement_vector_difference_m':float(np.max(np.linalg.norm(d_difference,axis=1))),
            'trajectory_difference_rmse_m':float(np.sqrt(np.mean(np.sum(trajectory_difference**2,axis=1)))),
            'maximum_trajectory_difference_m':float(np.max(np.linalg.norm(trajectory_difference,axis=1)))}
