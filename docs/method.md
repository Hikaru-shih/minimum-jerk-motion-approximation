# Method and evaluation

## Representation

For component i, a_i and b_i are its onset and completion, and d_i is its XYZ displacement. Define u_i(t)=clip((t-a_i)/(b_i-a_i), 0, 1) and S_i(t)=10u_i^3-15u_i^4+6u_i^5. The reconstructed position is p_hat(t)=p_0+sum_i S_i(t)d_i. Components may overlap. Cumulative goals are g_i=p_0+sum_{j<=i}d_j; they need not be positions visited during overlap. K counts components and cumulative goals, excluding the starting point.

The individual components use the minimum-jerk profile; the fitting objective does not separately minimize jerk of their sum.

## Motion-related proposals

Detection uses a third-order Savitzky-Golay derivative filter with 17 samples at 120 Hz, applied to the full clip before cropping. Speed is the norm of the filtered XYZ velocity. Fitting and evaluation use unfiltered positions.

Retain speed peaks with prominence at least 10% of the crop maximum speed and minimum separation 0.15 s. The minimum-speed sample between consecutive retained peaks provides a boundary. Crop endpoints complete the partition; one component is proposed per retained speed lobe, with one component as fallback when no peak is retained. The nominal counts are K=4 in 18-21 s and K=3 in 30-33 s.

## Endpoint-constrained fitting

For given timings, the progress matrix B is known. With delta=p_final-p_initial, eliminate the last displacement using d_K=delta-sum_{i<K}d_i. Set F_ni=B_ni-B_nK and Y_n=p_n-p_initial-B_nK*delta, then solve min_D ||FD-Y||_F^2. K=1 has no free displacement. Recompute the linear solution at every outer timing trial.

At an internal candidate boundary, the local radius r is the smaller adjacent interval width; at a crop edge, use the single adjacent width. For an interval [L,R], onset lies in [L-0.45r_L, L+0.20r_L], and completion lies in [R-0.20r_R, R+0.45r_R], clipped to the crop. These empirical bounds preserve local association while allowing overlap; they are not physiological constants.

The outer solver is bounded trust-region reflective least squares, with three-point numerical Jacobian, x_scale='jac', ftol=xtol=gtol=1e-12 and max_nfev=1000. Four starts use seed 23. The first extends interval boundaries outward by 0.25r; the others use identical normalized random draws within the feasible box. Select the lowest position-RMSE successful fit. No opposition penalty is used.

## Comparisons and sensitivity

Uniform controls use equal intervals with a 0.25-interval overlap extension, either sampled goal positions or fitted displacements. All conditions share the same raw target, position endpoints and K. The timing search uses more computation than a fixed-time linear fit, so this is not an equal-compute comparison.

Position RMSE is sqrt(mean(||p_hat-p||^2)); velocity RMSE uses the same definition on XYZ velocity vectors. Both velocities use numpy.gradient with edge_order=2. Maximum position error is the largest 3D distance, not a per-axis maximum.

Count sensitivity varies relative prominence from 0.01 to 0.40 in steps of 0.01, retaining every distinct partition. Filtering and peak separation stay fixed. Both count and boundaries change, so error need not decrease monotonically. The nominal setting is not replaced by the partition with lowest error.

Timing-neighborhood sensitivity keeps nominal boundaries and K fixed and compares outward/inward fractions (0.30,0.10), (0.45,0.20) and (0.60,0.30). Wider bounds lower position errors in the tested cases, demonstrating that the constraints matter. The first window's worse velocity and second window's better velocity relative to uniform baselines persist across these three conditions.

## Supplementary joint position and velocity objective

Use J=position_RMSE^2/L^2 + lambda*velocity_RMSE^2/V^2. L is the RMS distance of raw positions from their mean; V is RMS raw velocity magnitude. Both scales are fixed per window, with floor 1e-12. Test lambda=0, 0.01, 0.1 and 1 using unchanged nominal partitions, bounds and initializations.

Let G denote the same numerical differentiation operator used for evaluation. The inner linear solve stacks F/L with sqrt(lambda)*GF/V, and Y/L with sqrt(lambda)*GY/V. The outer solver minimizes the same joint objective and selects successful endpoint-valid fits by J. Do not compare J across different weights to select a weight.

Both tested windows show lower velocity RMSE and higher position RMSE as the tested weights increase. Raw derivatives may amplify noise. The comparison does not establish a transferable weight, a new count rule, or physiological movement units.

## Evidence

- [Main comparisons and count partitions](../results/mainline/summary.json)
- [Timing sensitivity](../results/timing_bound_sensitivity/summary.json)
- [Joint-objective fits and all starts](../results/velocity_objective/summary.json)
- [Report](../report/final_motion_decomposition_report.pdf)
