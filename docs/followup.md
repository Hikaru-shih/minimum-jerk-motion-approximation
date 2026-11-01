# Follow-up Experiment

The follow-up experiment compares fixed-interval goals, first-improvement error-driven insertion, and raw acceleration-based selection at matched goal counts on the same clip. The first matched-count comparison is complete. The original report and stage-one outputs remain unchanged.

The shared evaluation uses 3D position RMSE (m), 3D velocity-vector RMSE (m/s), and maximum position error (m), computed over all original timestamps without filtering. These scores are reported separately. The original error-driven version selects the largest eligible per-frame 3D position error unconditionally. The revised version tries that ordering but accepts only the first candidate that improves global position RMSE. Velocity is evaluated but is not used for acceptance.

For the new common timing rule, each interior source-time boundary is extended on each side by `0.25 * min(left_interval, right_interval)`. The clip endpoints remain fixed. This differs from the old constant extension for unequal sampling intervals and will also be applied to the new fixed-interval comparison group.

The first insertion demonstration and the iterative error-driven method are implemented:

```bash
python3 -m unittest discover -s tests -v
python3 -m src.followup.demo_error_goal
python3 -m src.followup.error_insertion
```

The demonstration starts with the initial and final source positions (one goal excluding the initial position), then inserts one goal at the largest eligible error. A provisional minimum source-time separation of 0.1 seconds prevents coincident or immediately adjacent selections; it is not an accuracy threshold. No error-based stopping criterion is used in this one-step demonstration.

The first seven hand-calculated and input-validation checks passed; four additional iterative-selection checks now also pass. The first inserted goal is frame 5038, at 41.983333 seconds. Position RMSE increases from 36.59 cm to 44.02 cm; velocity RMSE changes from 0.43592 to 0.43714 m/s. The maximum position error increases from 82.37 cm to 84.77 cm. Adding a goal changes both displacement and timing, so improvement is not guaranteed. This is an initial diagnostic, not evidence that one selection method outperforms another.

Results are stored separately in `results/followup/first_insertion/`: [comparison figure](../results/followup/first_insertion/comparison.png), [settings and scores](../results/followup/first_insertion/summary.json), and `trajectories.npz` with before/after positions, source indices, and timing.

## Original Unconditional Insertion

The method starts with the initial and final samples and repeatedly adds the source frame with the largest eligible 3D position error. All previous samples are retained, goals are sorted in time, and all activation/completion times are recomputed using the shared rule. The minimum source-time separation is 0.1 seconds and the overlap fraction is 0.25. Positions remain sampled from the original data, without optimization or smoothing.

The current run ends at the predeclared budget of 102 goals. It does not stop at an accuracy threshold. If no frame satisfies the minimum separation, the run terminates early with an explicit reason. Goal counts exclude the initial position and include the final position.

| Goals | Position RMSE (cm) | Velocity RMSE (m/s) |
| --- | --- | --- |
| 10 | 20.426 | 0.39064 |
| 20 | 9.994 | 0.31080 |
| 40 | 4.732 | 0.23822 |
| 60 | 2.665 | 0.20652 |
| 80 | 2.132 | 0.17863 |
| 102 | 1.604 | 0.16457 |

Across 101 insertions, position RMSE increases on seven steps. Every iteration is retained; no rising-error steps are removed, and the plot is not a running-best envelope. The overall decrease does not prove superiority over either other method, which has not yet been run with matched settings and counts.

The first added goal describes a short peak near 42 seconds, but with only two trajectories the time intervals remain very long. In the demonstration plot, the new peak displacement changes the reconstruction over much of the clip. This explains how the local intent of adding a sample can produce worse global error with the current timing rule. More samples subsequently shorten intervals and change the reconstruction, but neither position nor velocity error is guaranteed to decrease at each insertion.

Outputs under `results/followup/error_insertion/`:

- [Error versus goal count](../results/followup/error_insertion/error_vs_goals.png).
- [Every iteration as CSV](../results/followup/error_insertion/history.csv), with the added frame and source time.
- [Settings and complete history](../results/followup/error_insertion/summary.json), including the termination reason.
- `snapshots.npz`: timestamps and indices, reconstructed positions, activation times, and completion times at counts 1, 2, 10, 20, 40, 60, 80, and 102. The final reached state is also retained if the run stops early.

The later sections record the revised method and completed matched-count comparison. These original unconditional results remain available as a separate diagnostic.

## Revised First-Improvement Insertion

Run from the repository root:

```bash
python3 -m src.followup.accepted_error_insertion
```

At each accepted state, compute all eligible source frames and sort them by descending current 3D position error (ties use the earlier frame). Try them in that order. Each trial creates independent goal indices, timings, and reconstruction arrays. Accept the first candidate whose global position RMSE decreases by more than `1e-10 m`; rejected trials do not alter the accepted state. After an acceptance, recompute all errors and candidate rankings. A point rejected earlier can be reconsidered in a later round.

Use the same 0.1 s minimum source-time gap, 0.25 overlap fraction, original sampled goal positions, and 102-goal budget as the original version. The numerical improvement tolerance is not a target accuracy. Stop at the goal budget, if no eligible source frame remains, or if every eligible candidate in the current round fails to improve position RMSE. The latter does not establish a global optimum.

The run reached 102 goals after 6,283 candidate trials: 101 accepted and 6,182 rejected. In the first round, the largest-error candidate at frame 5038 was rejected; candidate rank 30 (frame 5020, 41.833333 s) was accepted, reducing position RMSE from 36.589 cm to 36.560 cm.

| At 102 goals | Original unconditional | Revised first improvement |
| --- | --- | --- |
| Position RMSE (cm) | 1.604 | 1.478 |
| Velocity RMSE (m/s) | 0.16457 | 0.17128 |
| Maximum position error (cm) | 4.402 | 5.846 |

Accepted position RMSE decreases at every step by construction. Velocity RMSE and maximum position error can increase and are not acceptance criteria. The revised version gives lower final position RMSE in this run but worse final velocity RMSE and maximum position error than the unconditional version; it is not uniformly better. See the matched-count comparison below for the fixed-interval and motion-change results.

Six new tests cover real triangular motion, all-rejected rollback, deterministic candidate ordering, reconsideration after acceptance, no eligible candidates, and the numerical improvement tolerance. Together with existing checks, all 17 tests pass.

Outputs under `results/followup/accepted_error_insertion/`:

- [Accepted-state error curves](../results/followup/accepted_error_insertion/error_vs_goals.png).
- [Accepted-state history](../results/followup/accepted_error_insertion/history.csv).
- [All candidate trials](../results/followup/accepted_error_insertion/trials.csv), including rejected trials and before/after RMSE.
- [Settings, stop reason, and history](../results/followup/accepted_error_insertion/summary.json).
- `snapshots.npz` stores indices, positions, activation times, and completion times for every accepted goal count, including the initial state. The plot shows accepted states, not rejected trials; the latter are retained in `trials.csv`.

## Motion-Change Method: Candidate Inspection

### Motivation for the report

The initial idea was to choose goals according to changes in the original motion, rather than according to reconstruction error or fixed time intervals. To make motion change measurable, acceleration magnitude was chosen as the selection score: `score(t) = ||a(t)||_2`, where acceleration is the rate of change of the velocity vector. This captures both changes in speed and changes in movement direction. It is not the rate of change of acceleration (jerk), and it is a proxy for motion change rather than a guarantee of reconstruction quality.

### Current diagnostic

```bash
python3 -m src.followup.motion_change
```

The current implementation computes velocity and acceleration with `numpy.gradient` on the original timestamps (`edge_order=2` for both derivatives), without smoothing. Candidate times are positive interior local maxima of acceleration magnitude. An exact flat peak uses its earliest frame; endpoints are retained separately. Candidates are ranked by descending score, with earlier frames breaking ties. Starting from the endpoints, select candidates at least 0.1 s from every selected source timestamp until reaching the 102-goal budget or exhausting candidates. No amplitude threshold or approximation-error acceptance test is applied.

The raw score has 1,945 candidate peaks. The selected set reaches 102 goals including the final position and excluding the initial position. The largest candidate is at 31.441667 s with an estimated acceleration magnitude of 144.35 m/s^2. High derivative values are diagnostic outputs, not independently validated physical accelerations. Candidate peaks cluster near some motion transitions, but some samples are also selected in relatively steady regions. This does not establish which fluctuations are real movement, fitting artifacts, or noise.

Candidate inspection was performed first. After reviewing it, the raw, unsmoothed rule was retained for the matched-count comparison below. No smoothing or error-based acceptance was added to the motion-change selector. Any future smoothed variant must be labeled and evaluated separately.

- [Full-clip diagnostic](../results/followup/motion_change_diagnostic/candidates_full.png)
- [17-23 s detail](../results/followup/motion_change_diagnostic/candidates_detail.png)
- [Ranked candidate table](../results/followup/motion_change_diagnostic/candidates.csv)
- [Settings and top candidates](../results/followup/motion_change_diagnostic/summary.json)
- `candidates.npz` contains timestamps, raw score/velocity/acceleration, ranked peak indices, selected indices, and insertion order. Future matched-count runs can use prefixes of this insertion order, without reranking from approximation error.

Five checks cover known constant acceleration, constant velocity, peak plateaus and boundaries, ranking with spacing, and candidate exhaustion without arbitrary fill points. All 22 existing and new tests pass.

## Matched-Count Comparison

```bash
python3 -m src.followup.compare_methods
```

The first comparison evaluates all three methods at every integer goal count from 1 to 102 (306 states, no unavailable budgets). All use the same raw wrist trajectory, sampled original goal positions, fixed endpoints, 0.1 s minimum source-time gap, 0.25 neighbor-dependent overlap fraction, and full-timestamp position and velocity evaluation. Initial position is excluded from goal counts; final position is included.

For the motion method, compute the raw acceleration-peak ranking once and retain the first eligible candidates in that order. Smaller budgets use prefixes of the same addition order. Reconstruction errors do not affect selection or acceptance. For the error method, re-evaluate every saved accepted state and check its positions and all three metrics against the recorded run before using it.

For the fixed baseline, distribute exactly `K+1` source indices uniformly across the full clip, including both endpoints, and round to the nearest source frame. Each count is selected independently. This baseline uses the common timing rule. It is not the original stage-one 0.5 s sampling: the sampling locations also differ because stage one appended a shorter final interval, whereas this matched-count baseline distributes the intervals across the full clip. Consequently, its results must not be labeled as unchanged stage-one results. The stage-one files and report are preserved.

### Results at 102 goals

| Method | Position RMSE (cm) | Velocity RMSE (m/s) | Maximum position error (cm) |
| --- | --- | --- | --- |
| Fixed interval with shared timing | 3.817 | 0.28461 | 31.204 |
| Error-driven first improvement | 1.478 | 0.17128 | 5.846 |
| Raw acceleration magnitude | 5.531 | 0.24271 | 19.567 |

At this budget, the error-driven method has the lowest values of all three metrics. The motion method has worse position RMSE than the fixed baseline but lower velocity RMSE and lower maximum position error. Its score therefore captures some useful changes without necessarily allocating goals to minimize the overall position error. The raw selection inspection also showed samples in relatively steady regions and gaps elsewhere; the current comparison alone does not prove that noise caused its errors.

These are same-clip reconstruction results, not held-out generalization. The methods have equal representation size, not equal computation: the error method directly tests position RMSE and required 6,283 candidate trials, whereas the motion method ranks input features without reconstruction-error acceptance. Position-RMSE monotonicity is enforced only by the first-improvement method; neither other method is required to improve at every count. No parameters were retuned for these results.

- [All three error curves](../results/followup/comparison/error_vs_goals.png)
- [Full-clip position comparison at 102 goals](../results/followup/comparison/position_comparison.png)
- [17-23 s position detail at 102 goals](../results/followup/comparison/position_detail.png)
- [All 306 metric records](../results/followup/comparison/metrics.csv)
- [Settings, input hash, and metric records](../results/followup/comparison/summary.json)
- `snapshots.npz` stores positions, source indices, activation times, and completion times at counts 1, 2, 10, 20, 40, 60, 80, and 102 for each method.

Three further tests check exact matched counts, infeasible fixed budgets, and motion-order prefixes without arbitrary fill points. All 25 checks pass. This is an initial three-method comparison. The [updated PDF report](../report/followup_report.pdf) includes the goal-count analysis.

## Further analysis of goal count

The goal-count analysis compares saved trajectories at 10, 20, 40, 60, 80, and 102 goals. Run `python3 -m src.followup.analyze_goal_counts` after the comparison. This verifies the 18 trajectories against the existing global metrics and writes six per-count views, six within-method position/velocity comparisons, and separately labelled global and 17-23 s window metrics to `results/followup/goal_count_analysis/`. Velocities are differentiated over the whole clip before cropping; each coordinate uses the same axis limits across all six counts.

This analysis illustrates how count changes reconstruction, without introducing an unsupported accuracy or improvement threshold. For example, error-driven full-clip position RMSE improves from 10.207 cm at 20 goals to 5.384 cm at 40, while window RMSE stays approximately 9.216 cm. From 60 to 80 goals, its global maximum position error rises from 7.281 to 8.520 cm despite lower position and velocity RMSE. These observations do not establish a unique appropriate count or an automatic stopping rule.

## Files

- `src/followup/common.py`: shared error metrics, overlap timing, reconstruction from selected frames, and error-based candidate selection.
- `src/followup/demo_error_goal.py`: one insertion demonstration; run as a module from the repository root.
- `src/followup/error_insertion.py`: original unconditional insertion and history plotting.
- `src/followup/accepted_error_insertion.py`: first-improvement selection with rejected-trial logging.
- `src/followup/motion_change.py`: raw acceleration scoring and candidate inspection.
- `src/followup/compare_methods.py`: matched-count reconstruction, validation, metrics, and plots.
- `src/followup/analyze_goal_counts.py`: saved trajectory position/velocity views and global/window metric checks.
- `tests/test_compare_methods.py`: matched-count and candidate availability checks.
- `tests/test_motion_change.py`: motion score and selection checks.
- `tests/test_followup_common.py` and `tests/test_error_insertion.py`: numerical, selection, spacing, and stopping checks.
- `results/followup/first_insertion/`: demonstration settings, scores, trajectories, and comparison figure.

The module command uses the repository's `src` package so imports resolve correctly. Do not run the nested file directly with `python3 src/followup/demo_error_goal.py`.
