# Minimum-Jerk Motion Approximation

This project approximates the **right-wrist trajectory from one motion capture clip** using a sequence of overlapping minimum jerk trajectories. It includes the extraction and reconstruction code, position and velocity comparison figures, and a report describing goal selection and limitations.

The initial fixed-interval experiment and the first follow-up comparison are complete. The follow-up compares fixed-interval sampling, error-driven first-improvement insertion, and raw acceleration-based selection at 1-102 goals. No learning model or full-body motion generation system is implemented.

The follow-up selects source-sample times from the motion, but does not yet select a final goal count automatically. This run sweeps prescribed counts up to 102; the error-driven method reaches that cap rather than its no-improvement stopping condition. Activation and completion times use a shared overlap heuristic, rather than being independently fitted to motion events.

## Reports and Results

- [First experiment report (PDF)](report/report.pdf)
- [Follow-up report (PDF, including goal-count analysis)](report/followup_report.pdf)
- [Follow-up error curves at matched goal counts](results/followup/comparison/error_vs_goals.png)
- [Follow-up position comparison at 102 goals](results/followup/comparison/position_detail.png)
- [Follow-up numerical results](results/followup/comparison/metrics.csv)

The following figures and observations are from the first, fixed-interval experiment:

- [Position comparison](results/figures/position_comparison.png)
- [Position comparison detail, 17-23 seconds](results/figures/position_comparison_detail.png)
- [Velocity comparison](results/figures/velocity_comparison.png)

![Position comparison detail](results/figures/position_comparison_detail.png)

The reconstruction follows many broad position changes and relatively steady portions of the source trajectory. It misses some brief peaks and rapid changes, including the Z peak near 17.3 seconds and X oscillations around 21.5-22 seconds. Velocity peak amplitudes and timing also differ.

These observations concern one wrist, one clip, and one goal/timing setting. They do not establish a general limit of minimum jerk approximation.

## Motion Data and Extraction

| Item | Value |
| --- | --- |
| Dataset | ACCAD subset of AMASS |
| Clip | `Female1Gestures_c3d/D3 - Conversation Gestures_poses.npz` |
| Frames | 6,086 |
| Frame rate | 120 Hz |
| Duration | Approximately 50.7 seconds |
| Body model | Female SMPL+H, 16 shape coefficients |
| Joint | Right wrist, index 21 (zero-based); parent joint 19 |
| Coordinates | Global XYZ positions in meters, including root orientation and translation |

`src/extract_wrist.py` processes **all frames** in batches of 32 and saves a `(6086, 3)` position array. No temporal cropping, resampling, or smoothing is applied. In this project, the original trajectory means wrist positions computed from the clip's pose, shape, root orientation, and translation parameters using SMPL+H.

Timestamps are `t_k = k / 120`, from 0 to 50.708333 seconds. The frame-count duration, `6086 / 120`, is 50.716667 seconds; the final sample timestamp is one sample interval shorter.

The original data can also be viewed as a [3D path](results/figures/original_trajectory_3d.png) and [XYZ positions over time](results/figures/original_position.png).

## First Experiment Goal Selection and Reconstruction

Goal positions are sampled every **0.5 seconds** (60 frames). Frame 0 supplies the initial position, and the last frame is always included. This produces **102 goals in addition to the initial position**. The final source sampling interval is shorter than 0.5 seconds.

For consecutive sampled positions `g_(i-1)` and `g_i` with source timestamps `t_(i-1)` and `t_i`, activation and completion times are:

```text
a_i = max(first_timestamp, t_(i-1) - 0.125)
b_i = min(last_timestamp, t_i + 0.125)
```

Neighboring instances overlap by **0.25 seconds**. The sampling interval and overlap are simple initial experimental choices, not optimized parameters or settings prescribed by the reference paper. Source timestamps identify where goal positions were sampled; they are not instance completion times.

Following the position-displacement formulation of Sato et al. (2025), each instance uses normalized time `u = (t - a_i) / (b_i - a_i)`:

```text
s_i(t) = 0                         if t < a_i
         10u^3 - 15u^4 + 6u^5      if a_i <= t <= b_i
         1                         if t > b_i

p_hat(t) = g_0 + sum_i [(g_i - g_(i-1)) * s_i(t)]
```

The reconstruction sums overlapping displacements rather than sequentially concatenating rest-to-rest segments. Although each individual instance starts and ends with zero velocity and acceleration, the combined trajectory need not stop at intermediate goals. It is not constrained to pass exactly through every goal at its source timestamp or completion time.

Both trajectories are evaluated at the original timestamps. Velocity is estimated with `numpy.gradient`, using centered interior differences and second-order one-sided endpoint differences (`edge_order=2`). The same method is applied to both trajectories, without filtering. Differentiation emphasizes small fluctuations, and these comparisons do not determine whether the source fluctuations reflect movement, fitting artifacts, or noise.

## Running the First Experiment

The scripts have been run successfully in Ubuntu WSL with Python 3.10.12, `human-body-prior==0.8.5.0`, and `smplx==0.1.28`. They also require NumPy, PyTorch, and Matplotlib. A complete dependency lockfile and a tested fresh-environment installation procedure are not yet provided.

A scoped adapter in `extract_wrist.py` handles the installed packages' `lbs` signature mismatch: the older caller supplies `dtype`, while the installed `smplx` implementation infers it from the input tensor. Installed packages are not modified.

Obtain the motion data and body model separately under their respective licenses and place them at:

```text
ACCAD/Female1Gestures_c3d/D3 - Conversation Gestures_poses.npz
smplh/female/model.npz
```

With the working Python environment active, run from the repository root:

```bash
python3 src/inspect_motion.py
python3 src/extract_wrist.py
python3 src/plot_wrist.py
python3 src/approximate_wrist.py
python3 src/verify_results.py
```

All scripts resolve input and output paths relative to their own file location. Extraction and reconstruction overwrite their corresponding result files, and plotting overwrites the figures. Verification reads the saved trajectory without changing experiment outputs.

| Script | Purpose |
| --- | --- |
| `src/inspect_motion.py` | Print array shapes, frame count, frame rate, and duration |
| `src/extract_wrist.py` | Extract and save the complete right-wrist trajectory |
| `src/plot_wrist.py` | Plot the original 3D path and XYZ positions |
| `src/approximate_wrist.py` | Select goals, reconstruct positions, estimate velocities, and plot comparisons |
| `src/verify_results.py` | Check known formula values, hand-calculated displacement sums, and extraction consistency |

## Verification

The following checks passed in the working WSL environment:

- Progress values before/start/midpoint/end/after a 2-4 second instance match `[0, 0, 0.5, 1, 1]` within absolute tolerance `1e-12`.
- For a unit-duration instance, one-sided finite differences with `h=0.001` give endpoint velocities about `-1.99e-5` and accelerations about `+/-3.29e-4`, within tolerances `3e-5` and `1e-3`. These are numerical estimates of zero endpoint derivatives, not measured wrist velocities or accelerations.
- Five positions in a two-instance overlapping example match hand-calculated XYZ values within `1e-12`.
- Frames 0, 31, 32, 3000, and 6085 were computed individually and in a batch and compared with the saved trajectory. The maximum absolute coordinate discrepancy was approximately `5.96e-8 m`, below the `1e-6 m` tolerance for float32 model arithmetic.
- The saved trajectory has the expected shape, finite positions, and timestamps equal to frame indices divided by frame rate.

These checks support the tested calculations and extraction consistency. They do not independently establish anatomical ground truth or prove that the selected goals give an accurate approximation.

## Repository Contents

```text
minimum-jerk-motion-approximation/
|-- README.md
|-- requirements-followup.txt          Direct follow-up dependency versions
|-- docs/
|   `-- followup.md                    Second-stage method and progress notes
|-- report/
|   |-- report.pdf                     First-stage report
|   `-- followup_report.pdf            Second-stage report with goal-count analysis
|-- src/
|   |-- __init__.py
|   |-- inspect_motion.py              Read source metadata
|   |-- extract_wrist.py               Extract the shared wrist trajectory
|   |-- plot_wrist.py                  Plot the original trajectory
|   |-- approximate_wrist.py           First-stage fixed-interval experiment
|   |-- verify_results.py              First-stage verification
|   `-- followup/
|       |-- __init__.py
|       |-- common.py                  Shared metrics and timing
|       |-- demo_error_goal.py         First insertion demonstration
|       |-- error_insertion.py         Original unconditional insertion
|       |-- accepted_error_insertion.py  Revised first-improvement insertion
|       |-- motion_change.py           Acceleration candidate inspection
|       |-- compare_methods.py         Three-method comparison
|       `-- analyze_goal_counts.py     Saved trajectory comparisons by count
|-- tests/
|   |-- test_followup_common.py
|   |-- test_error_insertion.py
|   |-- test_accepted_error_insertion.py
|   |-- test_motion_change.py
|   `-- test_compare_methods.py
`-- results/
    |-- wrist_trajectory.npz           Shared input to both stages
    |-- goals.npz                      First-stage goals and timing
    |-- approximation.npz              First-stage reconstruction
    |-- figures/                       First-stage figures
    `-- followup/
        |-- first_insertion/           Original one-step diagnostic
        |-- error_insertion/           Original iterative history
        |-- accepted_error_insertion/  Accepted states and candidate trials
        |-- motion_change_diagnostic/  Raw change scores and candidate plots
        |-- comparison/                Matched-count metrics and figures
        `-- goal_count_analysis/       Six-count position and velocity views
```

The data and models are local assets. `.gitignore` excludes `ACCAD/`, `models/`, `smplh/`, and the temporary build directory `tmp/`. The scripts use `smplh/female/model.npz`, not the additional local model copies under `models/`.

## Running the Follow-up Experiment

The second stage compares goal-selection methods on the same clip. Unconditional and first-improvement error-driven insertion have both been run from 1 to 102 goals; the first three-method comparison is complete at every goal count from 1 to 102, using raw acceleration for motion-change selection. See [the follow-up notes](docs/followup.md) for the agreed metrics, timing rule, current results, and limitations.

The follow-up uses the included `results/wrist_trajectory.npz`. It does not require the original motion dataset, SMPL+H model, PyTorch, or `human-body-prior` unless you want to repeat extraction or run `src/verify_results.py`.

The observed working versions are Python 3.10.12, NumPy 2.1.3, and Matplotlib 3.10.1. Direct versions are recorded in [requirements-followup.txt](requirements-followup.txt). An isolated environment can be prepared from the repository root with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-followup.txt
```

These pins record the working direct dependencies; a separate clean-environment installation has not been tested and transitive dependencies are not fully locked.

Run from the repository root, in this order (the comparison consumes accepted-state snapshots from the preceding run):

```bash
python3 -m unittest discover -s tests -v
python3 -m src.followup.demo_error_goal
python3 -m src.followup.error_insertion
python3 -m src.followup.accepted_error_insertion
python3 -m src.followup.motion_change
python3 -m src.followup.compare_methods
python3 -m src.followup.analyze_goal_counts
```

See the [matched-count error curves](results/followup/comparison/error_vs_goals.png) and [metrics table](results/followup/comparison/metrics.csv). The new fixed baseline uses evenly spaced samples at each count and shared timing; it differs from the original 0.5 s stage-one baseline.

The [goal-count analysis](results/followup/goal_count_analysis/metrics.csv) reuses saved trajectories at 10, 20, 40, 60, 80, and 102 goals. It compares XYZ position and velocity over the same 17-23 s window with common axis limits. Full-clip metrics and window metrics are labelled separately. This extends analysis of the existing experiment; it does not introduce an automatic count-selection criterion.

Second-stage code is under `src/followup/`, and its outputs are under `results/followup/`. The first-stage commands, files, and report remain at their original paths.

## Reference

H. Sato, H. Mitake, and S. Hasegawa. "Artist-Directable Motion Generation Using Overlapping Minimum Jerk Trajectories for Interactive Embodied Social Agent Adapting to Dynamic Environments." *Computer Animation and Virtual Worlds*, 36, e70075, 2025. [DOI: 10.1002/cav.70075](https://doi.org/10.1002/cav.70075).

Relevant sections: 3.3-3.6 (representation and trajectory generation), 6.2 (switchability and failures), and 7 (limitations).
