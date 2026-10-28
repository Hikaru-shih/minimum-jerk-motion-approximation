# Minimum-Jerk Motion Approximation

A preliminary experiment on representing **one wrist trajectory from one motion capture clip** as a sequence of minimum jerk trajectories.

The long-term motivation is to convert motion capture data into an intermediate goal representation that could later support learning-based goal generation. The current task is trajectory approximation and evaluation; no learning model has been implemented.

## 1. First Experiment and Deliverables

Following research discussions with Professor Shoichi Hasegawa, the first experiment will:

1. Choose one clip from an open motion capture dataset.
2. Extract the complete 3D trajectory of one wrist.
3. Approximate it using a sequence of minimum jerk trajectories.
4. Compare original and reconstructed positions, and velocities if possible.
5. Share the code, figures, and a short explanation of goal selection and failure cases.

Whole-body reconstruction, interactive motion switching, inverse kinematics, dynamics simulation, and model training are outside this first experiment.

## 2. Current Progress

Status reviewed on **2026-10-27** from the repository files. Code availability is distinguished from successful execution; runtime verification is not implied by this checklist.

| Item | Status | Evidence / remaining work |
| --- | --- | --- |
| Select dataset | Selected | Local ACCAD motion files in AMASS format are present. |
| Select one clip | Selected | Both scripts reference `Female1Gestures_c3d/D3 - Conversation Gestures_poses.npz`; the file exists locally. |
| Inspect motion metadata | Script written | `src/inspect_motion.py` prints NPZ keys, shapes, frame count, frame rate, and duration. User execution confirmed 6,086 frames at 120 Hz (approximately 50.72 seconds). |
| Load SMPL+H and compute joints | First-frame computation verified | `src/extract_wrist.py` passes the first frame to `human_body_prior.BodyModel` and prints the joint array shape and first five joints. Verified in WSL after a local compatibility adapter for the obsolete `dtype` argument; output shape is `(1, 52, 3)`. |
| Extract a wrist trajectory | Completed | Right wrist (index 21, parent elbow 19); all 6,086 frames saved with timestamps in `results/wrist_trajectory.npz`. All positions are finite. |
| Minimum jerk reconstruction | First run completed | `src/approximate_wrist.py` sums overlapping minimum jerk displacements; all 6,086 samples reconstructed, finite values and endpoint positions verified. |
| Goal selection | First rule implemented | Every 0.5 s plus the last frame; 102 goals excluding the initial position; neighboring intervals overlap by 0.25 s. |
| Position / velocity comparisons | Figures generated | Full position and velocity comparisons plus a 17-23 s position detail view are saved. |
| Failure analysis | Initial observations recorded | Brief peaks and rapid changes are missed; see the first-run observations below. |
| Reproducible setup | Incomplete | Ubuntu WSL default Python is 3.10.12; the project environment, package versions, and dependency manifest still need verification. |

**Next milestone:** review the first comparison figures and the goal-selection explanation before preparing the material for the professor.

## 3. Data and Existing Code

Selected clip:

```text
ACCAD/Female1Gestures_c3d/D3 - Conversation Gestures_poses.npz
```

Current extraction prototype uses:

```text
smplh/female/model.npz
```

`extract_wrist.py` verifies female clip metadata, uses the female SMPL+H model, and processes all frames in batches of 32. The right wrist is joint 21 (zero-based), consistent with the body joint names; the script also checks that its parent is right elbow 19 in the actual model topology.

The saved positions are global coordinates including root orientation and translation, in meters using the AMASS/SMPL model scale. No smoothing, resampling, or minimum jerk approximation is applied. The output contains positions, timestamps, frame rate, joint name/index, coordinate frame, units, gender, source clip, and model path.

Verified output: `(6086, 3)` at 120 Hz, sample timestamps from 0 to 50.708333 seconds. The frame-count duration is 50.716667 seconds; the final sample time is one sample interval shorter. All positions are finite.

Original trajectory figures:

- [3D path](results/figures/original_trajectory_3d.png)
- [XYZ position over time](results/figures/original_position.png)

### Current WSL commands

Activate the existing Python environment containing the required packages, then run:

```bash
cd /mnt/d/minimum-jerk-motion-approximation/src
python inspect_motion.py
python extract_wrist.py
python plot_wrist.py
python approximate_wrist.py
```

Use `python3` instead if that is the interpreter configured in the active environment. All experiment scripts resolve data and output paths relative to their file location. They can also run from the repository root using `python3 src/<script_name>.py`.

Ubuntu WSL is available, and its default `python3 --version` reports Python 3.10.12. The first-frame script was subsequently verified in WSL with `human-body-prior==0.8.5.0` and `smplx==0.1.28`. A scoped adapter in `extract_wrist.py` handles their `lbs` signature mismatch without editing installed packages.

Direct imports are `numpy`, `torch`, `human_body_prior`, and `matplotlib`. Their compatible versions and transitive dependencies have not yet been documented; this is not a complete installation guide. Record the working environment before adding a tested `requirements.txt` or environment file.

## 4. Minimum Jerk Representation

The reference paper uses the standard minimum jerk function together with **temporally overlapping trajectory displacements** (Sections 3.3-3.6, Equations 3-6 and Figure 4).

For an instance with activation time `a`, completion time `b > a`, and normalized time `u = (t - a) / (b - a)`:

```text
s(t; a, b) = 0                        if t < a
            10u^3 - 15u^4 + 6u^5     if a <= t <= b
            1                        if t > b
```

A single trajectory from `p_start` to `g` is:

```text
p(t) = p_start + (g - p_start) * s(t; a, b)
```

Its endpoint velocity and acceleration are zero. Sequentially concatenating such segments therefore makes the reconstructed wrist stop at each segment boundary. This differs from the paper's overlapping method.

### Planned overlapping reconstruction

For the position-only experiment, keep the initial position `g_0 = p_initial` and represent each subsequent instance by:

```text
goal_position g_i, activation_time a_i, completion_time b_i
```

The paper sets each new instance's starting pose to the preceding instance's goal pose. For fixed position goals, the corresponding displacement sum is:

```text
p_hat(t) = p_initial + sum_i [(g_i - g_(i-1)) * s(t; a_i, b_i)]
```

A later instance can activate before an earlier one completes. Individual instances start and end at zero velocity, while the combined trajectory can continue moving through those times. During overlap, the output need not pass exactly through each goal at that instance's completion time.

The polynomial above is already consistent with the paper; the missing implementation is the timing and displacement composition. The first experiment will use fixed global position goals, without the paper's scene-dependent goal coordinate systems or orientation handling.

### Goal selection used in the first run

The initial rule samples the original wrist position every 0.5 seconds (60 frames at 120 Hz), includes frame 0 as the initial position, and always includes the last frame. This produces 102 subsequent goals. This interval is a simple experimental starting choice, not an optimized value or a setting prescribed by the professor.

For consecutive source sample times `t_(i-1)` and `t_i`, use:

```text
a_i = max(first_timestamp, t_(i-1) - 0.125)
b_i = min(last_timestamp, t_i + 0.125)
```

This expands each interval on both sides to make neighboring trajectories overlap by 0.25 seconds. Clip the beginning and end to the data timestamps; the final source interval is shorter than 0.5 seconds. Save the source frame indices, goal positions, activation times, completion times, and chosen settings in `results/goals.npz`. Source sample times identify where goals were taken from; they are not the trajectories' completion times.

## 5. Comparison and Explanation

The professor requested figures comparing the original and approximated wrist positions, and velocities if possible. Compare the trajectories at matching timestamps and state the coordinate frame and units.

Provide a short explanation of:

- How the goals and their timing were decided.
- Which parts of the approximation did not work well.

The first run provides full XYZ position and velocity figures and a 17-23 s position detail view. Additional experiments, multiple goal-count comparisons, adaptive selection, and numerical metric suites are not part of the current task.

### First-run observations

- In the 17-23 s detail, slow changes and plateaus are followed more closely than brief peaks. For example, the Z peak near 17.3 s is missed because the sampled goals do not capture its maximum. The X oscillations around 21.5-22 s are also missed.
- Some transitions start earlier or finish later than in the source because timing is fixed by sampling rather than fitted to the movement.
- Velocity peaks are often reduced or displaced. Both velocities use `numpy.gradient` with the same timestamps, centered interior differences and second-order one-sided endpoint differences (`edge_order=2`), without filtering. Differentiation also emphasizes small fluctuations in the source; the plots do not establish whether those fluctuations are real motion or fitting noise.
- These are limitations of this selected goal/timing rule. They do not establish a general limitation of every possible minimum jerk approximation.

Figures: [positions](results/figures/position_comparison.png), [position detail](results/figures/position_comparison_detail.png), [velocities](results/figures/velocity_comparison.png).

### Implementation verification

Run from the repository root in the working WSL environment:

```bash
python3 src/verify_results.py
```

The following checks passed on 2026-10-27 without changing experiment outputs:

- Minimum jerk progress before/start/midpoint/end/after a 2-4 s instance matches `[0, 0, 0.5, 1, 1]` within absolute tolerance `1e-12`.
- For a unit-duration instance, one-sided finite differences (`h=0.001`) give endpoint velocities about `-1.99e-5` and accelerations about `+/-3.29e-4`, within tolerances `3e-5` and `1e-3`. These are numerical estimates of the polynomial's zero endpoint derivatives, not wrist motion measurements.
- Five XYZ positions in a two-instance overlapping example match literal hand-calculated values within `1e-12`.
- Frames 0, 31, 32, 3000, and 6085 were recomputed individually and in a batch and compared with the saved trajectory. The maximum absolute coordinate discrepancy was `5.96e-8 m`, below the `1e-6 m` tolerance for float32 model arithmetic.
- The full saved trajectory has the expected shape, finite positions, and timestamps equal to frame indices divided by the frame rate.

These checks support the tested numerical operations and extraction consistency. They do not independently establish anatomical ground truth, validate every input or motion, or establish that the chosen goal rule produces an accurate approximation.

## 6. Repository Structure

Existing relevant files and local assets:

```text
minimum-jerk-motion-approximation/
|-- README.md
|-- .gitignore
|-- src/
|   |-- inspect_motion.py
|   |-- extract_wrist.py
|   |-- plot_wrist.py
|   |-- approximate_wrist.py
|   `-- verify_results.py
|-- ACCAD/                  # local motion dataset; ignored by Git
|-- smplh/                  # local model used by current code; ignored by Git
`-- models/smplh/           # additional local models; ignored by Git
```

The two model locations have different filenames; the scripts currently use `smplh/`, not `models/smplh/`.

The motion data and body models must be obtained separately under their respective licenses. `.gitignore` excludes `ACCAD/`, `models/`, `smplh/`, and the temporary build directory `tmp/`.

Current experiment outputs:

```text
results/
|-- wrist_trajectory.npz       # positions, timestamps, and extraction metadata
|-- goals.npz                  # goal positions and activation/completion times
|-- approximation.npz          # reconstructed positions and both velocities
`-- figures/
    |-- original_trajectory_3d.png
    |-- original_position.png
    |-- position_comparison_detail.png
    |-- position_comparison.png
    `-- velocity_comparison.png # if possible
```

## 7. Next Steps

1. Verify the selected clip and first-frame body-model call in WSL; record the working environment and metadata.
2. Completed: verify the wrist joint index, extract all frames, save the trajectory and timestamps, and plot the original motion.
3. Completed first run: implement the minimum jerk function and overlapping displacement sum.
4. Completed first run: save the fixed sampling and overlap rule and reconstruct the trajectory.
5. Completed first run: produce position and velocity comparison figures.
6. Explain successful and unsuccessful regions, update execution instructions, and prepare the code and figures for review.

The current work ends with the small-scale deliverables requested by the professor. Further research will be discussed after reviewing these results.

## 8. Reference

H. Sato, H. Mitake, and S. Hasegawa,
“Artist-Directable Motion Generation Using Overlapping Minimum Jerk Trajectories for Interactive Embodied Social Agent Adapting to Dynamic Environments,”
*Computer Animation and Virtual Worlds*, 36, e70075, 2025.
DOI: [10.1002/cav.70075](https://doi.org/10.1002/cav.70075)

Relevant sections: 3.3-3.6 (representation and trajectory generation), 6.2 (switchability and failures), and 7 (limitations).
