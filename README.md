# Minimum Jerk Motion Decomposition

Fit overlapping minimum-jerk displacement components to a recorded right-wrist trajectory. The current method proposes component count and movement intervals from speed structure, fits onset and completion near those intervals, and solves goal displacements by endpoint-constrained linear least squares.

## Report and results

- **[Current report — PDF](report/final_motion_decomposition_report.pdf)** · [Editable Word](report/final_motion_decomposition_report.docx) — final method, essential equations, main experiments, comparisons, joint position/velocity supplement and limitations.
- [Detailed method and results](docs/mainline_results.md)
- [Timing-bound explanation and sensitivity](docs/timing_bound_sensitivity.md)
- [Position + velocity objective comparison](docs/velocity_objective.md) — exploratory extension; the current report and position-only main results are retained.
- [Numerical comparison](results/mainline/metrics.csv) · [Count sweep](results/mainline/count_sweep.csv) · [Fitted goals and timing](results/mainline/goals.csv)

The experiment uses the right wrist from one AMASS ACCAD conversation-gesture clip, analyzing 18–21 s and 30–33 s separately. The nominal speed detector proposes **4 and 3 components**, respectively. At the same component count, both windows improve position RMSE over uniform baselines; velocity RMSE is worse in the first window and better in the second. Counts and timing neighborhoods remain heuristic and parameter-sensitive. The method does not establish true physiological movement units or complete full-body motion conversion.

| Window | Components | Position RMSE | Velocity RMSE |
|---|---:|---:|---:|
| 18–21 s | 4 | 4.194 cm | 0.3620 m/s |
| 30–33 s | 3 | 3.022 cm | 0.2631 m/s |

![Position and velocity errors against uniform baselines at matched counts](results/mainline/errors_vs_count.png)

## Run from the repository root

The included [processed wrist trajectory](data/wrist_trajectory.npz) is sufficient for the fitting experiments. Repeating extraction from the original AMASS clip requires the separately obtained motion data and SMPL+H model; see [data preparation](docs/data_preparation.md).

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python3 -m motion_decomposition.final_pipeline
python3 -m motion_decomposition.complete_mainline
python3 -m motion_decomposition.export_mainline_goals
python3 -m motion_decomposition.timing_bound_sensitivity
python3 -m motion_decomposition.audit_report_evidence
python3 -m unittest discover -s tests -q
```

The first command generates uniform baselines and the free-timing control; `complete_mainline` runs the final speed-interval method and count sweep. Saved control checkpoints, when present locally, are reused. A fresh clone recomputes them. All commands use the repository root; no nested research directory is required.

To reproduce the supplementary joint objective and independently audit its saved results:

```bash
python3 -m motion_decomposition.velocity_objective
python3 scripts/audit_velocity_results.py
```

The tested numerical environment is WSL Ubuntu with Python 3.10.12, NumPy 2.1.3, SciPy 1.15.2 and Matplotlib 3.10.1. There are **58 passing tests**. The evidence audit verifies saved reconstructions, metrics, endpoints, bounds, CSV/NPZ consistency and goals. Report generation has separate document dependencies, described in [report/](report/README.md).

## Repository layout

```text
README.md                 Current research overview and entry points
motion_decomposition/     Current algorithms, numerical utilities and controls
scripts/                  Data extraction and layout verification
tests/                    Numerical regression tests
data/                     Processed wrist trajectory and provenance
results/                  Current main experiment, controls and sensitivity
docs/                     Current method, interpretation and data preparation
report/                   Current editable report and figure assets
archive/                  Earlier experiments and the previous research layout
requirements.txt          Direct numerical dependencies
```

Earlier experiments are consolidated under **[archive/](archive/README.md)**. They are historical evidence rather than the default implementation or results. The original reports and pre-migration numerical records are retained there.

## Detailed records and implementation notes

The report is written for an external reader. Operational instructions, complete numerical inventories and development history are kept in this repository:

| Record | Location |
|---|---|
| Complete method settings, fitting procedure and limitations | [Method notes](docs/mainline_results.md) |
| All count partitions and their errors | [Count sweep CSV](results/mainline/count_sweep.csv) |
| Fitted component timing, displacement and cumulative goals | [Goals CSV](results/mainline/goals.csv) |
| Timing-neighborhood coefficients and full sensitivity records | [Sensitivity notes](docs/timing_bound_sensitivity.md) |
| Solver runs, termination status and source hashes | [Main experiment](results/mainline/summary.json), [sensitivity](results/timing_bound_sensitivity/summary.json) |
| Numerical consistency and regression checks | [Evidence audit](results/timing_bound_sensitivity/audit.json), [tests](tests/) |
| Original user-formatted report before condensation | [Report revision archive](archive/report_revisions/README.md) |

The exploratory sequence moved from uniform sampled goals to error/acceleration heuristics, then to fitted displacement and timing. Free-time fits exposed redundant opposing contributions; the custom opposition penalty and cost-curve count rule also had structural limitations. These findings motivated the current speed-interval association, but they are not combined with the final method's scores. Detailed historical figures, numerical comparisons and diagnostics remain in [archived experiments](archive/README.md).

## References

The representation follows Sato, Mitake and Hasegawa (2025), *Artist-Directable Motion Generation Using Overlapping Minimum Jerk Trajectories for Interactive Embodied Social Agent Adapting to Dynamic Environments*, [DOI](https://doi.org/10.1002/cav.70075). The report and [reference notes](docs/references.md) distinguish published ideas from this project's heuristic settings and historical prototypes.
