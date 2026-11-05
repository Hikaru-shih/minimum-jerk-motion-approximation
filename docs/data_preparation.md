# Data preparation

The current fitting experiments use `data/wrist_trajectory.npz`. It is the same processed input used before the layout change; its bytes are checked by `scripts/verify_layout_migration.py`.

| Item | Value |
|---|---|
| Dataset | ACCAD subset of AMASS |
| Clip | Female1Gestures_c3d/D3 - Conversation Gestures_poses.npz |
| Samples | 6086 at 120 Hz |
| Joint | Right wrist, index 21, parent 19 |
| Model | Female SMPL+H, 16 shape coefficients |
| Coordinates | Global XYZ including root orientation and translation |
| Units | Position m, timestamp s |
| Timestamp range | 0 to 50.708333 s |

The analysis windows 18–21 s and 30–33 s each include 361 samples. Detection filtering uses the full processed clip before cropping; fitting and evaluation use unfiltered positions.

## Optional extraction

Extraction is not required to reproduce fitting from the included processed trajectory. To repeat it, obtain the motion data and body model under their respective access conditions, placing them locally at:

```text
ACCAD/Female1Gestures_c3d/D3 - Conversation Gestures_poses.npz
smplh/female/model.npz
```

These original assets are ignored by Git. The extraction environment additionally needs PyTorch, human-body-prior and smplx. The observed working package combination was human-body-prior 0.8.5.0 with smplx 0.1.28; the numerical fitting requirements file does not install these optional extraction packages.

From the repository root:

```bash
python3 scripts/inspect_motion.py
python3 scripts/extract_wrist.py
```

Extraction processes every frame in batches of 32 and writes `data/wrist_trajectory.npz`; it does not crop, smooth or resample the positions. A local compatibility adapter handles the older body-model caller's `dtype` argument without editing installed packages. The processed file records source clip, model path, joint, units and coordinate metadata.

Original AMASS and body-model access: [AMASS](https://amass.is.tue.mpg.de/). The research evaluates kinematic reconstruction and requires no physics engine.
