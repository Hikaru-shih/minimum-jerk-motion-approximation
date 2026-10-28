"""Extract the complete right-wrist trajectory from the selected AMASS clip."""

import inspect
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from human_body_prior.body_model import body_model as body_model_module
from human_body_prior.body_model.body_model import BodyModel

ROOT = Path(__file__).resolve().parents[1]
MOTION = ROOT / 'ACCAD/Female1Gestures_c3d/D3 - Conversation Gestures_poses.npz'
MODEL = ROOT / 'smplh/female/model.npz'
OUTPUT = ROOT / 'results/wrist_trajectory.npz'
# SMPL+H body joint order: elbow 19 -> right wrist 21 (zero based).
WRIST_INDEX = 21
BATCH_SIZE = 32
original_lbs = body_model_module.lbs


def compatible_lbs(*args, dtype=None, **kwargs):
    # human_body_prior 0.8.5.0 passes dtype; smplx 0.1.28 infers it.
    if 'dtype' in inspect.signature(original_lbs).parameters:
        return original_lbs(*args, dtype=dtype, **kwargs)
    if dtype is not None and kwargs['betas'].dtype != dtype:
        raise ValueError('Body model dtype does not match betas dtype')

    return original_lbs(*args, **kwargs)


def load_motion():
    """Read the clip and check its shape, frame rate, and gender."""

    with np.load(MOTION) as data:
        poses = data['poses'].copy()
        trans = data['trans'].copy()
        betas = data['betas'].copy()
        fps = float(data['mocap_framerate'])
        gender = str(data['gender'].item())
    if gender.lower() != 'female':
        raise ValueError(f'Expected female model metadata, got {gender!r}')
    if poses.shape != (len(trans), 156) or trans.shape[1:] != (3,):
        raise ValueError('Unexpected motion array shapes')
    if len(poses) == 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Empty clip or invalid frame rate')

    return poses, trans, betas, fps, gender


def extract_positions(poses, trans, betas):
    """Compute global right-wrist positions in small batches."""

    model = BodyModel(bm_path=str(MODEL), model_type='smplh', num_betas=16)


    if int(model.kintree_table[0, WRIST_INDEX]) != 19:
        raise ValueError('Unexpected right-wrist parent in model topology')


    positions = np.empty((len(poses), 3), dtype=np.float32)
    shape = torch.as_tensor(betas[:16], dtype=torch.float32)
    with torch.no_grad(), patch.object(body_model_module, 'lbs', compatible_lbs):
        for start in range(0, len(poses), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(poses))
            pose = torch.as_tensor(poses[start:end], dtype=torch.float32)
            body = model(
                root_orient=pose[:, :3],
                pose_body=pose[:, 3:66],
                pose_hand=pose[:, 66:156],
                betas=shape.unsqueeze(0).expand(end - start, -1),
                trans=torch.as_tensor(trans[start:end], dtype=torch.float32),
            )
            positions[start:end] = body.Jtr[:, WRIST_INDEX].cpu().numpy()
            if start == 0 or end == len(poses) or (start // BATCH_SIZE) % 25 == 0:
                print(f'Frames: {end}/{len(poses)}', flush=True)

    return positions


def save_trajectory(positions, fps, gender):
    """Save the trajectory together with timing and source metadata."""

    if not np.isfinite(positions).all():
        raise ValueError('Non-finite wrist positions')
    timestamps = np.arange(len(positions), dtype=np.float64) / fps
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        positions=positions,
        timestamps=timestamps,
        frame_rate=fps,
        joint_name='right_wrist',
        joint_index=WRIST_INDEX,
        coordinate_frame='global (root orientation and translation included)',
        units='m',
        gender=gender,
        source_clip=str(MOTION.relative_to(ROOT)),
        model_path=str(MODEL.relative_to(ROOT)),
    )

    return timestamps


def main():
    poses, trans, betas, fps, gender = load_motion()
    positions = extract_positions(poses, trans, betas)
    timestamps = save_trajectory(positions, fps, gender)


    print(f'Saved: {OUTPUT}')
    print(f'Shape: {positions.shape}; rate: {fps} Hz')
    print(f'Sample times: {timestamps[0]:.6f} to {timestamps[-1]:.6f} s')
    print(f'XYZ min: {positions.min(axis=0)}')
    print(f'XYZ max: {positions.max(axis=0)}')


if __name__ == '__main__':
    main()
