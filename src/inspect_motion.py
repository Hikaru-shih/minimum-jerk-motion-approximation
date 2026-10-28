"""Inspect the selected motion clip before extracting its wrist trajectory."""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOTION_PATH = (
    ROOT / "ACCAD/Female1Gestures_c3d/D3 - Conversation Gestures_poses.npz"
)


def main():
    """Print the available arrays and basic timing information."""
    with np.load(MOTION_PATH) as data:
        print("Keys in the NPZ file:")
        print(data.files)

        for key in data.files:
            print(key, data[key].shape)

        frame_count = data["poses"].shape[0]
        frame_rate = float(data["mocap_framerate"])

        print("\nMotion info:")
        print("Frames:", frame_count)
        print("Frame rate:", frame_rate)
        print("Duration:", frame_count / frame_rate, "seconds")


if __name__ == "__main__":
    main()
