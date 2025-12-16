"""Debug script to visualize camera output.

This script runs the environment and saves camera images to debug folder.

Usage:
    python scripts/debug_camera_view.py --task Isaac-Lift-UR10-v0
"""

import argparse
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Add the manipulation_rl_new directory to Python path
MANIP_RL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MANIP_RL_DIR))

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Debug camera visualization")
parser.add_argument("--task", type=str, default="Isaac-Lift-UR10-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)

# Parse args
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # Force enable cameras

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching the app
import gymnasium as gym
import torch
import numpy as np

# Import custom tasks
import manip_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

def save_camera_images(env, timestep):
    """Save camera images to debug folder."""
    camera = env.unwrapped.scene["wrist_camera"]

    # Get camera data
    rgb = camera.data.output["rgb"][0].cpu().numpy()  # First environment
    depth = camera.data.output["distance_to_image_plane"][0].cpu().numpy()
    segmentation = camera.data.output["instance_segmentation_fast"][0].cpu().numpy()

    # Handle multi-channel outputs
    if rgb.ndim == 3 and rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    if depth.ndim == 3:
        depth = depth[..., 0]
    if segmentation.ndim == 3:
        segmentation = segmentation[..., 0]

    # Create debug directory
    debug_dir = Path("debug_camera")
    debug_dir.mkdir(exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # RGB
    axes[0].imshow(rgb)
    axes[0].set_title(f"RGB (timestep {timestep})")
    axes[0].axis('off')

    # Depth
    depth_vis = axes[1].imshow(depth, cmap='viridis')
    axes[1].set_title(f"Depth (timestep {timestep})")
    axes[1].axis('off')
    plt.colorbar(depth_vis, ax=axes[1], fraction=0.046)

    # Segmentation
    seg_vis = axes[2].imshow(segmentation, cmap='tab20')
    axes[2].set_title(f"Segmentation (timestep {timestep})")
    axes[2].axis('off')
    plt.colorbar(seg_vis, ax=axes[2], fraction=0.046)

    # Save
    plt.tight_layout()
    save_path = debug_dir / f"camera_view_{timestep:04d}.png"
    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] Saved camera view to: {save_path}")


def main():
    """Main debug function."""

    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    # Create environment
    print("[INFO] Creating environment...")
    env = gym.make(args_cli.task, cfg=env_cfg)

    print("[INFO] Environment created successfully!")
    print(f"[INFO] Camera path: {env.unwrapped.scene['wrist_camera'].cfg.prim_path}")

    # Reset environment
    obs, _ = env.reset()

    print("\n[INFO] Camera is active! Saving images every 50 timesteps...")
    print("[INFO] Images will be saved to: debug_camera/")
    print("[INFO] Press Ctrl+C to stop\n")

    timestep = 0
    try:
        while simulation_app.is_running():
            # Random actions for debugging
            actions = torch.randn(args_cli.num_envs, env.action_space.shape[0], device=env.unwrapped.device)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(actions)

            # Save camera images periodically
            if timestep % 50 == 0:
                save_camera_images(env, timestep)

            timestep += 1

            if timestep >= 500:  # Stop after 500 steps
                print(f"\n[INFO] Captured {timestep} timesteps. Exiting...")
                break

    except KeyboardInterrupt:
        print(f"\n[INFO] Stopped by user after {timestep} timesteps")

    # Close environment
    env.close()

    print(f"\n[INFO] Camera debug complete!")
    print(f"[INFO] Check the 'debug_camera/' folder for saved images")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Debug failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
