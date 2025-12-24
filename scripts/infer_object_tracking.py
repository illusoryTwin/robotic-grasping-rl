#!/usr/bin/env python3
"""Inference script for object tracking: Move end-effector 7cm above object.

This script loads a trained policy and commands the robot to place its end-effector
exactly 7cm above the object position, matching the object's pose.

The trained policy was trained to track arbitrary EE poses. Here we use it to track
a specific pose: object_position + [0, 0, 0.07].

Usage:
    # Use latest checkpoint
    python scripts/infer_object_tracking.py

    # Use specific checkpoint
    python scripts/infer_object_tracking.py --checkpoint /path/to/model.pt

    # Multiple environments
    python scripts/infer_object_tracking.py --num_envs 4
"""

import argparse
import sys
from pathlib import Path

# Add the robotic-grasping-rl directory to Python path
MANIP_RL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MANIP_RL_DIR))

# Default logs directory
LOGS_DIR = Path.cwd() / "logs" / "rsl_rl" / "ur10_lift"


def find_latest_checkpoint(logs_dir: Path = LOGS_DIR) -> tuple[Path, Path]:
    """Find the latest checkpoint and its run directory from the most recent training run.

    Returns:
        Tuple of (checkpoint_path, run_dir)
    """
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    run_dirs = sorted(logs_dir.glob("*/"), key=lambda x: x.name, reverse=True)
    if not run_dirs:
        raise FileNotFoundError(f"No training runs found in: {logs_dir}")

    for run_dir in run_dirs:
        model_files = list(run_dir.glob("model_*.pt"))
        if model_files:
            def get_iteration(f):
                try:
                    return int(f.stem.split("_")[1])
                except (IndexError, ValueError):
                    return -1

            model_files.sort(key=get_iteration, reverse=True)
            latest_model = model_files[0]

            print(f"[INFO] Auto-detected latest checkpoint:")
            print(f"       Run: {run_dir.name}")
            print(f"       Model: {latest_model.name}")

            return latest_model, run_dir

    raise FileNotFoundError(f"No model checkpoints (model_*.pt) found in: {logs_dir}")


from omni.isaac.lab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Inference: Move EE 7cm above object")
parser.add_argument("--task", type=str, default="Isaac-Reach-UR10-Infer-v0", help="Task name")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--z_offset", type=float, default=0.07, help="Z offset above object (meters)")
AppLauncher.add_app_launcher_args(parser)

# Parse args
args_cli = parser.parse_args()

# Enable visualization
args_cli.enable_cameras = True

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching app
import gymnasium as gym
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Import custom tasks
import manip_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

# Torch settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Main inference loop."""

    # Parse environment configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    # Get agent configuration
    agent_cfg_entry_point = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]

    if isinstance(agent_cfg_entry_point, str):
        module_path, class_name = agent_cfg_entry_point.rsplit(":", 1)
        import importlib
        module = importlib.import_module(module_path)
        agent_cfg_class = getattr(module, class_name)
        agent_cfg: RslRlOnPolicyRunnerCfg = agent_cfg_class()
    else:
        agent_cfg: RslRlOnPolicyRunnerCfg = agent_cfg_entry_point

    # Override configurations
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
        env_cfg.seed = args_cli.seed
    else:
        env_cfg.seed = agent_cfg.seed

    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Find checkpoint
    if args_cli.checkpoint is not None:
        checkpoint_path = Path(args_cli.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        checkpoint_path, _ = find_latest_checkpoint()

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Create runner and load policy
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env_cfg.sim.device)

    print(f"[INFO] Loading model checkpoint...")
    runner.load(checkpoint_path)
    print(f"[INFO] Model loaded successfully!")

    # Get policy for inference
    policy = runner.get_inference_policy(device=env_cfg.sim.device)

    # Reset environment
    obs, _ = env.get_observations()

    # Access the underlying Isaac Lab environment
    isaac_env = env.unwrapped

    print("=" * 80)
    print(f"[INFO] Starting object tracking inference")
    print(f"[INFO] Number of environments: {args_cli.num_envs}")
    print(f"[INFO] Z offset above object: {args_cli.z_offset} m ({args_cli.z_offset * 100:.1f} cm)")
    print(f"[INFO] The robot will move its end-effector to match object pose + Z offset")
    print("=" * 80)
    print("[INFO] Press Ctrl+C to stop")
    print()

    timestep = 0
    episode_lengths = torch.zeros(args_cli.num_envs, device=env_cfg.sim.device)

    try:
        while simulation_app.is_running():
            # Get object position from scene
            object_asset = isaac_env.scene["object"]
            object_pos = object_asset.data.root_pos_w.clone()  # (num_envs, 3)
            object_quat = object_asset.data.root_quat_w.clone()  # (num_envs, 4)

            # Compute target pose: object position + Z offset
            target_pos = object_pos.clone()
            target_pos[:, 2] += args_cli.z_offset  # Add Z offset (7cm default)

            # Target orientation: same as object (or you can set a fixed orientation)
            target_quat = object_quat.clone()

            # IMPORTANT: Override the command with our custom target pose
            # The command format is: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
            command = torch.cat([target_pos, target_quat], dim=-1)  # (num_envs, 7)

            # Get the command manager and manually set the command
            command_manager = isaac_env.command_manager
            if hasattr(command_manager, '_terms') and 'ee_pose' in command_manager._terms:
                command_manager._terms['ee_pose'].command = command

            # Get observations (which now include our custom command)
            obs, _ = env.get_observations()

            with torch.inference_mode():
                # Get actions from policy
                actions = policy(obs)

                # Step environment
                obs, rewards, dones, extras = env.step(actions)

            # Track episode lengths
            episode_lengths += 1

            # Print periodic updates
            if timestep % 100 == 0:
                # Get end-effector position
                robot = isaac_env.scene["robot"]
                ee_frame = isaac_env.scene["ee_frame"]
                ee_pos = ee_frame.data.target_pos_w[:, 0, :]  # (num_envs, 3)

                # Compute tracking error
                position_error = torch.norm(ee_pos - target_pos, dim=-1)  # (num_envs,)

                avg_error = position_error.mean().item()
                max_error = position_error.max().item()
                min_error = position_error.min().item()

                print(f"[Step {timestep:5d}] "
                      f"Error (cm): avg={avg_error*100:.2f}, "
                      f"min={min_error*100:.2f}, "
                      f"max={max_error*100:.2f} | "
                      f"Reward: {rewards.mean().item():.3f}")

                # Print detailed info for first environment
                if timestep % 500 == 0:
                    print(f"\n  [Env 0] Object pos: [{object_pos[0,0]:.3f}, {object_pos[0,1]:.3f}, {object_pos[0,2]:.3f}]")
                    print(f"  [Env 0] Target pos: [{target_pos[0,0]:.3f}, {target_pos[0,1]:.3f}, {target_pos[0,2]:.3f}]")
                    print(f"  [Env 0] EE pos:     [{ee_pos[0,0]:.3f}, {ee_pos[0,1]:.3f}, {ee_pos[0,2]:.3f}]")
                    print(f"  [Env 0] Position error: {position_error[0].item()*100:.2f} cm\n")

            # Handle episode resets
            if dones.any():
                reset_envs = torch.where(dones)[0]
                print(f"\n[INFO] Episodes finished in envs: {reset_envs.tolist()}")
                print(f"       Episode lengths: {episode_lengths[reset_envs].tolist()}")
                episode_lengths[reset_envs] = 0
                print()

            timestep += 1

    except KeyboardInterrupt:
        print(f"\n[INFO] Inference stopped by user")
        print(f"[INFO] Total timesteps: {timestep}")

    # Cleanup
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
