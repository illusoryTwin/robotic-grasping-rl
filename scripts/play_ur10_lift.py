"""Play script for UR10 lift task using trained RSL-RL PPO policy.

This script loads a trained policy and visualizes it in the simulation.

Usage:
    # Play with a trained checkpoint
    python scripts/play_ur10_lift.py --task Isaac-Lift-UR10-v0 --checkpoint /path/to/model.pt

    # Play with custom number of environments
    python scripts/play_ur10_lift.py --task Isaac-Lift-UR10-v0 --checkpoint /path/to/model.pt --num_envs 32
"""

import argparse
import sys
import os
from pathlib import Path

# Add the manipulation_rl_new directory to Python path
MANIP_RL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MANIP_RL_DIR))

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Play UR10 lift task with trained RSL-RL PPO policy")
parser.add_argument("--task", type=str, default="Isaac-Lift-UR10-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pt file)")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
AppLauncher.add_app_launcher_args(parser)

# Parse args
args_cli = parser.parse_args()

# Force enable video rendering for visualization
args_cli.enable_cameras = True

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching the app
import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Import custom tasks - this registers the environment
import manip_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Set torch settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Main play function."""

    # Parse environment configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    # Get agent configuration class from registry
    agent_cfg_entry_point = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]

    # Import the agent config class
    if isinstance(agent_cfg_entry_point, str):
        # Split module path and class name
        module_path, class_name = agent_cfg_entry_point.rsplit(":", 1)
        # Import the module
        import importlib
        module = importlib.import_module(module_path)
        # Get the class and instantiate it
        agent_cfg_class = getattr(module, class_name)
        agent_cfg: RslRlOnPolicyRunnerCfg = agent_cfg_class()
    else:
        agent_cfg: RslRlOnPolicyRunnerCfg = agent_cfg_entry_point

    # Override configurations with CLI arguments
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
        env_cfg.seed = args_cli.seed
    else:
        env_cfg.seed = agent_cfg.seed

    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Check if checkpoint exists
    checkpoint_path = Path(args_cli.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env_cfg.sim.device)

    # Load the trained model
    print(f"[INFO] Loading model checkpoint...")
    runner.load(checkpoint_path)
    print(f"[INFO] Model loaded successfully!")

    # Get policy
    policy = runner.get_inference_policy(device=env_cfg.sim.device)

    # Reset environment
    obs_dict = env.get_observations()
    # Extract policy observations if dict, otherwise use directly
    obs = obs_dict if not isinstance(obs_dict, dict) else obs_dict.get('policy', obs_dict)

    # Simulate
    print(f"[INFO] Starting visualization with {args_cli.num_envs} environments")
    print(f"[INFO] Press Ctrl+C to stop")

    timestep = 0
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                # Get actions from policy
                actions = policy(obs)

                # Step environment
                obs_dict, _, _, _ = env.step(actions)

                # Extract policy observations for next iteration
                obs = obs_dict if not isinstance(obs_dict, dict) else obs_dict.get('policy', obs_dict)

            timestep += 1

            if timestep % 100 == 0:
                print(f"[INFO] Timestep: {timestep}")

    except KeyboardInterrupt:
        print(f"\n[INFO] Visualization stopped by user")

    # Close environment
    env.close()


if __name__ == "__main__":
    # Run play
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Close simulation app
        simulation_app.close()
