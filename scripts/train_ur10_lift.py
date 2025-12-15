"""Training script for UR10 lift task using RSL-RL PPO.

This script trains a UR10 manipulator to lift objects using your custom environment.

Usage:
    # Train with default settings
    python scripts/train_ur10_lift.py --task Isaac-Lift-UR10-v0

    # Train with custom number of environments
    python scripts/train_ur10_lift.py --task Isaac-Lift-UR10-v0 --num_envs 2048

    # Train in headless mode (no GUI)
    python scripts/train_ur10_lift.py --task Isaac-Lift-UR10-v0 --headless --num_envs 4096
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the manipulation_rl_new directory to Python path
MANIP_RL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MANIP_RL_DIR))

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train UR10 lift task with RSL-RL PPO")
parser.add_argument("--task", type=str, default="Isaac-Lift-UR10-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum number of training iterations")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
AppLauncher.add_app_launcher_args(parser)

# Parse args
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching the app
import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Import custom tasks - this registers the environment
import manip_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Set torch settings for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""

    # Parse environment and agent configurations from registry
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    # Get agent configuration class from registry and instantiate it
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
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
        env_cfg.seed = args_cli.seed
    else:
        env_cfg.seed = agent_cfg.seed

    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Logging directory
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=env_cfg.sim.device)

    # Dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Start training
    print(f"[INFO] Starting training for {agent_cfg.max_iterations} iterations")
    print(f"[INFO] Logging to: {log_dir}")

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    print(f"[INFO] Training completed!")
    print(f"[INFO] Logs saved to: {log_dir}")

    # Close environment
    env.close()


if __name__ == "__main__":
    # Run training
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Close simulation app
        simulation_app.close()
