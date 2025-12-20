"""Play script for UR10 lift task using trained RSL-RL PPO policy.

This script loads a trained policy and visualizes it in the simulation.

Usage:
    # Play with the latest checkpoint (auto-detected)
    python scripts/play_ur10_lift.py --task Isaac-Lift-UR10-v0

    # Play with a specific checkpoint
    python scripts/play_ur10_lift.py --task Isaac-Lift-UR10-v0 --checkpoint /path/to/model.pt

    # Play with custom number of environments
    python scripts/play_ur10_lift.py --task Isaac-Lift-UR10-v0 --num_envs 32
"""

import argparse
import sys
import os
import glob
from pathlib import Path

# Add the robotic-grasping-rl directory to Python path
MANIP_RL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MANIP_RL_DIR))

# Default logs directory (relative to current working directory, where isaaclab.sh runs from)
LOGS_DIR = Path.cwd() / "logs" / "rsl_rl" / "ur10_lift"


def find_latest_checkpoint(logs_dir: Path = LOGS_DIR) -> Path:
    """Find the latest checkpoint from the most recent training run.
    Returns:
        Path to the latest checkpoint file.
    """
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")
    run_dirs = sorted(logs_dir.glob("*/"), key=lambda x: x.name, reverse=True)
    if not run_dirs:
        raise FileNotFoundError(f"No training runs found in: {logs_dir}")
    
    # Search through runs starting from the most recent
    for run_dir in run_dirs:
        # Find all model checkpoints in this run
        model_files = list(run_dir.glob("model_*.pt"))
        
        if model_files:
            # Sort by iteration number (model_0.pt, model_50.pt, model_100.pt, ...)
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
            
            return latest_model
    
    raise FileNotFoundError(f"No model checkpoints (model_*.pt) found in any run under: {logs_dir}")


from omni.isaac.lab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Play UR10 lift task with trained RSL-RL PPO policy")
parser.add_argument("--task", type=str, default="Isaac-Lift-UR10-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt). If not provided, uses latest.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--run", type=str, default=None, help="Specific run directory name (e.g., '2025-12-16_22-35-35')")
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

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Import custom tasks - this registers the environment
import manip_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

# Set torch settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
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

    # Find checkpoint path
    if args_cli.checkpoint is not None:
        # Use provided checkpoint
        checkpoint_path = Path(args_cli.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    elif args_cli.run is not None:
        # Use specific run directory
        run_dir = LOGS_DIR / args_cli.run
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        checkpoint_path = find_latest_checkpoint(run_dir.parent)
        # Filter to only this run
        model_files = sorted(run_dir.glob("model_*.pt"), 
                           key=lambda f: int(f.stem.split("_")[1]) if f.stem.split("_")[1].isdigit() else -1,
                           reverse=True)
        if not model_files:
            raise FileNotFoundError(f"No checkpoints in run: {run_dir}")
        checkpoint_path = model_files[0]
        print(f"[INFO] Using run: {args_cli.run}")
    else:
        # Auto-detect latest checkpoint
        checkpoint_path = find_latest_checkpoint()

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
    obs, _ = env.get_observations()

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
                obs, _, _, _ = env.step(actions)

            timestep += 1

            if timestep % 100 == 0:
                print(f"[INFO] Timestep: {timestep}")

    except KeyboardInterrupt:
        print(f"\n[INFO] Visualization stopped by user")

    env.close()


if __name__ == "__main__":
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
