"""Manipulation RL Tasks package."""

import gymnasium as gym

from .tasks.ur10_lift_env_cfg import UR10LiftEnvCfg 
from .tasks.ur10_reach_env_cfg import UR10ReachEnvCfg
from .tasks.ur10_reach_infer_env_cfg import UR10ReachInferEnvCfg

# Register the UR10 lift environment
gym.register(
    id="Isaac-Lift-UR10-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UR10LiftEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:UR10LiftPPORunnerCfg",
    },
)

# Register the UR10 lift environment
gym.register(
    id="Isaac-Reach-UR10-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UR10ReachEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:UR10LiftPPORunnerCfg",
    },
)

# Register the UR10 reach inference environment (matches training checkpoint)
gym.register(
    id="Isaac-Reach-UR10-Infer-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UR10ReachInferEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:UR10LiftPPORunnerCfg",
    },
)
