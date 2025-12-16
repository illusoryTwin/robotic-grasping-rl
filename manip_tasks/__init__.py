"""Manipulation RL Tasks package."""

import gymnasium as gym

from .tasks.ur10_lift_env_cfg import UR10LiftEnvCfg

# Register the UR10 lift environment
gym.register(
    id="Isaac-Lift-UR10-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UR10LiftEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:UR10LiftPPORunnerCfg",
    },
)

# gym.register(
#     id="Isaac-Lift-UR10-v1",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": UR10LiftEnvCfg2,
#         "rsl_rl_cfg_entry_point": f"{__name__}.agents:UR10LiftPPORunnerCfg",
#     },
# )