"""Manipulation RL Tasks package."""

import gymnasium as gym

from .tasks.ur10_lift_env_cfg import UR10LiftEnvCfg
from .tasks.two_stage_grasp_env_cfg import TwoStageGraspEnvCfg

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

# Register the two-stage grasp environment
gym.register(
    id="Isaac-TwoStageGrasp-UR10-v0",
    entry_point="manip_tasks.tasks.two_stage_grasp_env:TwoStageGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TwoStageGraspEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:TwoStageGraspPPORunnerCfg",
    },
)
