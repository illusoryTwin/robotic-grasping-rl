# Copyright (c) 2025 Ekaterina Mozhegova
#
# SPDX-License-Identifier: MIT

"""Configuration for the two-stage grasping environment.

Stage 1 (RL): Policy predicts 6-DoF grasp pose from observations
Stage 2 (IK + Scripted): Robot executes grasp via differential IK
"""

from __future__ import annotations

from dataclasses import MISSING
from enum import IntEnum

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG

import math
import os
from pathlib import Path
import torch

# Import robot configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from assets import UR10_WITH_GRIPPER_CFG

# Import custom observation functions
from manip_tasks.observations import (
    object_orientation_in_robot_root_frame,
    ee_pose_in_robot_frame,
    object_pose_in_robot_frame,
)


class GraspPhase(IntEnum):
    """Grasp execution phases."""
    OBSERVE = 0      # RL prediction phase - policy outputs grasp pose
    APPROACH = 1     # IK moves EE to pre-grasp position
    GRASP = 2        # IK moves to grasp pose, close gripper
    LIFT = 3         # IK lifts object
    EVALUATE = 4     # Compute reward, end episode


##
# Scene definition (reused from ur10_lift_env_cfg)
##


@configclass
class TwoStageGraspSceneCfg(InteractiveSceneCfg):
    """Configuration for the two-stage grasp scene."""

    # UR10 with Hand-E gripper
    robot = UR10_WITH_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Object to grasp (cylinder)
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5, 0, 0.075],
            rot=[1, 0, 0, 0],
        ),
        spawn=sim_utils.CylinderCfg(
            radius=0.033,
            height=0.08,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    # Frame transformer for end-effector tracking
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/hande_end",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
        ],
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class TwoStageGraspActionsCfg:
    """Action specifications for two-stage grasp.

    The RL policy outputs a 6-DoF grasp pose [x, y, z, roll, pitch, yaw].
    This is converted to joint commands via differential IK in the environment.

    We use joint position actions as a placeholder (6D) - the actual pose-to-joint
    conversion is handled by the custom environment's step() method.
    """
    # Use 6 arm joints as placeholder for 6D pose action
    # The environment's step() will interpret these as [x, y, z, roll, pitch, yaw]
    grasp_pose = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                     "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        scale=1.0,  # Actions will be in [-1, 1], interpreted as normalized pose
        use_default_offset=False,
    )


@configclass
class TwoStageGraspObservationsCfg:
    """Observation specifications for grasp pose prediction."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Proprioceptive observations
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # End-effector pose (7D: pos + quat)
        ee_pose = ObsTerm(func=ee_pose_in_robot_frame)

        # Object pose (7D: pos + quat) - ground truth for now
        object_pose = ObsTerm(func=object_pose_in_robot_frame)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TwoStageGraspEventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.15, 0.15), "y": (-0.2, 0.2), "z": (0.075, 0.075)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class TwoStageGraspRewardsCfg:
    """Sparse reward for grasp success.

    Reward is computed only after the LIFT phase completes.
    """
    # Sparse success reward - computed in custom environment
    # grasp_success = RewTerm(func=grasp_success_sparse, params={"lift_height": 0.1}, weight=10.0)
    pass  # Rewards handled by custom environment


@configclass
class TwoStageGraspTerminationsCfg:
    """Termination conditions."""

    # Time out handled by custom environment based on phases
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Object dropped below table
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


##
# Environment configuration
##


@configclass
class TwoStageGraspEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for two-stage grasping environment."""

    # Scene settings
    scene: TwoStageGraspSceneCfg = TwoStageGraspSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True
    )

    # MDP settings
    observations: TwoStageGraspObservationsCfg = TwoStageGraspObservationsCfg()
    actions: TwoStageGraspActionsCfg = TwoStageGraspActionsCfg()
    rewards: TwoStageGraspRewardsCfg = TwoStageGraspRewardsCfg()
    terminations: TwoStageGraspTerminationsCfg = TwoStageGraspTerminationsCfg()
    events: TwoStageGraspEventCfg = TwoStageGraspEventCfg()

    # Two-stage specific settings
    # IK substeps per RL step (for smooth execution during scripted phases)
    ik_substeps: int = 100

    # Phase timing (in simulation steps)
    approach_timeout: int = 200   # Max steps for approach phase
    grasp_timeout: int = 100      # Max steps for grasp phase (gripper closing)
    lift_timeout: int = 200       # Max steps for lift phase

    # Grasp pose bounds (relative to robot base frame)
    grasp_pos_range: dict = None  # Set in __post_init__
    grasp_rot_range: dict = None  # Set in __post_init__

    # Approach and lift parameters
    approach_offset: float = 0.10  # 10cm offset for pre-grasp
    lift_height: float = 0.15      # 15cm lift height
    success_height: float = 0.10   # Object height threshold for success

    # Transition thresholds
    position_threshold: float = 0.02   # 2cm position accuracy
    velocity_threshold: float = 0.05   # 0.05 m/s velocity threshold

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 2
        self.episode_length_s = 10.0  # Longer episodes for scripted execution

        # Simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Grasp pose bounds (workspace limits)
        self.grasp_pos_range = {
            "x": (0.3, 0.7),   # Forward from robot
            "y": (-0.3, 0.3),  # Left/right
            "z": (0.05, 0.3),  # Above table
        }
        self.grasp_rot_range = {
            "roll": (-math.pi, math.pi),
            "pitch": (-math.pi / 2, math.pi / 2),
            "yaw": (-math.pi, math.pi),
        }
