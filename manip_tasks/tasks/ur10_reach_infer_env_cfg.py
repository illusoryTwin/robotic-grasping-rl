# Copyright (c) 2025 Ekaterina Mozhegova
#
# SPDX-License-Identifier: MIT
"""
Inference environment configuration for UR10 reach policy.

This config matches the training configuration from logs/rsl_rl/ur10_lift/2025-12-17_18-24-09
to ensure checkpoint compatibility.

Observation space (37 dims):
  - joint_pos: 8 (6 arm + 2 gripper)
  - joint_vel: 8
  - object_position: 3
  - object_orientation: 4
  - target_object_position (command): 7
  - actions: 7

Action space (7 dims):
  - arm_action: 6 joints
  - gripper_action: 1 binary
"""
from __future__ import annotations

from dataclasses import MISSING

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
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.shapes import CuboidCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

import math
import sys
from pathlib import Path

# Import robot configuration
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from assets import UR10_WITH_GRIPPER_CFG

# Import custom observation functions
from manip_tasks.observations import object_orientation_in_robot_root_frame


##
# Scene definition
##

@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Scene configuration matching training."""

    robot = UR10_WITH_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5, 0, 0.055],
            rot=[0.7071, 0.7071, 0, 0],
        ),
        spawn=UsdFileCfg(
            usd_path="/home/ekaterina-mozhegova/Workspace/robotic-grasping-rl/objects/tin-can.usd",
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

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

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings - matching training config exactly
##

@configclass
class CommandsCfg:
    """Command configuration matching training."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="hande_end",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.25, 0.25),
            pos_z=(0.25, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action configuration matching training (7 dims total)."""

    # Arm: 6 joints only (shoulder_*, elbow_joint, wrist_*)
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_.*", "elbow_joint", "wrist_.*"],
        scale=0.2,
        use_default_offset=True,
    )

    # Gripper: 1 binary action
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*finger.*"],
        open_command_expr={".*finger.*": 0.0425},
        close_command_expr={".*finger.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation configuration matching training (37 dims total)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations matching training exactly."""

        # joint_pos: 8 dims (6 arm + 2 gripper)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)

        # joint_vel: 8 dims
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # object_position: 3 dims
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)

        # object_orientation: 4 dims
        object_orientation = ObsTerm(func=object_orientation_in_robot_root_frame)

        # target_object_position (command): 7 dims
        target_object_position = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "object_pose"}
        )

        # actions: 7 dims (last action)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Event configuration matching training."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.6, 0.6),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_.*", "elbow_joint"]),
        },
    )


@configclass
class RewardsCfg:
    """Reward configuration matching training."""

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1},
        weight=1.0
    )

    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04},
        weight=15.0
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-5)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination configuration."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


##
# Environment configuration
##

@configclass
class UR10ReachInferEnvCfg(ManagerBasedRLEnvCfg):
    """Inference environment configuration matching training checkpoint."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=10, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
