# Copyright (c) 2025 Ekaterina Mozhegova
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
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
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.sensors import FrameTransformerCfg, CameraCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG, VisualizationMarkersCfg  # isort: skip
from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms


import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv





import math
import os
from pathlib import Path
import numpy as np
import torch

# Import robot configuration from custom assets (UR10 with Hand-E gripper)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from assets import UR10_WITH_GRIPPER_CFG

# Import custom observation functions
from manip_tasks.observations import (
    object_orientation_in_robot_root_frame,
    wrist_camera_rgb,
    wrist_camera_depth,
    visual_object_features,
)

##
# Scene definition
##




def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    # print("object height", object.data.root_pos_w[:, 2])
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    # print("cube_pos_w", cube_pos_w)
    # print("ee_w", ee_w)
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


MODALITIES = {
    "rgb": 4,
    "distance_to_image_plane": 1,
    "normals": 4,
    "instance_segmentation_fast": 1,
}

# Object paths (defined outside class to avoid being treated as asset config)
OBJECTS_DIR = os.path.join(str(Path.home()), "Workspace/Projects/robotic-grasping-rl/objects")


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # Set UR10 with Hand-E gripper
    robot = UR10_WITH_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # # OPTION 1: Built-in DexCube (recommended - has proper collision)
    # object = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    #         scale=(0.8, 0.8, 0.8),
    #         rigid_props=RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=1,
    #             max_angular_velocity=1000.0,
    #             max_linear_velocity=1000.0,
    #             max_depenetration_velocity=5.0,
    #             disable_gravity=False,
    #         ),
    #     ),
    # )

    # OPTION 2: Custom tin-can - using cylinder collision approximation
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5, 0, 0.075],  # Cylinder height/2 + table height
            rot=[1, 0, 0, 0],  # Upright cylinder
        ),
        spawn=sim_utils.CylinderCfg(
            radius=0.033,  # ~33mm radius (typical soup can)
            height=0.08,   # ~80mm height (shorter can)
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)), # (0.5, 0.5, 0.5)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),  # 50g
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    # OPTION 3: YCB tomato soup can (try if DexCube works - may need nucleus server)
    # object = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.1], rot=[1, 0, 0, 0]),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd",
    #         rigid_props=RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=1,
    #             max_angular_velocity=1000.0,
    #             max_linear_velocity=1000.0,
    #             max_depenetration_velocity=5.0,
    #             disable_gravity=False,
    #         ),
    #     ),
    # )

    # # Frame transformer for object tracking
    # object_frame = FrameTransformerCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     debug_vis=True,
    #     visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/ObjectFrameTransformer"),
    #     target_frames=[
    #         FrameTransformerCfg.FrameCfg(
    #             prim_path="{ENV_REGEX_NS}/Object",
    #             name="object",
    #             offset=OffsetCfg(
    #                 pos=[0.0, 0.0, 0.0],
    #             ),
    #         ),
    #     ],
    # )

    # Frame transformer for end-effector tracking (Hand-E gripper)
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/hande_end",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0], # hande_end is at gripper tip
                ),
            ),
        ],
    )

    # # Table
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    # )

   # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Camera disabled for faster training with privileged info
    # Uncomment to enable vision-based training
    # wrist_camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link/wrist_camera",
    #     update_period=0.001,
    #     height=480,
    #     width=640,
    #     data_types=[*MODALITIES],
    #     colorize_instance_id_segmentation=False,
    #     colorize_semantic_segmentation=False,
    #     colorize_instance_segmentation=False,
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0,
    #         focus_distance=400.0,
    #         horizontal_aperture=20.955,
    #         clipping_range=(0.01, 1e5),
    #     ),
    #     offset=CameraCfg.OffsetCfg(
    #         pos=(0.1, 0.1, 0.1),
    #         rot=(0.1, 0.1, 0.1, 0.1),
    #         convention="ros",
    #     ),
    # )



##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="hande_end",  # Hand-E gripper tip
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), 
            pos_y=(-0.25, 0.25), 
            pos_z=(0.25, 0.5), 
            roll=(0.0, 0.0), 
            pitch=(0.0, 0.0), 
            yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Set actions for UR10
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"], # ["shoulder_.*", "elbow_joint", "wrist_.*"],
        scale=0.5,
        use_default_offset=True,
    )
    # Hand-E gripper (parallel jaw gripper)
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*finger.*"],
        open_command_expr={".*finger.*": 0.0425},  # Open gripper
        close_command_expr={".*finger.*": 0.0},    # Close gripper
    )



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Proprioceptive observations
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # # OPTION A: Vision-based (no ground truth)
        # visual_features = ObsTerm(func=visual_object_features)

        # OPTION B: Ground truth (privileged info)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        object_orientation = ObsTerm(func=object_orientation_in_robot_root_frame)

        # Task command
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ImageCfg(ObsGroup):
        """Visual observations from wrist camera."""

        rgb = ObsTerm(func=wrist_camera_rgb)
        depth = ObsTerm(func=wrist_camera_depth)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False  # Keep images separate

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    # image: ImageCfg = ImageCfg()  # Disabled: not using vision-based policy


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Randomize object position on the table (relative to base position [0.5, 0, 0.075])
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

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (-0.6, 0.6),
    #         "velocity_range": (0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_.*", "elbow_joint"]), #, "wrist_.*"]),
    #     },
    # )


@configclass
class RewardsCfg:
    # reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    # lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)

    # object_goal_tracking = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=16.0,
    # )

    # object_goal_tracking_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=5.0,
    # )


    reaching_object = RewTerm(func=object_ee_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=object_is_lifted, params={"minimal_height": 0.05}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.05, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.05, "command_name": "object_pose"},
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class UR10LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings (replicate_physics=True since using single object type)
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


