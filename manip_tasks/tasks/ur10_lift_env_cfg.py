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
from omni.isaac.lab.sim.spawners.shapes import CuboidCfg
from manip_tasks.events import reset_robot_to_vertical_grasp_pose


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

# Import custom reward functions
from manip_tasks.rewards import (
    object_is_lifted,
    object_ee_distance,
    object_goal_distance,
    center_gripper_on_object,
    object_is_lifted_and_grasped,
    finger_object_distance_shaping,
    both_fingers_contact_soft,
    object_height_dense_reward,
    grasp_stability_reward,
)


MODALITIES = {
    "rgb": 4,
    "distance_to_image_plane": 1,
    "normals": 4,
    "instance_segmentation_fast": 1,
}

# Object paths (defined outside class to avoid being treated as asset config)
OBJECTS_DIR = os.path.join(str(Path.home()), "Workspace/Projects/robotic-grasping-rl/objects")


##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # Set UR10 with Hand-E gripper
    robot = UR10_WITH_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Custom Tetra Pak
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5, 0, 0.027],  # Adjust height for tetra pak
            rot=[0.707, 0.707, 0, 0],  # Rotated 90° to lie on long side
        ),
        spawn=CuboidCfg(
            size=(0.06, 0.06, 0.18),  # (width, depth, height) - tetra pak proportions
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=50.0,
                max_linear_velocity=50.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # Light like a carton
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.6, 0.4)),  # Cardboard color
        ),
    )


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
        joint_names=[".*"],
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

    # # reset_object_position = EventTerm(
    # #     func=mdp.reset_root_state_uniform,
    # #     mode="reset",
    # #     params={
    # #         # Randomize object position on the table (relative to base position [0.5, 0, 0.075])
    # #         "pose_range": {"x": (-0.15, 0.15), "y": (-0.2, 0.2), "z": (0.075, 0.075)},
    # #         "velocity_range": {},
    # #         "asset_cfg": SceneEntityCfg("object", body_names="Object"),
    # #     },
    # # )

    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         # Randomize object position on the table (relative to base position for tetra pak)
    #         "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0.1, 0.1)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object", body_names="Object"),
    #     },
    # )

    # Curriculum: Start with vertical grasp pose
    reset_robot_vertical_grasp = EventTerm(
        func=reset_robot_to_vertical_grasp_pose,
        mode="reset",
        params={
            "vertical_height_range": (0.25, 0.35),  # 25-35cm above object
            "horizontal_offset_range": (-0.03, 0.03),  # Small random offset
        },
    )

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.5, 1.5),
    #         "velocity_range": (0.0, 0.0),
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


    # reaching_object = RewTerm(func=object_ee_distance, params={"std": 0.5}, weight=1.0) #params={"std": 0.1}, weight=1.0)

    reaching_object = RewTerm(func=object_ee_distance, params={"std": 0.7}, weight=1.0) #params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )


    both_fingers_contact = RewTerm(
        func=both_fingers_contact_soft,
        params={"std": 0.1},
        weight=4.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01) #-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001, #-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-4, #-1e-5
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


