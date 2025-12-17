# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import FrameTransformerCfg, CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, VisualizationMarkersCfg  # isort: skip


import math
import os
from pathlib import Path
import numpy as np
import torch

# Import robot configuration from assets
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
    object_rotation_penalty,
    object_translation_penalty,
    asymmetric_finger_contact_penalty,
    centered_grasp_reward,
    grasp_stability_reward,
    finger_object_proximity_reward,
    object_height_reward,
    object_lift_progress_reward,
    both_fingers_contact_reward,
    symmetric_grasp_reward,
    grasp_force_reward,
)

##
# Scene definition
##



MODALITIES = {
    "rgb": 4,
    "distance_to_image_plane": 1,
    "normals": 4,
    "instance_segmentation_fast": 1,
}

# Object paths (defined outside class to avoid being treated as asset config)
OBJECTS_DIR = os.path.join(str(Path.home()), "Workspace/manipulation_rl_new/objects")


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # Set UR10 as robot
    robot = UR10_WITH_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Multi-object configuration: randomly spawns one of three objects per environment
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                # Tetra Pak (height ~0.2m scaled)
                UsdFileCfg(
                    usd_path=os.path.join(OBJECTS_DIR, "tetra-pak-carton.usd"),
                    scale=(0.8, 0.8, 0.8),
                ),
                # Tin Can (height ~0.11m scaled)
                UsdFileCfg(
                    usd_path=os.path.join(OBJECTS_DIR, "tin-can.usd"),
                    scale=(0.8, 0.8, 0.8),
                ),
                # Chips Bag (height ~0.27m scaled)
                UsdFileCfg(
                    usd_path=os.path.join(OBJECTS_DIR, "chips-bag.usd"),
                    scale=(0.8, 0.8, 0.8),
                ),
            ],
            random_choice=False,  # Cycle through objects: env_idx % num_objects
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


    # Frame transformer for end-effector tracking
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/hande_end",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],  # hande_end is already at gripper tip
                ),
            ),
        ],
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
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
        body_name="hande_end",  # End-effector frame (gripper tip)
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
        joint_names=["shoulder_.*", "elbow_joint", "wrist_.*"],
        scale=0.5,
        use_default_offset=True,
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*finger.*"],
        open_command_expr={".*finger.*": 0.0425},  # Open gripper
        close_command_expr={".*finger.*": 0.0},  # Close gripper
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

        # OPTION A: Vision-based (no ground truth) - DISABLED
        # Uses visual_features from camera to locate object
        # More realistic for sim-to-real transfer
        # visual_features = ObsTerm(func=visual_object_features)

        # OPTION B: Ground truth (privileged info) - ENABLED
        # Uses perfect knowledge from simulator
        # Faster to train but won't transfer to real robot
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
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # === Standard rewards (from IsaacLab) ===
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    # # Object heights (scaled 0.8x):
    # #   - Tin Can:    ~0.11m (smallest)
    # #   - Tetra Pak:  ~0.20m
    # #   - Chips Bag:  ~0.27m
    # # Use 0.06m as minimal_height (works for all objects - about half of smallest object height)
    # lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.06}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.06, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.06, "command_name": "object_pose"},
        weight=5.0,
    )

    # # === Custom height-based rewards (similar to MetaIsaacGrasp) ===
    # # Dense reward proportional to object height when gripper is close
    # # Similar to MetaIsaacGrasp's obj_height_reward but without state machine
    # height_reward = RewTerm(
    #     func=object_height_reward,
    #     params={"max_height": 0.5, "proximity_threshold": 0.1},
    #     weight=10.0,
    # )

    # Progress toward target lift height
    # lift_progress = RewTerm(
    #     func=object_lift_progress_reward,
    #     params={"target_height": 0.3},
    #     weight=5.0,
    # )

    # === Finger contact rewards (for stable grasping) ===
    # Reward when BOTH fingers are in contact with object (key for stable grasps!)
    both_fingers_contact = RewTerm(
        func=both_fingers_contact_reward,
        params={"contact_threshold": 0.05},
        weight=5.0,
    )

    # Reward for symmetric finger positioning around object
    symmetric_grasp = RewTerm(
        func=symmetric_grasp_reward,
        params={"contact_threshold": 0.08},
        weight=2.0,
    )

    # Reward for appropriate gripper closure when near object
    # grasp_force = RewTerm(
    #     func=grasp_force_reward,
    #     params={"min_closure": 0.01, "max_closure": 0.04},
    #     weight=1.0,
    # )

    # === Custom grasp quality rewards ===
    # Penalize object rotation during grasp (encourages stable grasps)
    # object_rotation = RewTerm(func=object_rotation_penalty, weight=-0.5)

    # Penalize object XY translation during grasp (encourages centered grasps)
    # object_translation = RewTerm(func=object_translation_penalty, weight=-0.5)

    # Penalize asymmetric finger contacts (encourages symmetric grasps)
    # asymmetric_contact = RewTerm(func=asymmetric_finger_contact_penalty, weight=-0.3)

    # Reward for centering object between fingers
    # centered_grasp = RewTerm(func=centered_grasp_reward, params={"threshold": 0.02}, weight=2.0)

    # Reward for stable grasps (low object velocity)
    # grasp_stability = RewTerm(func=grasp_stability_reward, weight=1.0)

    # Reward for optimal finger-object distance
    # finger_proximity = RewTerm(func=finger_object_proximity_reward, params={"optimal_distance": 0.02}, weight=1.0)

    # === Action penalties ===
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

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

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
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


