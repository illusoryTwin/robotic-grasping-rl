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
# from . import mdp  # Using mdp from isaaclab_tasks.manager_based.manipulation.lift instead

##
# Scene definition
##



UR10_WITH_GRIPPER_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path="/home/ekaterina-mozhegova/Workspace/manipulation_rl/assets/ur10e_with_hand_e_and_camera_mount.usd",
        activate_contact_sensors=False,
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_joint", "wrist_.*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[".*finger.*"],
            velocity_limit=0.2,
            effort_limit=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
)



@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # Set UR10 as robot
    robot = UR10_WITH_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(
                str(Path.home()),
                "Workspace/manipulation_rl/objects/tetra-pak-carton.usd"
                # "Workspace/manipulation_rl/objects/tin-can.usd"
            ),
            # scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            # collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            )
        ),
    )

    # # Set Cube as object
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

    # Wrist-mounted camera for visual observations
    # Captures depth and segmentation for visual feature extraction
    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link/wrist_camera",
        update_period=0.1,  # 10 Hz update
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane", "instance_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 1e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.15),  # Offset from wrist_3_link
            rot=(0.5, -0.5, 0.5, -0.5),  # Look forward along gripper axis
            convention="world",
        ),
    )


##
# MDP settings
##

# Camera observation functions (for raw image-based learning - requires CNN)
def wrist_camera_rgb(env) -> torch.Tensor:
    """RGB image from wrist camera."""
    camera_data = env.scene["wrist_camera"].data.output["rgb"]
    # Return as (batch, height, width, channels)
    return camera_data[..., :3].float()


def wrist_camera_depth(env) -> torch.Tensor:
    """Depth image from wrist camera (normalized)."""
    camera_data = env.scene["wrist_camera"].data.output["distance_to_image_plane"]
    # Normalize depth to [0, 1] range, clip at 2 meters
    depth_max = 2.0
    depth_normalized = torch.clamp(camera_data, 0.0, depth_max) / depth_max
    return depth_normalized.unsqueeze(-1)  # Add channel dimension


def visual_object_features(env) -> torch.Tensor:
    """
    Extract engineered visual features from camera data.
    This is more sample-efficient than raw images for RL.

    Similar to MetaIsaacGrasp but used continuously during learning.
    """
    # Get camera data
    camera = env.scene["wrist_camera"]
    depth = camera.data.output["distance_to_image_plane"]
    segmentation = camera.data.output["instance_segmentation_fast"]

    batch_size = depth.shape[0]
    features = torch.zeros(batch_size, 7, device=env.device)

    for i in range(batch_size):
        # Get instance segmentation (take first channel if multi-channel)
        seg = segmentation[i]
        if seg.ndim == 3:
            seg = seg[..., 0]  # Take first channel

        # Find object pixels (ID > 1, where 0=background, 1=robot)
        object_mask = seg > 1

        if object_mask.any():
            # Get object pixels coordinates
            object_pixels = object_mask.nonzero()

            # Compute centroid in image space
            centroid_v = object_pixels[:, 0].float().mean()  # y (height)
            centroid_u = object_pixels[:, 1].float().mean()  # x (width)

            # Get average depth at object (handle depth shape)
            depth_map = depth[i]
            if depth_map.ndim == 3:
                depth_map = depth_map[..., 0]  # Take first channel if multi-channel
            object_depth = depth_map[object_mask].mean()

            # Simple projection to camera frame (approximate)
            # For more accurate: use camera.data.intrinsic_matrices
            cam_height, cam_width = depth.shape[1:3]
            focal_length = 24.0  # From camera config

            # Normalized image coordinates
            u_norm = (centroid_u - cam_width / 2) / focal_length
            v_norm = (centroid_v - cam_height / 2) / focal_length

            # 3D position in camera frame (approximate)
            x_cam = u_norm * object_depth
            y_cam = v_norm * object_depth
            z_cam = object_depth

            # Get end-effector pose to transform to robot frame
            ee_pose = env.scene["ee_frame"].data.target_pos_source[i, 0, :]

            # Compute distance to object (in camera frame)
            distance = torch.sqrt(x_cam**2 + y_cam**2 + z_cam**2)

            # Pack features
            features[i] = torch.tensor([
                x_cam, y_cam, z_cam,           # Object position in camera frame (3)
                distance,                       # Distance to object (1)
                centroid_u / cam_width,        # Normalized image x (1)
                centroid_v / cam_height,       # Normalized image y (1)
                object_depth / 2.0,            # Normalized depth (1)
            ], device=env.device)
        else:
            # No object visible - use default values
            features[i] = torch.zeros(7, device=env.device)

    return features


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="wrist_3_link",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
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
        open_command_expr={".*finger.*": 0.04},  # Open gripper
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

        # Visual observations (replaces ground truth object_position)
        visual_features = ObsTerm(func=visual_object_features)

        # Task command
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        # OPTION A: Vision-based (no ground truth)
        # Uses visual_features from camera to locate object
        # More realistic for sim-to-real transfer

        # OPTION B: Ground truth (uncomment below, comment visual_features above)
        # object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # Uses perfect knowledge from simulator
        # Faster to train but won't transfer to real robot

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

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    # lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)
    # lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.15}, weight=15.0) # tin cad 
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.4}, weight=15.0) # tetra cad

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

    # action penalty
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
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
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


