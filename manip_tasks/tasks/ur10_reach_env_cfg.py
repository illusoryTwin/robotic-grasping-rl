# Copyright (c) 2025 Ekaterina Mozhegova
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Isaac Lab core imports
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# Isaac Lab tasks MDP
from isaaclab_tasks.manager_based.manipulation.lift import mdp

# Local imports - robot configuration
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from assets import UR10_WITH_GRIPPER_CFG

# Local imports - custom rewards
from manip_tasks.rewards import (
    orientation_command_error,
    position_command_error,
    position_command_error_tanh,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


MODALITIES = {
    "rgb": 4,
    "distance_to_image_plane": 1,
    "normals": 4,
    "instance_segmentation_fast": 1,
}

# Object paths (defined outside class to avoid being treated as asset config)
OBJECTS_DIR = os.path.join(str(Path.home()), "Workspace/Projects/robotic-grasping-rl/objects")



@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # Set UR10 with Hand-E gripper
    robot = UR10_WITH_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Custom tin-can
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),  # 50g
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
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


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="hande_end",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.85),
            pos_y=(-0.3, 0.3),
            pos_z=(0.08, 0.5),
            roll=(-math.pi, math.pi),
            pitch=(-math.pi, math.pi),
            yaw=(-math.pi, math.pi),
            # pos_x=(0.35, 0.65),
            # pos_y=(-0.2, 0.2),
            # pos_z=(0.15, 0.5),
            # roll=(0.0, 0.0),
            # pitch=(-math.pi / 2, math.pi / 2),
            # yaw=(-3.14, 3.14),
        ),
    )



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Set actions for UR10 arm only (6 joints, excluding gripper)
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_.*", "elbow_joint", "wrist_.*"],
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

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:

    # task terms
    end_effector_position_tracking = RewTerm(
        func=position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["hande_end"]), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=position_command_error_tanh,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["hande_end"]), "std": 0.1, "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=orientation_command_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["hande_end"]), "command_name": "ee_pose"},
    )

    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    # joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-1e-4,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )
    # joint_acc = RewTerm(
    #     func=mdp.joint_acc_l2,
    #     weight=-1e-4, #-1e-5
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-4, #-1e-5
        params={"asset_cfg": SceneEntityCfg("robot")},
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    )



@configclass
class UR10ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reaching environment."""

    # Scene settings (replicate_physics=True since using single object type)
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
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




