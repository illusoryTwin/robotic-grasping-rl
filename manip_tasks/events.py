# Copyright (c) 2025 Ekaterina Mozhegova
#
# SPDX-License-Identifier: MIT

"""Custom event functions for curriculum learning."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import sample_uniform

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def reset_robot_to_vertical_grasp_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    vertical_height_range: tuple[float, float] = (0.3, 0.4),
    horizontal_offset_range: tuple[float, float] = (-0.05, 0.05),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Reset robot to vertical grasp pose above the object (curriculum learning).
    
    This function:
    1. Gets the current object position
    2. Computes desired end-effector pose (vertical above object)
    3. Uses inverse kinematics to find joint positions
    4. Resets robot joints to achieve vertical grasp pose
    
    Args:
        env: The environment.
        env_ids: Environment indices to reset.
        vertical_height_range: Height range above object for gripper.
        horizontal_offset_range: Small random offset in XY plane.
        robot_cfg: Robot scene entity config.
        object_cfg: Object scene entity config.
    """
    if len(env_ids) == 0:
        return
        
    robot: Articulation = env.scene[robot_cfg.name]
    object = env.scene[object_cfg.name]
    
    # Get object positions for specified envs
    object_pos = object.data.root_pos_w[env_ids]
    
    # Sample random heights and offsets
    num_envs = len(env_ids)
    heights = sample_uniform(
        vertical_height_range[0], vertical_height_range[1], 
        (num_envs, 1), device=env.device
    )
    x_offsets = sample_uniform(
        horizontal_offset_range[0], horizontal_offset_range[1], 
        (num_envs, 1), device=env.device
    )
    y_offsets = sample_uniform(
        horizontal_offset_range[0], horizontal_offset_range[1], 
        (num_envs, 1), device=env.device
    )
    
    # Compute target end-effector positions (above object)
    target_ee_pos = object_pos.clone()
    target_ee_pos[:, 0] += x_offsets.squeeze()  # X offset
    target_ee_pos[:, 1] += y_offsets.squeeze()  # Y offset  
    target_ee_pos[:, 2] += heights.squeeze()     # Z height above object
    
    # Set orientation for vertical grasp (gripper pointing down)
    # Quaternion for 180° rotation around X-axis: [0, 1, 0, 0]
    target_ee_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=env.device)
    target_ee_quat = target_ee_quat.repeat(num_envs, 1)
    
    # Use IK to compute joint positions
    # This is a simplified approach - for UR10, we set some reasonable joint angles
    # that typically result in a vertical end-effector pose
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    
    # For UR10 with vertical grasp - trying alternative configuration
    # Experimenting to find the right orientation for vertical gripper
    joint_pos[env_ids, 0] = 0.0      # shoulder_pan: facing forward
    joint_pos[env_ids, 1] = -1.57    # shoulder_lift: -90 degrees
    joint_pos[env_ids, 2] = 1.57     # elbow: 90 degrees
    joint_pos[env_ids, 3] = 0.0      # wrist_1: 0 degrees
    joint_pos[env_ids, 4] = -1.57    # wrist_2: -90 degrees (try this for vertical)
    joint_pos[env_ids, 5] = 0.0      # wrist_3: 0 degrees
    
    # Add small random variations to avoid perfect symmetry
    joint_noise = sample_uniform(-0.1, 0.1, (num_envs, 6), device=env.device)
    joint_pos[env_ids, :6] += joint_noise
    
    # Open gripper initially (Hand-E finger joints at indices 6, 7)
    joint_pos[env_ids, 6:8] = 0.0425  # Fully open
    
    # Zero velocities
    joint_vel[env_ids] = 0.0
    
    # Set the joint state
    robot.set_joint_position_target(joint_pos[env_ids], env_ids=env_ids)
    robot.set_joint_velocity_target(joint_vel[env_ids], env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)