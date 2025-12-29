# Copyright (c) 2025 Ekaterina Mozhegova
#
# SPDX-License-Identifier: MIT

"""Custom reward functions for UR10 lift task."""

from __future__ import annotations

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv



def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    # print("object height", object.data.root_pos_w[:, 2])
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

# def object_is_lifted(
#     env: ManagerBasedRLEnv,
#     minimal_height: float,
#     max_velocity: float = 0.05,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:

#     object = env.scene[object_cfg.name]

#     height = object.data.root_pos_w[:, 2]
#     vel = torch.norm(object.data.root_lin_vel_w, dim=-1)

#     height_reward = torch.clamp(height / minimal_height, 0.0, 1.0)
#     stability = torch.exp(-vel / max_velocity)

#     return height_reward * stability


def object_is_lifted_and_grasped(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height while end-effector is close."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    
    # Check if object is lifted
    lifted = object.data.root_pos_w[:, 2] > minimal_height
    
    # Check if end-effector is close to object (indicating grasp)
    object_pos = object.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(object_pos - ee_pos, dim=1)
    close = object_ee_distance < 0.04
    
    # Only reward if both conditions are met: lifted AND close
    return torch.where(lifted & close, 1.0, 0.0)


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



def object_rotation_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize object rotation during grasp.
    
    Encourages the robot to grasp the object without rotating it.
    Uses the change in orientation between current and initial quaternion.
    """
    obj = env.scene["object"]
    
    # Current orientation (quaternion)
    quat_w = obj.data.root_quat_w  # [qw, qx, qy, qz]
    
    # Compute rotation magnitude from identity quaternion
    # For small rotations, |qx| + |qy| + |qz| approximates rotation angle
    rotation_magnitude = torch.abs(quat_w[:, 1]) + torch.abs(quat_w[:, 2]) + torch.abs(quat_w[:, 3])
    
    return rotation_magnitude



def center_gripper_on_object(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for centering end effector between long sides of tetra pak object."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # Get object and end effector positions
    object_pos = object.data.root_pos_w[:, :3]  # (num_envs, 3)
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)
    
    # Get object orientation (quaternion)
    object_quat = object.data.root_quat_w  # (num_envs, 4)
    
    # Transform end effector position to object frame
    # For tetra pak: long sides are along Y axis (front/back faces)
    # We want equal distance from front and back faces
    
    # Rotation matrix from quaternion (simplified for Y-axis)
    # quat = [w, x, y, z], rotation matrix R
    w, x, y, z = object_quat[:, 0], object_quat[:, 1], object_quat[:, 2], object_quat[:, 3]
    
    # Y-axis direction vector in world frame (object's local Y transformed to world)
    y_world = torch.stack([
        2 * (x*y - w*z),
        1 - 2 * (x*x + z*z), 
        2 * (y*z + w*x)
    ], dim=1)  # (num_envs, 3)
    
    # Vector from object center to end effector
    ee_to_obj = ee_pos - object_pos  # (num_envs, 3)
    
    # Project this vector onto the object's Y-axis (long axis)
    projection_y = torch.sum(ee_to_obj * y_world, dim=1)  # (num_envs,)
    
    # Reward is higher when projection is close to zero (centered)
    # Use absolute value since we want minimal distance from center
    centering_error = torch.abs(projection_y)
    
    return 1 - torch.tanh(centering_error / std)


def object_translation_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize object translation during finger closing."""
    obj = env.scene["object"]
    pos_w = obj.data.root_pos_w[:, :3]
    # Default position (from scene config)
    default_pos = obj.data.default_root_state[:, :3]
    
    # Compute XY displacement (ignore Z since we want lifting)
    xy_displacement = torch.sqrt(
        (pos_w[:, 0] - default_pos[:, 0])**2 + 
        (pos_w[:, 1] - default_pos[:, 1])**2
    )
    
    return xy_displacement


def asymmetric_finger_contact_penalty(
    env: ManagerBasedRLEnv,
    left_finger_body: str = "hande_left_finger",
    right_finger_body: str = "hande_right_finger",
) -> torch.Tensor:
    """Penalize asymmetric finger contacts with the object.
    Penalty: 1.0 if asymmetric contact, 0.0 if symmetric.
    """
    robot = env.scene["robot"]
    
    # Get finger positions
    left_idx = robot.find_bodies(left_finger_body)[0][0]
    right_idx = robot.find_bodies(right_finger_body)[0][0]
    
    left_finger_pos = robot.data.body_pos_w[:, left_idx, :]
    right_finger_pos = robot.data.body_pos_w[:, right_idx, :]
    
    # Get object position
    obj = env.scene["object"]
    obj_pos = obj.data.root_pos_w[:, :3]
    
    # Compute distances from each finger to object center
    left_dist = torch.norm(left_finger_pos - obj_pos, dim=-1)
    right_dist = torch.norm(right_finger_pos - obj_pos, dim=-1)
    
    # Asymmetry penalty: difference in distances
    return torch.abs(left_dist - right_dist)
    


def centered_grasp_reward(
    env: ManagerBasedRLEnv,
    threshold: float = 0.02,
) -> torch.Tensor:
    """Reward for centering the object between gripper fingers.
    Reward: 1.0 if centered, decays with distance (num_envs,).
    """
    robot = env.scene["robot"]
    obj = env.scene["object"]
    
    # Get gripper center (midpoint between fingers)
    left_idx = robot.find_bodies("hande_left_finger")[0][0]
    right_idx = robot.find_bodies("hande_right_finger")[0][0]
    
    left_finger_pos = robot.data.body_pos_w[:, left_idx, :]
    right_finger_pos = robot.data.body_pos_w[:, right_idx, :]
    
    gripper_center = (left_finger_pos + right_finger_pos) / 2.0
    
    # Distance from object to gripper center
    obj_pos = obj.data.root_pos_w[:, :3]
    dist_to_center = torch.norm(obj_pos - gripper_center, dim=-1)
    
    # reward 1.0 when perfectly centered)
    return torch.exp(-dist_to_center / threshold)
    
    

def grasp_stability_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for stable grasps with low object velocity."""
    obj = env.scene["object"]
    
    # Combined linear and angular velocity magnitude
    lin_vel = obj.data.root_lin_vel_w
    ang_vel = obj.data.root_ang_vel_w
    velocity_magnitude = torch.norm(lin_vel, dim=-1) + 0.1 * torch.norm(ang_vel, dim=-1)
    
    # high reward when velocity is low
    return 1.0 / (1.0 + velocity_magnitude)
    


def object_height_reward(
    env: ManagerBasedRLEnv,
    max_height: float = 0.5,
    proximity_threshold: float = 0.1,
) -> torch.Tensor:
    """Dense reward based on object height when gripper is close.
    
    Rewards the agent proportionally to how high the object is lifted,
    but only when the gripper is close to the object (actively grasping).
    
    Args:
        env: The environment instance.
        max_height: Maximum height for valid reward (meters).
        proximity_threshold: Maximum gripper-object distance to receive reward.
    
    Returns:
        Reward proportional to object height when conditions are met (num_envs,).
    """
    obj = env.scene["object"]
    
    # Get object position
    obj_pos = obj.data.root_pos_w[:, :3]
    obj_height = obj_pos[:, 2]  # Z coordinate
    
    # Get end-effector position
    ee_frame = env.scene["ee_frame"]
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]  # (num_envs, 3)
    
    # distance from gripper to object
    dist_to_obj = torch.norm(ee_pos - obj_pos, dim=-1)
    
    # 1. Object height is in valid range (above ground, below max)
    condition_height = (obj_height > 0.0) & (obj_height < max_height)
    # 2. Gripper is close to object (actively grasping)
    condition_proximity = dist_to_obj < proximity_threshold
    # 3. Gripper is above ground level
    condition_gripper_up = ee_pos[:, 2] > 0.0
    
    # Combine all conditions
    all_conditions = condition_height & condition_proximity & condition_gripper_up
    reward = torch.where(all_conditions, obj_height, torch.zeros_like(obj_height))
    
    return reward


def object_lift_progress_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.3,
) -> torch.Tensor:
    """Reward progress toward lifting object to target height."""
    obj = env.scene["object"]
    obj_height = obj.data.root_pos_w[:, 2]
    
    return torch.clamp(obj_height / target_height, 0.0, 1.0)
    


def finger_object_proximity_reward(
    env: ManagerBasedRLEnv,
    optimal_distance: float = 0.02,
) -> torch.Tensor:
    """Reward for optimal finger-to-object distance."""
    robot = env.scene["robot"]
    obj = env.scene["object"]
    
    # Get finger positions
    left_idx = robot.find_bodies("hande_left_finger")[0][0]
    right_idx = robot.find_bodies("hande_right_finger")[0][0]
    
    left_finger_pos = robot.data.body_pos_w[:, left_idx, :]
    right_finger_pos = robot.data.body_pos_w[:, right_idx, :]
    
    # Object position
    obj_pos = obj.data.root_pos_w[:, :3]
    
    # Distances
    left_dist = torch.norm(left_finger_pos - obj_pos, dim=-1)
    right_dist = torch.norm(right_finger_pos - obj_pos, dim=-1)
    
    # Reward for being close to optimal distance
    left_reward = torch.exp(-torch.abs(left_dist - optimal_distance) / 0.01)
    right_reward = torch.exp(-torch.abs(right_dist - optimal_distance) / 0.01)
    
    return (left_reward + right_reward) / 2.0


def both_fingers_contact_reward(
    env: ManagerBasedRLEnv,
    contact_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward when both fingers are in contact with the object."""
    robot = env.scene["robot"]
    obj = env.scene["object"]
    
    # Get finger positions
    left_idx = robot.find_bodies("hande_left_finger")[0][0]
    right_idx = robot.find_bodies("hande_right_finger")[0][0]
    
    left_finger_pos = robot.data.body_pos_w[:, left_idx, :]
    right_finger_pos = robot.data.body_pos_w[:, right_idx, :]
    
    # Object position
    obj_pos = obj.data.root_pos_w[:, :3]
    
    # Compute distances from each finger to object center
    left_dist = torch.norm(left_finger_pos - obj_pos, dim=-1)
    right_dist = torch.norm(right_finger_pos - obj_pos, dim=-1)
    
    # Check if each finger is in contact
    left_in_contact = left_dist < contact_threshold
    right_in_contact = right_dist < contact_threshold
    
    # Both fingers in contact: 1.0
    # One finger in contact: 0.3
    # No fingers in contact: 0.0
    both_contact = (left_in_contact & right_in_contact).float()
    one_contact = ((left_in_contact | right_in_contact) & ~(left_in_contact & right_in_contact)).float()
    
    return both_contact * 1.0 + one_contact * 0.3
    

def symmetric_grasp_reward(
    env: ManagerBasedRLEnv,
    contact_threshold: float = 0.08,
) -> torch.Tensor:
    """Reward for symmetric finger positioning around the object."""
    robot = env.scene["robot"]
    obj = env.scene["object"]
    
    # Get finger positions
    left_idx = robot.find_bodies("hande_left_finger")[0][0]
    right_idx = robot.find_bodies("hande_right_finger")[0][0]
    
    left_finger_pos = robot.data.body_pos_w[:, left_idx, :]
    right_finger_pos = robot.data.body_pos_w[:, right_idx, :]
    
    # Object position
    obj_pos = obj.data.root_pos_w[:, :3]
    
    # Compute distances from each finger to object center
    left_dist = torch.norm(left_finger_pos - obj_pos, dim=-1)
    right_dist = torch.norm(right_finger_pos - obj_pos, dim=-1)
    
    # Symmetry: reward when distances are equal
    dist_diff = torch.abs(left_dist - right_dist)
    symmetry_reward = torch.exp(-dist_diff / 0.02)
    
    # Only apply when fingers are close enough to matter
    avg_dist = (left_dist + right_dist) / 2.0
    proximity_mask = (avg_dist < contact_threshold).float()
    
    return symmetry_reward * proximity_mask
    

def grasp_force_reward(
    env: ManagerBasedRLEnv,
    min_closure: float = 0.01,
    max_closure: float = 0.04,
) -> torch.Tensor:
    """Reward for appropriate gripper closure (proxy for grasp force).
    
    min_closure: Minimum finger closure for reward (fully open ~0.0425).
    max_closure: Maximum finger closure for reward (fully closed ~0.0).
    """
    robot = env.scene["robot"]
    obj = env.scene["object"]
    
    joint_pos = robot.data.joint_pos
    left_joint_idx = robot.find_joints("hande_left_finger_joint")[0][0]
    right_joint_idx = robot.find_joints("hande_right_finger_joint")[0][0]
    
    left_pos = joint_pos[:, left_joint_idx]
    right_pos = joint_pos[:, right_joint_idx]
    
    # Average closure (lower = more closed)
    avg_closure = (left_pos + right_pos) / 2.0
    
    # Get finger positions and object position to check proximity
    left_body_idx = robot.find_bodies("hande_left_finger")[0][0]
    right_body_idx = robot.find_bodies("hande_right_finger")[0][0]
    
    left_finger_pos = robot.data.body_pos_w[:, left_body_idx, :]
    right_finger_pos = robot.data.body_pos_w[:, right_body_idx, :]
    obj_pos = obj.data.root_pos_w[:, :3]
    
    avg_dist = (torch.norm(left_finger_pos - obj_pos, dim=-1) + 
                torch.norm(right_finger_pos - obj_pos, dim=-1)) / 2.0
    
    # Reward gripper closure when close to object
    in_good_range = (avg_closure > max_closure) & (avg_closure < min_closure + 0.02)
    near_object = avg_dist < 0.1
    
    return (in_good_range & near_object).float()
    

def grasp_success_reward(
    env: ManagerBasedRLEnv,
    lift_height_threshold: float = 0.15,
) -> torch.Tensor:
    """Sparse reward when object is successfully lifted.
    1.0 if object is lifted, 0.0 otherwise.
    """
    obj = env.scene["object"]
    robot = env.scene["robot"]
    
    # Get object height relative to robot base (approximate table height)
    obj_height = obj.data.root_pos_w[:, 2]
    robot_base_height = robot.data.root_pos_w[:, 2]
    relative_height = obj_height - robot_base_height
    
    # Success if object is above threshold
    return (relative_height > lift_height_threshold).float()
    

def gripper_distance_reward(
    env: ManagerBasedRLEnv,
    alpha: float = 8.0,
) -> torch.Tensor:
    """Dense reward based on gripper-to-object distance.
    
    Uses exponential decay: reward = exp(-alpha * distance), 
    encourages the gripper to get closer to the object.
    """
    obj = env.scene["object"]
    
    # Get gripper position (end-effector)
    ee_frame = env.scene["ee_frame"]
    gripper_pos = ee_frame.data.target_pos_w[..., 0, :]
    
    # Get object position
    obj_pos = obj.data.root_pos_w[:, :3]
    
    # Compute distance
    distance = torch.norm(gripper_pos - obj_pos, dim=-1)
    
    return torch.exp(-alpha * distance)
    

def object_height_dense_reward(
    env: ManagerBasedRLEnv,
    max_height: float = 0.4,
    proximity_threshold: float = 0.1,
) -> torch.Tensor:
    """Dense reward proportional to object height when gripper is close.
    
    The reward equals the object height when:
    1. Height is in valid range (0, max_height)
    2. Gripper is close to object (< proximity_threshold)
    3. Gripper is above ground
    """
    obj = env.scene["object"]
    robot = env.scene["robot"]
    
    # Get object height relative to robot base
    obj_height = obj.data.root_pos_w[:, 2]
    robot_base_height = robot.data.root_pos_w[:, 2]
    relative_height = obj_height - robot_base_height
    
    # Get gripper position
    ee_frame = env.scene["ee_frame"]
    gripper_pos = ee_frame.data.target_pos_w[..., 0, :]
    gripper_height = gripper_pos[:, 2] - robot_base_height
    
    # Get object position
    obj_pos = obj.data.root_pos_w[:, :3]
    
    # Compute distance to object
    distance = torch.norm(gripper_pos - obj_pos, dim=-1)
    
    condition1 = (relative_height > 0) & (relative_height < max_height)  # Valid height range
    condition2 = distance < proximity_threshold  # Gripper close to object
    condition3 = gripper_height > 0  # Gripper above ground
    
    all_conditions = condition1 & condition2 & condition3
    return torch.where(all_conditions, relative_height, torch.zeros_like(relative_height))
    

def time_penalty_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Small constant penalty per step to encourage faster grasping."""
    return torch.full((env.num_envs,), 0.05, device=env.device)



def finger_object_distance_shaping(
    env: ManagerBasedRLEnv,
    std: float = 0.02,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense reward shaping for finger-to-object distance (both fingers)."""
    robot = env.scene["robot"]
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get finger positions
    left_idx = robot.find_bodies("hande_left_finger")[0][0]
    right_idx = robot.find_bodies("hande_right_finger")[0][0]
    
    left_finger_pos = robot.data.body_pos_w[:, left_idx, :]
    right_finger_pos = robot.data.body_pos_w[:, right_idx, :]
    
    # Object position
    obj_pos = object.data.root_pos_w[:, :3]
    
    # Distances from each finger to object
    left_dist = torch.norm(left_finger_pos - obj_pos, dim=-1)
    right_dist = torch.norm(right_finger_pos - obj_pos, dim=-1)
    
    # Average distance reward using tanh kernel
    avg_dist = (left_dist + right_dist) / 2.0
    return 1 - torch.tanh(avg_dist / std)


def both_fingers_contact_soft(
    env: ManagerBasedRLEnv,
    std: float = 0.02,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Soft reward for both fingers being in contact with object."""
    robot = env.scene["robot"]
    object: RigidObject = env.scene[object_cfg.name]

    # Get finger positions
    left_idx = robot.find_bodies("hande_left_finger")[0][0]
    right_idx = robot.find_bodies("hande_right_finger")[0][0]

    left_finger_pos = robot.data.body_pos_w[:, left_idx, :]
    right_finger_pos = robot.data.body_pos_w[:, right_idx, :]

    # Object position
    obj_pos = object.data.root_pos_w[:, :3]

    # Soft contact rewards (exponential decay)
    left_dist = torch.norm(left_finger_pos - obj_pos, dim=-1)
    right_dist = torch.norm(right_finger_pos - obj_pos, dim=-1)

    left_contact = torch.exp(-left_dist / std)
    right_contact = torch.exp(-right_dist / std)

    # Reward is product of both contacts (encourages both fingers to be close)
    return left_contact * right_contact


def object_centered_between_fingers(
    env: ManagerBasedRLEnv,
    std: float = 0.02,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward when object's COM is centered BETWEEN gripper fingers.

    This reward encourages the object to be:
    1. On the line connecting the two fingers (perpendicular distance = 0)
    2. At the midpoint between the fingers (centered)
    3. Actually between the fingers, not outside them

    Args:
        env: The RL environment.
        std: Standard deviation for reward shaping (smaller = sharper).
        object_cfg: Configuration for the object entity.

    Returns:
        Reward tensor of shape (num_envs,). Higher when object is centered between fingers.
    """
    robot = env.scene["robot"]
    object: RigidObject = env.scene[object_cfg.name]

    # Get finger positions
    left_idx = robot.find_bodies("hande_left_finger")[0][0]
    right_idx = robot.find_bodies("hande_right_finger")[0][0]

    left_finger_pos = robot.data.body_pos_w[:, left_idx, :]  # (num_envs, 3)
    right_finger_pos = robot.data.body_pos_w[:, right_idx, :]  # (num_envs, 3)

    # Object COM position
    obj_pos = object.data.root_pos_w[:, :3]  # (num_envs, 3)

    # Compute midpoint between fingers
    midpoint = (left_finger_pos + right_finger_pos) / 2.0  # (num_envs, 3)

    # Compute finger axis (direction from left to right finger)
    finger_axis = right_finger_pos - left_finger_pos  # (num_envs, 3)
    finger_axis_length = torch.norm(finger_axis, dim=-1, keepdim=True) + 1e-6
    finger_axis_normalized = finger_axis / finger_axis_length  # (num_envs, 3)
    half_span = finger_axis_length.squeeze(-1) / 2.0  # (num_envs,) - half distance between fingers

    # Vector from midpoint to object
    midpoint_to_obj = obj_pos - midpoint  # (num_envs, 3)

    # Project object position onto finger axis
    # This gives us how far along the axis the object is from the midpoint
    # Positive = toward right finger, Negative = toward left finger
    projection_along_axis = torch.sum(midpoint_to_obj * finger_axis_normalized, dim=-1)  # (num_envs,)

    # Compute perpendicular distance from object to finger axis
    projection_vector = projection_along_axis.unsqueeze(-1) * finger_axis_normalized
    perpendicular_vector = midpoint_to_obj - projection_vector
    perpendicular_dist = torch.norm(perpendicular_vector, dim=-1)  # (num_envs,)

    # Check if object is BETWEEN the fingers (projection within [-half_span, +half_span])
    # If |projection| > half_span, object is outside the finger span
    abs_projection = torch.abs(projection_along_axis)

    # Distance outside the finger span (0 if between fingers)
    outside_span_dist = torch.clamp(abs_projection - half_span, min=0.0)  # (num_envs,)

    # Combined reward:
    # 1. Reward for being on the axis (low perpendicular distance)
    # 2. Reward for being centered (projection close to 0)
    # 3. Penalty for being outside the finger span
    on_axis_reward = torch.exp(-perpendicular_dist / std)
    centered_reward = torch.exp(-abs_projection / std)  # Reward for being at midpoint
    between_fingers_reward = torch.exp(-outside_span_dist / std)  # 1.0 if between, decays if outside

    # Multiply all: high only when all conditions are met
    return on_axis_reward * centered_reward * between_fingers_reward


def penalize_non_finger_contact(
    env: ManagerBasedRLEnv,
    contact_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize contact of object with non-gripper links.

    This reward function:
    - Returns 0 (no penalty) when all non-finger links are far from object
    - Returns negative values when non-finger links are close to object
    - Uses exponential decay to create smooth gradients

    Args:
        env: The RL environment.
        contact_threshold: Distance threshold for considering contact (in meters).
        object_cfg: Configuration for the object entity.

    Returns:
        Penalty tensor of shape (num_envs,). Values are 0 or negative.
    """
    robot = env.scene["robot"]
    object: RigidObject = env.scene[object_cfg.name]

    # Get object position (num_envs, 3)
    obj_pos = object.data.root_pos_w[:, :3]

    # Get all robot body positions (num_envs, num_bodies, 3)
    all_body_positions = robot.data.body_pos_w

    # Find finger body indices to exclude them
    left_finger_idx = robot.find_bodies("hande_left_finger")[0][0]
    right_finger_idx = robot.find_bodies("hande_right_finger")[0][0]
    finger_indices = {left_finger_idx, right_finger_idx}

    # Get indices of all non-finger bodies
    num_bodies = all_body_positions.shape[1]
    non_finger_indices = [i for i in range(num_bodies) if i not in finger_indices]

    # Get positions of non-finger bodies (num_envs, num_non_finger_bodies, 3)
    non_finger_positions = all_body_positions[:, non_finger_indices, :]

    # Compute distances from object to each non-finger link
    # obj_pos: (num_envs, 3) -> (num_envs, 1, 3) for broadcasting
    # non_finger_positions: (num_envs, num_non_finger_bodies, 3)
    obj_pos_expanded = obj_pos.unsqueeze(1)
    distances = torch.norm(non_finger_positions - obj_pos_expanded, dim=-1)  # (num_envs, num_non_finger_bodies)

    # Find minimum distance to any non-finger link for each environment
    min_distances = torch.min(distances, dim=-1)[0]  # (num_envs,)

    # Penalize when min_distance < contact_threshold using exponential penalty
    # When distance is large: penalty -> 0
    # When distance is small: penalty -> -1
    penalty = -torch.exp(-min_distances / contact_threshold) + 1.0

    # Only apply penalty when actually close (distance < 2*contact_threshold)
    # This avoids small penalties when robot is far away
    penalty = torch.where(
        min_distances < 2 * contact_threshold,
        penalty,
        torch.zeros_like(penalty)
    )

    return penalty


def gripper_contact_reward(
    env: ManagerBasedRLEnv,
    finger_std: float = 0.02,
    non_finger_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Combined reward: encourage finger contact, penalize non-finger contact.

    This reward function:
    - Positive reward when gripper fingers are close to the object
    - Negative penalty when other robot links are close to the object
    - The reward encourages proper grasping with fingers only

    Args:
        env: The RL environment.
        finger_std: Standard deviation for finger contact reward (smaller = sharper).
        non_finger_threshold: Distance threshold for non-finger penalty.
        object_cfg: Configuration for the object entity.

    Returns:
        Combined reward tensor of shape (num_envs,).
        Positive values indicate good finger contact, negative indicate bad contact.
    """
    robot = env.scene["robot"]
    object: RigidObject = env.scene[object_cfg.name]

    # Get object position (num_envs, 3)
    obj_pos = object.data.root_pos_w[:, :3]

    # === FINGER CONTACT REWARD (positive) ===
    # Get finger positions
    left_finger_idx = robot.find_bodies("hande_left_finger")[0][0]
    right_finger_idx = robot.find_bodies("hande_right_finger")[0][0]

    left_finger_pos = robot.data.body_pos_w[:, left_finger_idx, :]
    right_finger_pos = robot.data.body_pos_w[:, right_finger_idx, :]

    # Compute distances from fingers to object
    left_dist = torch.norm(left_finger_pos - obj_pos, dim=-1)
    right_dist = torch.norm(right_finger_pos - obj_pos, dim=-1)

    # Soft contact rewards (exponential decay) - higher when closer
    left_contact = torch.exp(-left_dist / finger_std)
    right_contact = torch.exp(-right_dist / finger_std)

    # Product of both contacts (encourages BOTH fingers to be close)
    finger_reward = left_contact * right_contact

    # === NON-FINGER CONTACT PENALTY (negative) ===
    # Get all robot body positions
    all_body_positions = robot.data.body_pos_w
    finger_indices = {left_finger_idx, right_finger_idx}

    # Get indices of non-finger bodies
    num_bodies = all_body_positions.shape[1]
    non_finger_indices = [i for i in range(num_bodies) if i not in finger_indices]

    # Get positions of non-finger bodies
    non_finger_positions = all_body_positions[:, non_finger_indices, :]

    # Compute distances from object to each non-finger link
    obj_pos_expanded = obj_pos.unsqueeze(1)
    distances = torch.norm(non_finger_positions - obj_pos_expanded, dim=-1)

    # Find minimum distance to any non-finger link
    min_non_finger_dist = torch.min(distances, dim=-1)[0]

    # Penalty: exponential when close, zero when far
    # Returns value in [0, 1] where 1 = very close (bad), 0 = far (good)
    non_finger_penalty = torch.exp(-min_non_finger_dist / non_finger_threshold)

    # Only apply penalty when actually close
    non_finger_penalty = torch.where(
        min_non_finger_dist < 2 * non_finger_threshold,
        non_finger_penalty,
        torch.zeros_like(non_finger_penalty)
    )

    # === COMBINED REWARD ===
    # finger_reward: [0, 1] positive (good finger contact)
    # non_finger_penalty: [0, 1] negative (bad non-finger contact)
    # Result: positive when fingers touch, negative when other parts touch
    combined = finger_reward - non_finger_penalty

    return combined