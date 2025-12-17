# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for UR10 lift task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_rotation_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize object rotation during grasp.
    
    Encourages the robot to grasp the object without rotating it.
    Uses the change in orientation between current and initial quaternion.
    
    Returns:
        Penalty value proportional to rotation magnitude (num_envs,).
    """
    obj = env.scene["object"]
    
    # Current orientation (quaternion)
    quat_w = obj.data.root_quat_w  # [qw, qx, qy, qz]
    
    # Compute rotation magnitude from identity quaternion
    # For small rotations, |qx| + |qy| + |qz| approximates rotation angle
    rotation_magnitude = torch.abs(quat_w[:, 1]) + torch.abs(quat_w[:, 2]) + torch.abs(quat_w[:, 3])
    
    return rotation_magnitude


def object_translation_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize object translation during finger closing.
    
    Encourages stable grasps where the object doesn't slip during gripper closure.
    Compares current position to initial position within the episode.
    
    Returns:
        Penalty value proportional to XY displacement (num_envs,).
    """
    obj = env.scene["object"]
    
    # Current position
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
    
    Encourages the robot to grasp the object symmetrically with both fingers.
    Checks if both fingers are in contact with the object.
    
    Args:
        env: The environment instance.
        left_finger_body: Name of the left finger body.
        right_finger_body: Name of the right finger body.
    
    Returns:
        Penalty: 1.0 if asymmetric contact, 0.0 if symmetric (num_envs,).
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
    asymmetry = torch.abs(left_dist - right_dist)
    
    return asymmetry


def centered_grasp_reward(
    env: ManagerBasedRLEnv,
    threshold: float = 0.02,
) -> torch.Tensor:
    """Reward for centering the object between gripper fingers.
    
    Encourages the robot to position the gripper so the object
    is centered between the left and right fingers.
    
    Args:
        env: The environment instance.
        threshold: Distance threshold for considering grasp centered.
    
    Returns:
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
    
    # Object position
    obj_pos = obj.data.root_pos_w[:, :3]
    
    # Distance from object to gripper center
    dist_to_center = torch.norm(obj_pos - gripper_center, dim=-1)
    
    # Exponential reward (1.0 when perfectly centered)
    reward = torch.exp(-dist_to_center / threshold)
    
    return reward


def grasp_stability_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for stable grasps with low object velocity.
    
    Encourages the robot to grasp objects in a way that minimizes
    object movement during and after grasping.
    
    Returns:
        Reward inversely proportional to object velocity (num_envs,).
    """
    obj = env.scene["object"]
    
    # Get object linear and angular velocity
    lin_vel = obj.data.root_lin_vel_w
    ang_vel = obj.data.root_ang_vel_w
    
    # Combined velocity magnitude
    velocity_magnitude = torch.norm(lin_vel, dim=-1) + 0.1 * torch.norm(ang_vel, dim=-1)
    
    # Reward: high when velocity is low
    reward = 1.0 / (1.0 + velocity_magnitude)
    
    return reward


def object_height_reward(
    env: ManagerBasedRLEnv,
    max_height: float = 0.5,
    proximity_threshold: float = 0.1,
) -> torch.Tensor:
    """Dense reward based on object height when gripper is close.
    
    Similar to MetaIsaacGrasp's obj_height_reward but without state machine.
    Rewards the agent proportionally to how high the object is lifted,
    but only when the gripper is close to the object (actively grasping).
    
    Args:
        env: The environment instance.
        max_height: Maximum height for valid reward (meters).
        proximity_threshold: Maximum gripper-object distance to receive reward.
    
    Returns:
        Reward proportional to object height when conditions are met (num_envs,).
    """
    robot = env.scene["robot"]
    obj = env.scene["object"]
    
    # Get object position
    obj_pos = obj.data.root_pos_w[:, :3]
    obj_height = obj_pos[:, 2]  # Z coordinate
    
    # Get end-effector position
    ee_frame = env.scene["ee_frame"]
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]  # (num_envs, 3)
    
    # Calculate distance from gripper to object
    dist_to_obj = torch.norm(ee_pos - obj_pos, dim=-1)
    
    # Conditions for reward:
    # 1. Object height is in valid range (above ground, below max)
    condition_height = (obj_height > 0.0) & (obj_height < max_height)
    
    # 2. Gripper is close to object (actively grasping)
    condition_proximity = dist_to_obj < proximity_threshold
    
    # 3. Gripper is above ground level
    condition_gripper_up = ee_pos[:, 2] > 0.0
    
    # Combine all conditions
    all_conditions = condition_height & condition_proximity & condition_gripper_up
    
    # Reward = object height when all conditions met, else 0
    reward = torch.where(all_conditions, obj_height, torch.zeros_like(obj_height))
    
    return reward


def object_lift_progress_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.3,
) -> torch.Tensor:
    """Reward progress toward lifting object to target height.
    
    Provides smooth reward signal based on how close the object is
    to the target lift height.
    
    Args:
        env: The environment instance.
        target_height: Target height to lift object to (meters).
    
    Returns:
        Reward based on progress toward target height (num_envs,).
    """
    obj = env.scene["object"]
    
    # Get object height
    obj_height = obj.data.root_pos_w[:, 2]
    
    # Reward is fraction of target height achieved (capped at 1.0)
    progress = torch.clamp(obj_height / target_height, 0.0, 1.0)
    
    return progress


def finger_object_proximity_reward(
    env: ManagerBasedRLEnv,
    optimal_distance: float = 0.02,
) -> torch.Tensor:
    """Reward for optimal finger-to-object distance.
    
    Encourages fingers to be at an optimal distance from the object
    for a secure grasp.
    
    Args:
        env: The environment instance.
        optimal_distance: Target distance from finger to object surface.
    
    Returns:
        Reward: higher when fingers are at optimal distance (num_envs,).
    """
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
    """Reward when both fingers are in contact with the object.
    
    Encourages symmetric, two-finger grasps for stable object manipulation.
    Returns high reward only when BOTH fingers are close to the object.
    
    Args:
        env: The environment instance.
        contact_threshold: Maximum distance (m) to consider as "contact".
    
    Returns:
        Reward: 1.0 if both fingers in contact, decays if only one or none (num_envs,).
    """
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
    
    # Reward structure:
    # - Both fingers in contact: 1.0
    # - One finger in contact: 0.3
    # - No fingers in contact: 0.0
    both_contact = (left_in_contact & right_in_contact).float()
    one_contact = ((left_in_contact | right_in_contact) & ~(left_in_contact & right_in_contact)).float()
    
    reward = both_contact * 1.0 + one_contact * 0.3
    
    return reward


def symmetric_grasp_reward(
    env: ManagerBasedRLEnv,
    contact_threshold: float = 0.08,
) -> torch.Tensor:
    """Reward for symmetric finger positioning around the object.
    
    Encourages the gripper to approach the object such that both fingers
    are equidistant from the object center, leading to stable grasps.
    
    Args:
        env: The environment instance.
        contact_threshold: Distance threshold for proximity reward activation.
    
    Returns:
        Reward: higher when fingers are symmetric around object (num_envs,).
    """
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
    symmetry_reward = torch.exp(-dist_diff / 0.02)  # High when symmetric
    
    # Only apply when fingers are close enough to matter
    avg_dist = (left_dist + right_dist) / 2.0
    proximity_mask = (avg_dist < contact_threshold).float()
    
    # Combine: symmetry matters more when close to object
    reward = symmetry_reward * proximity_mask
    
    return reward


def grasp_force_reward(
    env: ManagerBasedRLEnv,
    min_closure: float = 0.01,
    max_closure: float = 0.04,
) -> torch.Tensor:
    """Reward for appropriate gripper closure (proxy for grasp force).
    
    Encourages the gripper to close on the object with appropriate force.
    Uses finger joint positions as proxy for grasp force.
    
    Args:
        env: The environment instance.
        min_closure: Minimum finger closure for reward (fully open ~0.0425).
        max_closure: Maximum finger closure for reward (fully closed ~0.0).
    
    Returns:
        Reward based on gripper closure state (num_envs,).
    """
    robot = env.scene["robot"]
    obj = env.scene["object"]
    
    # Get finger joint positions (Hand-E: 0.0425 = open, 0.0 = closed)
    # Joint names: hande_left_finger_joint, hande_right_finger_joint
    joint_pos = robot.data.joint_pos
    
    # Find finger joint indices
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
    # Good closure is between min_closure and max_closure
    in_good_range = (avg_closure > max_closure) & (avg_closure < min_closure + 0.02)
    near_object = avg_dist < 0.1
    
    reward = (in_good_range & near_object).float()
    
    return reward

