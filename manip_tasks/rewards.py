# Copyright (c) 2025 Ekaterina Mozhegova
#
# SPDX-License-Identifier: MIT

"""Custom reward functions for UR10 lift task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer, Camera
from omni.isaac.lab.utils.math import combine_frame_transforms
from manip_tasks.observations import visual_object_features

def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    print("object height", object.data.root_pos_w[:, 2])
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
    
    # Get end-effector orientation (quaternion)
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]  # (num_envs, 4) [w, x, y, z]
    
    # Print gripper orientation for first environment
    if env.cfg.scene.num_envs > 0:
        quat = ee_quat_w[0]
        # Convert quaternion to Euler angles for easier interpretation
        import numpy as np
        w, x, y, z = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()
        
        # Roll (x-axis rotation)
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        # Pitch (y-axis rotation)
        pitch = np.arcsin(2*(w*y - z*x))
        # Yaw (z-axis rotation)
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        
        # print(f"Gripper orientation - Quat: [{w:.3f}, {x:.3f}, {y:.3f}, {z:.3f}] | Euler (deg): [roll: {np.degrees(roll):.1f}, pitch: {np.degrees(pitch):.1f}, yaw: {np.degrees(yaw):.1f}]")

    # print("cube_pos_w", cube_pos_w)
    # print("ee_w", ee_w)
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


# def object_ee_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
# ) -> torch.Tensor:
#     """Reward the agent for reaching the object using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     object: RigidObject = env.scene[object_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
#     # Target object position: (num_envs, 3)
#     cube_pos_w = object.data.root_pos_w
#     # End-effector position: (num_envs, 3)
#     ee_w = ee_frame.data.target_pos_w[..., 0, :]

#     # print("cube_pos_w", cube_pos_w)
#     # print("ee_w", ee_w)
#     # Distance of the end-effector to the object: (num_envs,)
#     object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

#     return 1 - torch.tanh(object_ee_distance / std)


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


##
# Vision-based reward functions (using camera observations instead of privileged info)
##

def object_ee_distance_visual(
    env: ManagerBasedRLEnv,
    std: float,
) -> torch.Tensor:
    """Reward the agent for reaching the object using visual features from camera.

    Uses camera-based object detection instead of privileged ground truth position.
    """
    # Get visual features: [x_cam, y_cam, z_cam, distance, u_norm, v_norm, depth_norm]
    visual_features = visual_object_features(env)  # (num_envs, 7)

    # Extract distance to object in camera frame (feature index 3)
    object_distance = visual_features[:, 3]  # (num_envs,)

    # Return tanh-shaped reward based on distance
    return 1 - torch.tanh(object_distance / std)


def object_is_lifted_visual(
    env: ManagerBasedRLEnv,
    minimal_height: float,
) -> torch.Tensor:
    """Reward for lifting object, estimated from visual features.

    Uses camera depth and vertical position in image to estimate if object is lifted.
    Note: This is an approximation since absolute height requires camera calibration.
    """
    # Get visual features
    visual_features = visual_object_features(env)  # (num_envs, 7)

    # Extract normalized vertical position in image (feature index 5)
    # Objects higher in frame (smaller v_norm) are higher in world
    v_norm = visual_features[:, 5]  # (num_envs,)
    print("v_norm", v_norm)
    # Extract depth (feature index 6)
    depth_norm = visual_features[:, 6]  # (num_envs,)

    # Heuristic: Object is lifted if it appears higher in frame (v_norm < 0.4)
    # and is close to camera (depth_norm < 0.5, meaning < 1m away)
    # This is approximate but works for downward-facing wrist camera
    is_high_in_frame = v_norm < 0.4
    is_close = depth_norm < 0.5
    is_lifted = is_high_in_frame & is_close

    return torch.where(is_lifted, 1.0, 0.0)


def object_goal_distance_visual(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    minimal_height: float,
) -> torch.Tensor:
    """Reward for tracking goal pose, using visual estimation of object position.

    Combines visual lifting detection with distance to commanded goal.
    """
    # Get visual features
    visual_features = visual_object_features(env)  # (num_envs, 7)

    # Get commanded target position (in robot base frame)
    command = env.command_manager.get_command(command_name)  # (num_envs, 7)
    target_pos_b = command[:, :3]  # (num_envs, 3)

    # Transform camera-frame object position to robot base frame (approximate)
    # visual_features[0:3] = [x_cam, y_cam, z_cam]
    # For wrist camera pointing down, approximate transformation
    obj_pos_cam = visual_features[:, :3]  # (num_envs, 3)

    # Get end-effector frame to transform from camera to base
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]  # (num_envs, 4)

    # Transform object from camera frame to world frame (simplified)
    # obj_pos_w ≈ ee_pos_w + camera_offset + obj_pos_cam
    # This is approximate; for accurate transform use full camera extrinsics
    obj_pos_w_est = ee_pos_w + obj_pos_cam * torch.tensor([1.0, 1.0, -1.0], device=env.device)

    # Get robot base transform
    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_state_w[:, :3]
    robot_quat_w = robot.data.root_state_w[:, 3:7]

    # Transform target from base to world frame
    target_pos_w, _ = combine_frame_transforms(robot_pos_w, robot_quat_w, target_pos_b)

    # Distance from estimated object position to target
    distance = torch.norm(target_pos_w - obj_pos_w_est, dim=1)

    # Check if object is lifted (visual estimate)
    is_lifted = object_is_lifted_visual(env, minimal_height)

    # Reward: distance to goal if lifted
    return is_lifted * (1 - torch.tanh(distance / std))

