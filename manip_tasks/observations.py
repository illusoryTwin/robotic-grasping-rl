# Copyright (c) 2025 Ekaterina Mozhegova
#
# SPDX-License-Identifier: MIT

"""Custom observation functions for UR10 lift task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_orientation_in_robot_root_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The orientation (quaternion) of the object in the robot's root frame.
    
    Returns:
        Quaternion [qw, qx, qy, qz] of shape (num_envs, 4).
    """
    from omni.isaac.lab.utils.math import subtract_frame_transforms
    robot = env.scene["robot"]
    obj = env.scene["object"]
    object_pos_w = obj.data.root_pos_w[:, :3]
    object_quat_w = obj.data.root_quat_w  # (num_envs, 4) - [qw, qx, qy, qz]
    _, object_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w, object_quat_w
    )
    return object_quat_b


def wrist_camera_rgb(env: ManagerBasedRLEnv) -> torch.Tensor:
    """RGB image from wrist camera."""
    camera = env.scene["wrist_camera"]
    # Check if camera data is available (handles initialization phase)
    if not hasattr(camera._data, "output") or camera._data.output is None:
        # Return dummy RGB during initialization (height=480, width=640, channels=3)
        return torch.zeros(env.num_envs, 480, 640, 3, device=env.device)

    camera_data = camera.data.output["rgb"]
    # Return as (batch, height, width, channels)
    return camera_data[..., :3].float()


def wrist_camera_depth(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Depth image from wrist camera (normalized)."""
    camera = env.scene["wrist_camera"]
    # Check if camera data is available (handles initialization phase)
    if not hasattr(camera._data, "output") or camera._data.output is None:
        # Return dummy depth during initialization (height=480, width=640, channels=1)
        return torch.zeros(env.num_envs, 480, 640, 1, device=env.device)

    camera_data = camera.data.output["distance_to_image_plane"]
    # Normalize depth to [0, 1] range, clip at 2 meters
    depth_max = 2.0
    depth_normalized = torch.clamp(camera_data, 0.0, depth_max) / depth_max
    return depth_normalized.unsqueeze(-1)  # Add channel dimension


def visual_object_features(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Extract engineered visual features from camera data.
    This is more sample-efficient than raw images for RL.

    Similar to MetaIsaacGrasp but used continuously during learning.
    """
    # Get camera data
    camera = env.scene["wrist_camera"]

    # Check if camera is initialized and data is available
    if (not hasattr(camera, '_data') or camera._data is None or
        not hasattr(camera._data, "output") or camera._data.output is None or
        not camera._is_initialized):
        # Return dummy features if camera not initialized
        return torch.zeros(env.num_envs, 7, device=env.device)

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

