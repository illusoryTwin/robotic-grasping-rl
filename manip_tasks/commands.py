# Copyright (c) 2025 Ekaterina Mozhegova
#
# SPDX-License-Identifier: MIT

"""Custom command generators for UR10 manipulation tasks."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import CommandTerm, CommandTermCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.utils import configclass

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


class ObjectPoseCommand(CommandTerm):
    """Command that tracks object pose with configurable offset.

    This command generator outputs the object's pose (position + orientation)
    with an optional offset, typically used to command the end-effector to
    a position relative to the object.
    """

    cfg: "ObjectPoseCommandCfg"

    def __init__(self, cfg: "ObjectPoseCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        # Get object asset
        self.object: RigidObject = env.scene[cfg.object_name]

        # Get robot asset (for visualization reference frame)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Store offset
        self.z_offset = cfg.z_offset

        # Create command buffer: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
        self.pose_command = torch.zeros(self.num_envs, 7, device=self.device)

        # Set up visualization
        if self.cfg.debug_vis:
            self._setup_visualization()

    def __str__(self) -> str:
        return f"ObjectPoseCommand(object={self.cfg.object_name}, z_offset={self.z_offset})"

    @property
    def command(self) -> torch.Tensor:
        """The current command: object pose with offset."""
        return self.pose_command

    def _update_metrics(self):
        """Update metrics (optional)."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample command - just update from object pose."""
        self._update_command()

    def _update_command(self):
        """Update command to track object pose with offset."""
        # Get object pose in world frame
        object_pos = self.object.data.root_pos_w.clone()  # (num_envs, 3)
        object_quat = self.object.data.root_quat_w.clone()  # (num_envs, 4)

        # Apply Z offset
        object_pos[:, 2] += self.z_offset

        # Store command: [pos, quat]
        self.pose_command[:, :3] = object_pos
        self.pose_command[:, 3:7] = object_quat

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization."""
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self._setup_visualization()
            self.goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Callback for debug visualization."""
        if hasattr(self, "goal_visualizer"):
            # Update marker positions
            self.goal_visualizer.visualize(self.pose_command[:, :3], self.pose_command[:, 3:7])

    def _setup_visualization(self):
        """Setup visualization markers."""
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/Command/object_pose_goal"
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.goal_visualizer = VisualizationMarkers(marker_cfg)


@configclass
class ObjectPoseCommandCfg(CommandTermCfg):
    """Configuration for object pose command generator."""

    class_type: type = ObjectPoseCommand

    asset_name: str = "robot"
    """Name of the robot asset (for reference)."""

    object_name: str = "object"
    """Name of the object asset to track."""

    z_offset: float = 0.1
    """Offset above object along Z axis (meters). Default 10cm."""

    debug_vis: bool = True
    """Whether to visualize the target pose."""

    resampling_time_range: tuple[float, float] = (1.0, 1.0)
    """Time range for resampling (not used, but required by base class)."""
