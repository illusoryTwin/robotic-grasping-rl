# Copyright (c) 2025 Ekaterina Mozhegova
#
# SPDX-License-Identifier: MIT

"""Two-stage grasping environment.

Stage 1 (RL): Policy predicts 6-DoF grasp pose from observations
Stage 2 (IK + Scripted): Robot executes grasp via differential IK with state machine
"""

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, quat_rotate

if TYPE_CHECKING:
    from .two_stage_grasp_env_cfg import TwoStageGraspEnvCfg

from .two_stage_grasp_env_cfg import GraspPhase


class TwoStageGraspEnv(ManagerBasedRLEnv):
    """Two-stage grasping environment with RL pose prediction and IK execution."""

    cfg: TwoStageGraspEnvCfg

    def __init__(self, cfg: TwoStageGraspEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: Environment configuration.
            render_mode: Render mode for the environment.
        """
        super().__init__(cfg, render_mode, **kwargs)

        # State machine tracking
        self.phase = torch.full(
            (self.num_envs,), GraspPhase.OBSERVE, dtype=torch.int32, device=self.device
        )
        self.phase_step_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Grasp targets (set by RL policy)
        self.grasp_pose_target = torch.zeros((self.num_envs, 7), device=self.device)  # pos + quat
        self.approach_pose_target = torch.zeros((self.num_envs, 7), device=self.device)
        self.lift_pose_target = torch.zeros((self.num_envs, 7), device=self.device)

        # Gripper state
        self.gripper_closed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Episode rewards (accumulated)
        self.episode_rewards = torch.zeros(self.num_envs, device=self.device)

        # Setup IK controller
        self._setup_ik_controller()

        # Action dimension for 6-DoF pose
        self._action_dim = 6  # [x, y, z, roll, pitch, yaw]

    def _setup_ik_controller(self):
        """Initialize the differential IK controller."""
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.1},
        )
        self.ik_controller = DifferentialIKController(
            ik_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )

        # Robot entity configuration for IK
        self.robot_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            body_names=["hande_end"],
        )
        self.robot_entity_cfg.resolve(self.scene)

        # Gripper joint IDs
        self.gripper_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=[".*finger.*"],
        )
        self.gripper_entity_cfg.resolve(self.scene)

        # Jacobian index for fixed-base robot
        robot = self.scene["robot"]
        if robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

    def step(self, action: torch.Tensor):
        """Execute one step of the environment.

        During OBSERVE phase: action is the predicted grasp pose
        During other phases: action is ignored (scripted execution)

        Args:
            action: Policy action tensor of shape (num_envs, 6) for [x, y, z, roll, pitch, yaw]

        Returns:
            Tuple of (obs, reward, terminated, truncated, info)
        """
        # Phase 1: RL prediction (only on OBSERVE phase)
        observe_mask = self.phase == GraspPhase.OBSERVE
        if observe_mask.any():
            self._process_rl_action(action, observe_mask)
            self._transition_to_approach(observe_mask)

        # Phase 2+: Scripted execution with IK
        # Run multiple substeps for smooth motion
        for _ in range(self.cfg.ik_substeps):
            self._execute_scripted_motion()
            self._physics_step()
            self._update_phase_transitions()

            # Check if all environments are in EVALUATE phase
            if (self.phase == GraspPhase.EVALUATE).all():
                break

        # Compute observations, rewards, and terminations
        obs = self._get_observations()
        rewards = self._compute_rewards()
        terminated = self._check_termination()
        truncated = torch.zeros_like(terminated)

        # Reset environments that are done
        done_mask = terminated | truncated
        if done_mask.any():
            done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            self._reset_idx(done_ids)

        return obs, rewards, terminated, truncated, {}

    def _process_rl_action(self, action: torch.Tensor, mask: torch.Tensor):
        """Convert RL action (6D pose) to grasp target.

        Args:
            action: Raw policy output in range [-1, 1], shape (num_envs, 6)
            mask: Boolean mask for environments in OBSERVE phase
        """
        # Extract position and orientation from action
        pos_action = action[mask, :3]  # [x, y, z] in [-1, 1]
        rot_action = action[mask, 3:6]  # [roll, pitch, yaw] in [-1, 1]

        # Scale position to workspace bounds
        pos_min = torch.tensor([
            self.cfg.grasp_pos_range["x"][0],
            self.cfg.grasp_pos_range["y"][0],
            self.cfg.grasp_pos_range["z"][0],
        ], device=self.device)
        pos_max = torch.tensor([
            self.cfg.grasp_pos_range["x"][1],
            self.cfg.grasp_pos_range["y"][1],
            self.cfg.grasp_pos_range["z"][1],
        ], device=self.device)
        pos = (pos_action + 1.0) / 2.0 * (pos_max - pos_min) + pos_min

        # Scale rotation to bounds
        rot_min = torch.tensor([
            self.cfg.grasp_rot_range["roll"][0],
            self.cfg.grasp_rot_range["pitch"][0],
            self.cfg.grasp_rot_range["yaw"][0],
        ], device=self.device)
        rot_max = torch.tensor([
            self.cfg.grasp_rot_range["roll"][1],
            self.cfg.grasp_rot_range["pitch"][1],
            self.cfg.grasp_rot_range["yaw"][1],
        ], device=self.device)
        euler = (rot_action + 1.0) / 2.0 * (rot_max - rot_min) + rot_min

        # Convert Euler angles to quaternion
        quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])

        # Store grasp pose (in robot base frame)
        self.grasp_pose_target[mask, :3] = pos
        self.grasp_pose_target[mask, 3:7] = quat

        # Compute approach pose (offset along gripper's -Z axis)
        self.approach_pose_target[mask] = self._compute_approach_pose(
            self.grasp_pose_target[mask],
            offset=self.cfg.approach_offset
        )

        # Compute lift pose (grasp pose + world Z offset)
        self.lift_pose_target[mask] = self._compute_lift_pose(
            self.grasp_pose_target[mask],
            height=self.cfg.lift_height
        )

    def _compute_approach_pose(self, grasp_pose: torch.Tensor, offset: float) -> torch.Tensor:
        """Compute approach pose offset from grasp pose along gripper's -Z axis.

        Args:
            grasp_pose: Grasp pose tensor of shape (N, 7) [pos + quat]
            offset: Distance to offset for approach

        Returns:
            Approach pose tensor of shape (N, 7)
        """
        approach_pose = grasp_pose.clone()

        # Approach direction: local -Z axis of gripper (approach from above)
        local_z = torch.tensor([[0.0, 0.0, -1.0]], device=self.device).expand(grasp_pose.shape[0], -1)
        quat = grasp_pose[:, 3:7]

        # Rotate local Z axis by gripper orientation
        approach_dir = quat_rotate(quat, local_z)

        # Offset position along approach direction
        approach_pose[:, :3] = grasp_pose[:, :3] + offset * approach_dir

        return approach_pose

    def _compute_lift_pose(self, grasp_pose: torch.Tensor, height: float) -> torch.Tensor:
        """Compute lift pose (grasp pose + world Z offset).

        Args:
            grasp_pose: Grasp pose tensor of shape (N, 7)
            height: Height to lift

        Returns:
            Lift pose tensor of shape (N, 7)
        """
        lift_pose = grasp_pose.clone()
        lift_pose[:, 2] += height  # Add to Z component
        return lift_pose

    def _transition_to_approach(self, mask: torch.Tensor):
        """Transition environments from OBSERVE to APPROACH phase."""
        self.phase[mask] = GraspPhase.APPROACH
        self.phase_step_count[mask] = 0
        self.gripper_closed[mask] = False

    def _execute_scripted_motion(self):
        """Execute one step of IK-based motion for current phase."""
        robot = self.scene["robot"]

        # Get current target based on phase
        target_pose = self._get_current_target_pose()

        # Set IK command
        self.ik_controller.set_command(target_pose)

        # Compute IK
        jacobian = robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids
        ]

        # Get current EE pose in world frame
        ee_pose_w = robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], :7]

        # Get current joint positions
        joint_pos = robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]

        # Get robot root pose in world frame
        root_pose_w = robot.data.root_state_w[:, :7]

        # Transform EE pose to robot base frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7],
            ee_pose_w[:, :3], ee_pose_w[:, 3:7]
        )

        # Compute desired joint positions via IK
        joint_pos_des = self.ik_controller.compute(
            ee_pos_b, ee_quat_b, jacobian, joint_pos
        )

        # Get gripper command based on phase
        gripper_cmd = self._get_gripper_command()

        # Apply joint commands
        robot.set_joint_position_target(
            joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids
        )
        robot.set_joint_position_target(
            gripper_cmd, joint_ids=self.gripper_entity_cfg.joint_ids
        )

    def _get_current_target_pose(self) -> torch.Tensor:
        """Get target pose based on current phase.

        Returns:
            Target pose tensor of shape (num_envs, 7)
        """
        target = torch.zeros((self.num_envs, 7), device=self.device)

        # APPROACH phase: move to approach pose
        approach_mask = self.phase == GraspPhase.APPROACH
        target[approach_mask] = self.approach_pose_target[approach_mask]

        # GRASP phase: move to grasp pose
        grasp_mask = self.phase == GraspPhase.GRASP
        target[grasp_mask] = self.grasp_pose_target[grasp_mask]

        # LIFT phase: move to lift pose
        lift_mask = self.phase == GraspPhase.LIFT
        target[lift_mask] = self.lift_pose_target[lift_mask]

        # EVALUATE phase: maintain lift pose
        evaluate_mask = self.phase == GraspPhase.EVALUATE
        target[evaluate_mask] = self.lift_pose_target[evaluate_mask]

        # OBSERVE phase: maintain current pose (shouldn't reach here during execution)
        observe_mask = self.phase == GraspPhase.OBSERVE
        if observe_mask.any():
            ee_frame = self.scene["ee_frame"]
            ee_pos = ee_frame.data.target_pos_source[:, 0, :]
            ee_quat = ee_frame.data.target_quat_source[:, 0, :]
            target[observe_mask, :3] = ee_pos[observe_mask]
            target[observe_mask, 3:7] = ee_quat[observe_mask]

        return target

    def _get_gripper_command(self) -> torch.Tensor:
        """Get gripper command based on phase.

        Returns:
            Gripper joint position targets of shape (num_envs, 2)
        """
        gripper_open = 0.0425  # Open position
        gripper_close = 0.0    # Closed position

        cmd = torch.full(
            (self.num_envs, len(self.gripper_entity_cfg.joint_ids)),
            gripper_open,
            device=self.device
        )

        # Close gripper during GRASP, LIFT, and EVALUATE phases
        close_mask = (
            (self.phase == GraspPhase.GRASP) |
            (self.phase == GraspPhase.LIFT) |
            (self.phase == GraspPhase.EVALUATE)
        )
        cmd[close_mask] = gripper_close
        self.gripper_closed[close_mask] = True

        return cmd

    def _physics_step(self):
        """Advance physics simulation by one step."""
        self.sim.step(render=False)
        self.scene.update(self.cfg.sim.dt)

    def _update_phase_transitions(self):
        """Check and update phase transitions based on conditions."""
        # Get current EE pose
        ee_frame = self.scene["ee_frame"]
        ee_pos = ee_frame.data.target_pos_source[:, 0, :]
        ee_quat = ee_frame.data.target_quat_source[:, 0, :]
        ee_pose = torch.cat([ee_pos, ee_quat], dim=-1)

        # Get EE velocity for stability check
        robot = self.scene["robot"]
        ee_vel = robot.data.body_lin_vel_w[:, self.robot_entity_cfg.body_ids[0], :]
        ee_speed = torch.norm(ee_vel, dim=-1)

        # APPROACH -> GRASP
        approach_mask = self.phase == GraspPhase.APPROACH
        if approach_mask.any():
            reached = self._pose_reached(ee_pose, self.approach_pose_target)
            stable = ee_speed < self.cfg.velocity_threshold
            timeout = self.phase_step_count > self.cfg.approach_timeout
            transition = approach_mask & (reached & stable | timeout)
            self.phase[transition] = GraspPhase.GRASP
            self.phase_step_count[transition] = 0

        # GRASP -> LIFT
        grasp_mask = self.phase == GraspPhase.GRASP
        if grasp_mask.any():
            reached = self._pose_reached(ee_pose, self.grasp_pose_target)
            timeout = self.phase_step_count > self.cfg.grasp_timeout
            transition = grasp_mask & (reached | timeout)
            self.phase[transition] = GraspPhase.LIFT
            self.phase_step_count[transition] = 0

        # LIFT -> EVALUATE
        lift_mask = self.phase == GraspPhase.LIFT
        if lift_mask.any():
            reached = self._pose_reached(ee_pose, self.lift_pose_target)
            timeout = self.phase_step_count > self.cfg.lift_timeout
            transition = lift_mask & (reached | timeout)
            self.phase[transition] = GraspPhase.EVALUATE
            self.phase_step_count[transition] = 0

        # Increment phase step count
        self.phase_step_count += 1

    def _pose_reached(
        self, current: torch.Tensor, target: torch.Tensor, threshold: float = None
    ) -> torch.Tensor:
        """Check if current pose is close enough to target.

        Args:
            current: Current pose (num_envs, 7)
            target: Target pose (num_envs, 7)
            threshold: Position threshold (default: self.cfg.position_threshold)

        Returns:
            Boolean mask of shape (num_envs,)
        """
        if threshold is None:
            threshold = self.cfg.position_threshold

        # Check position distance
        pos_dist = torch.norm(current[:, :3] - target[:, :3], dim=-1)
        return pos_dist < threshold

    def _get_observations(self) -> dict:
        """Compute observations for all environments.

        Returns:
            Dictionary of observations.
        """
        # Use the observation manager from base class
        return self.observation_manager.compute()

    def _compute_rewards(self) -> torch.Tensor:
        """Compute sparse reward only during EVALUATE phase.

        Returns:
            Reward tensor of shape (num_envs,)
        """
        rewards = torch.zeros(self.num_envs, device=self.device)

        evaluate_mask = self.phase == GraspPhase.EVALUATE
        if not evaluate_mask.any():
            return rewards

        # Check if object is lifted
        obj = self.scene["object"]
        obj_height = obj.data.root_pos_w[:, 2]

        # Success if object is above threshold height
        success = obj_height > self.cfg.success_height
        rewards[evaluate_mask & success] = 10.0

        # Optional: partial reward for object height
        # rewards[evaluate_mask] = torch.clamp(obj_height[evaluate_mask] / self.cfg.success_height, 0, 1) * 5.0

        return rewards

    def _check_termination(self) -> torch.Tensor:
        """Check termination conditions.

        Returns:
            Boolean tensor of shape (num_envs,) indicating terminated environments.
        """
        # Terminate if in EVALUATE phase
        terminated = self.phase == GraspPhase.EVALUATE

        # Also terminate if object dropped
        obj = self.scene["object"]
        obj_height = obj.data.root_pos_w[:, 2]
        object_dropped = obj_height < -0.05
        terminated = terminated | object_dropped

        return terminated

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specified environments.

        Args:
            env_ids: Indices of environments to reset.
        """
        super()._reset_idx(env_ids)

        # Reset phase tracking
        self.phase[env_ids] = GraspPhase.OBSERVE
        self.phase_step_count[env_ids] = 0

        # Reset grasp targets
        self.grasp_pose_target[env_ids] = 0
        self.approach_pose_target[env_ids] = 0
        self.lift_pose_target[env_ids] = 0

        # Reset gripper state
        self.gripper_closed[env_ids] = False

        # Reset episode rewards
        self.episode_rewards[env_ids] = 0

        # Reset IK controller
        self.ik_controller.reset(env_ids)
