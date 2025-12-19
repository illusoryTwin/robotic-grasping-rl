# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10 robot with Hand-E gripper configuration."""

import os
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg


"""UR10 manipulator with a gripper and camera mount.

This configuration defines a UR10e manipulator with:
- 6-DOF arm with implicit PD control
- Robotiq Hand-E parallel gripper
- Camera mount on wrist
"""
UR10_WITH_GRIPPER_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=os.path.join(os.path.dirname(__file__), "..", "objects", "ur10e_with_hand_e_and_camera_mount.usd"),
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

