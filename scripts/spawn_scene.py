"""Script to spawn and visualize the UR10 manipulator with objects.

This script creates a scene with the UR10e robot and all three objects
(tin can, tetra pak, chips bag) for visualization and inspection.

Usage:
    python scripts/spawn_scene.py
    python scripts/spawn_scene.py --num_envs 3  # One env per object
"""

import argparse
import sys
from pathlib import Path

# Add the manipulation_rl_new directory to Python path
MANIP_RL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MANIP_RL_DIR))

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Spawn UR10 manipulator with objects")
parser.add_argument("--num_envs", type=int, default=3, help="Number of environments (default: 3, one per object)")
AppLauncher.add_app_launcher_args(parser)

# Parse args
args_cli = parser.parse_args()

# Launch omniverse app (with rendering enabled for visualization)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching the app
import torch
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

# Import robot configuration from custom assets (UR10 with Hand-E gripper)
import sys
sys.path.insert(0, str(MANIP_RL_DIR))
from assets import UR10_WITH_GRIPPER_CFG

# Objects directory
OBJECTS_DIR = os.path.join(str(Path.home()), "Workspace/manipulation_rl_new/objects")


@configclass
class SpawnSceneCfg(InteractiveSceneCfg):
    """Configuration for the visualization scene with robot and all objects."""

    # UR10 with Hand-E gripper
    robot = UR10_WITH_GRIPPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
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
                "hande_left_finger_joint": 0.0,  # Gripper open
                "hande_right_finger_joint": 0.0,  # Gripper open
            },
        ),
    )

    # Tin Can
    tin_can = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TinCan",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5, -0.2, 0.055],
            rot=[0.7071, 0.7071, 0, 0],  # Stand upright
        ),
        spawn=UsdFileCfg(
            usd_path=os.path.join(OBJECTS_DIR, "tin-can.usd"),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    # Tetra Pak
    tetra_pak = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TetraPak",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5, 0.0, 0.1],
            rot=[1, 0, 0, 0],
        ),
        spawn=UsdFileCfg(
            usd_path=os.path.join(OBJECTS_DIR, "tetra-pak-carton.usd"),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    # Chips Bag
    chips_bag = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ChipsBag",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5, 0.2, 0.15],
            rot=[1, 0, 0, 0],
        ),
        spawn=UsdFileCfg(
            usd_path=os.path.join(OBJECTS_DIR, "chips-bag.usd"),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    # Frame transformer for end-effector tracking (Hand-E gripper)
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=True,  # Show EE frame visualization
        visualizer_cfg=FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/FrameTransformer",
            markers={
                "frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.1, 0.1, 0.1),  # Smaller frame markers
                ),
            },
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/hande_end",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
        ],
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


def main():
    """Main function to spawn and visualize the scene."""
    
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    
    # Set camera view
    sim.set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.5, 0.0, 0.5])
    
    # Create scene configuration
    scene_cfg = SpawnSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    
    # Create scene
    scene = InteractiveScene(scene_cfg)
    
    # Play simulation
    sim.reset()
    
    print("[INFO] Scene spawned successfully!")
    print(f"[INFO] Number of environments: {args_cli.num_envs}")
    print("[INFO] Objects in scene:")
    print("       - UR10e with Robotiq 2F-140 gripper")
    print("       - Tin Can")
    print("       - Tetra Pak")
    print("       - Chips Bag")
    print("[INFO] Press Ctrl+C to exit")
    print("-" * 60)
    
    # Run simulation loop
    step = 0
    try:
        while simulation_app.is_running():
            # Step simulation
            sim.step()
            
            # Update scene
            scene.update(dt=sim_cfg.dt)
            
            # Print robot state periodically
            if step % 500 == 0:
                robot = scene["robot"]
                ee_frame = scene["ee_frame"]
                
                # Get end-effector position
                ee_pos = ee_frame.data.target_pos_w[0, 0, :]
                
                # Get gripper joint state (Hand-E)
                finger_joint_idx = robot.find_joints("hande_left_finger_joint")[0][0]
                gripper_pos = robot.data.joint_pos[0, finger_joint_idx].item()
                
                print(f"[Step {step}] EE Position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}], "
                      f"Gripper: {gripper_pos:.3f}")
            
            step += 1
            
    except KeyboardInterrupt:
        print("\n[INFO] Visualization stopped by user")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()

