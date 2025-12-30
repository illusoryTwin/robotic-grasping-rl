"""Spawn UR10 manipulator with gripper and table.

Usage:
    python tests/spawn_objects.py
"""

import argparse
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn UR10 with gripper")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from assets import UR10_WITH_GRIPPER_CFG


@configclass
class SpawnSceneCfg(InteractiveSceneCfg):
    """Scene with UR10 robot and table."""

    robot = UR10_WITH_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Tin can
    custom_tin_can = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CustomTinCan",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5, -0.3, 0.055],
            rot=[0.7071, 0.7071, 0, 0],
        ),
        spawn=UsdFileCfg(
            usd_path=str(REPO_DIR / "objects" / "tin-can.usd"),
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

    # Cylinder tin can
    tin_can = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TinCan",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.3, -0.3, 0.075],  # Cylinder height/2 + table height
            rot=[1, 0, 0, 0],  # Upright cylinder
        ),
        spawn=sim_utils.CylinderCfg(
            radius=0.033,  # ~33mm radius (typical soup can)
            height=0.08,   # ~80mm height (shorter can)
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)), # (0.5, 0.5, 0.5)),
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

    
    tetra_pak = RigidObjectCfg(
      prim_path="{ENV_REGEX_NS}/TetraPak",
      init_state=RigidObjectCfg.InitialStateCfg(
          pos=[0.7, -0.3, 0.1],
          rot=[1, 0, 0, 0],
      ),
      spawn=UsdFileCfg(
          usd_path=str(REPO_DIR / "objects" / "tetra-pak-carton.usd"),
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

    tetra_pak_custom = RigidObjectCfg(
      prim_path="{ENV_REGEX_NS}/TetraPakCustom",
      init_state=RigidObjectCfg.InitialStateCfg(
          pos=[0.9, -0.3, 0.027],
          rot=[0.707, 0.707, 0, 0],  # Rotated 90° to lie on long side
      ),
      spawn=sim_utils.CuboidCfg(
          size=(0.06, 0.06, 0.18),  # tetra pak proportions
          rigid_props=sim_utils.RigidBodyPropertiesCfg(
              solver_position_iteration_count=16,
              solver_velocity_iteration_count=1,
              max_angular_velocity=50.0,
              max_linear_velocity=50.0,
              max_depenetration_velocity=5.0,
              disable_gravity=False,
          ),
          mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
          collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
          visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.6, 0.4)),
      ),
    )

    chips_bag = RigidObjectCfg(
      prim_path="{ENV_REGEX_NS}/ChipsBag",
      init_state=RigidObjectCfg.InitialStateCfg(
          pos=[0.1, -0.3, 0.15],
          rot=[1, 0, 0, 0],
      ),
      spawn=UsdFileCfg(
          usd_path=str(REPO_DIR / "objects" / "chips-bag.usd"),
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

def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    sim.set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.5, 0.0, 0.5])

    scene = InteractiveScene(SpawnSceneCfg(num_envs=1, env_spacing=2.5))
    sim.reset()

    print("[INFO] UR10 with gripper spawned. Press Ctrl+C to exit.")

    while simulation_app.is_running():
        sim.step()
        scene.update(dt=0.01)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
