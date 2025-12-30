"""Spawn UR10 manipulator with gripper and table.

Usage:
    python tests/spawn_ur10_gripper.py
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
    finally:
        simulation_app.close()
