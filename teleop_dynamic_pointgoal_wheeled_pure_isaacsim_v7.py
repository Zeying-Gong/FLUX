import argparse
import os

parser = argparse.ArgumentParser("Teleoperation v7")
parser.add_argument("--scene_dir",   type=str, default="/workspace/FLUX/assets/scenes/cluttered_easy")
parser.add_argument("--scene_index", type=int, default=1)
args_cli = parser.parse_args()

from isaacsim import SimulationApp

CUSTOM_APP_PATH = os.path.join(
    os.environ["EXP_PATH"],
    "isaacsim.exp.action_and_event_data_generation.base.kit"
)

simulation_app = SimulationApp(
    launch_config={
        "renderer": "RayTracedLighting",
        "headless": False,
        "width": 1920,
        "height": 1080,
        "enable_cameras": True,
    },
    experience=CUSTOM_APP_PATH,
)

import json
import omni
import omni.usd
import omni.kit.app
import omni.kit.commands
import omni.timeline
import carb
import omni.replicator.core as rep

def set_simulation_settings():
    rep.settings.carb_settings("/omni/replicator/backend/writeThreads", 16)
    print("[INIT] Simulation settings applied")

def main():

    ## Scene loading and setup

    scene_entries = sorted(os.listdir(args_cli.scene_dir))
    scene_name    = scene_entries[args_cli.scene_index]
    scene_path    = os.path.join(args_cli.scene_dir, scene_name) + "/"

    from utils_tasks.basic_utils import find_usd_path
    usd_path, _ = find_usd_path(scene_path, 'pointgoal')
    print(f"[INFO] Scene USD: {usd_path}")

    episode_json_path = os.path.join(scene_path, "episode_0.json")
    if not os.path.exists(episode_json_path):
        print(f"[WARN] episode JSON not found: {episode_json_path}, will use default commands")
        episode_json_path = None
    else:
        print(f"[INFO] Episode JSON: {episode_json_path}")

    set_simulation_settings()
    
    ## Navmesh baking is required for pathfinding and navigation. It can be time-consuming, so we do it once at the start.

    from isaacsim.core.utils.stage import open_stage
    import omni.anim.navigation.core as nav
    from pxr import Sdf

    print(f"[NAVMESH] Opening scene: {usd_path}")
    open_stage(usd_path)
    for _ in range(30):
        simulation_app.update()
    while omni.usd.get_context().get_stage_loading_status()[1] > 0:
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/World/Scene").IsValid():
        stage.DefinePrim("/World/Scene", "Xform")

    omni.kit.commands.execute(
        'CreateNavMeshVolumeCommand',
        parent_prim_path=Sdf.Path("/World/Scene"),
        usd_context_name="",
        layer=None,
    )
    for _ in range(30):
        simulation_app.update()

    _inav = nav.acquire_interface()
    print("[NAVMESH] start_navmesh_baking_and_wait()...")
    _inav.start_navmesh_baking_and_wait()
    navmesh = _inav.get_navmesh()
    if navmesh is None:
        print("[NAVMESH] FAILED")
    else:
        pt = navmesh.query_random_point()
        print(f"[NAVMESH] SUCCESS test_point={pt}")

    ## people simulation setup. This is where the episode JSON commands are loaded and executed, and pedestrians are spawned and controlled.

    from people_simulation import PeopleSimulation

    people_sim = None
    if episode_json_path:
        people_sim = PeopleSimulation(episode_json_path)
        print("[PEOPLE] Running PeopleSimulation.setup_async()...")
        
        import asyncio
        task = asyncio.ensure_future(people_sim.setup_async())
        while not task.done():
            simulation_app.update()
        
        if people_sim.is_ready:
            print("[PEOPLE] ✓ People simulation ready")
        else:
            print("[PEOPLE] ✗ People setup failed")

    timeline = omni.timeline.get_timeline_interface()
    # timeline.play()

    print("\n[MAIN] ========== Main loop start ==========")
    print("[MAIN] Observation only — do pedestrians MOVE with episode JSON commands?\n")

    step_count = 0
    while simulation_app.is_running():
        simulation_app.update()
        step_count += 1
        if step_count % 300 == 0:
            print(f"[STEP {step_count}] running...")

    print("[MAIN] Shutting down...")
    timeline.stop()
    simulation_app.close()

if __name__ == "__main__":
    main()
    