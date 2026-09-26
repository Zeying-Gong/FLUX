"""Per-episode PeopleSimulation lifecycle, used by socialnav / dynpointgoal /
dynnogoal. Handles the "6-step dance":

    1. PeopleSimulation(...) + await setup_async()
    2. CLOSE character barrier (so characters freeze during warmup)
    3. world.reset() + robot.initialize() (re-bind physics handles)
    4. timeline.set_current_time(0.0); timeline.play()
    5. OPEN character barrier (characters and robot start together)

Return None if PeopleSimulation failed to become ready (caller should skip ep).
"""
from __future__ import annotations
import asyncio
import carb
import omni.timeline

from people_simulation import PeopleSimulation


"""NavMesh baking helper.

Appended as its own module to avoid overwriting the existing `sim_utils.py`
(which already provides `register_signal_handlers`). Task scripts import:

    from utils_tasks.navmesh_utils import bake_navmesh

You can alternatively paste the `bake_navmesh` body into your existing
`utils_tasks/sim_utils.py` if you prefer keeping sim-related helpers in one
module.
"""

import omni.usd
import omni.kit.commands
from pxr import Sdf


def bake_navmesh(simulation_app, parent_prim_path: str = "/World/Scene"):
    """Create a NavMesh volume under `parent_prim_path` and block until baked."""
    import omni.anim.navigation.core as nav

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(parent_prim_path).IsValid():
        stage.DefinePrim(parent_prim_path, "Xform")

    omni.kit.commands.execute(
        "CreateNavMeshVolumeCommand",
        parent_prim_path=Sdf.Path(parent_prim_path),
        usd_context_name="",
        layer=None,
    )
    for _ in range(30):
        simulation_app.update()

    inav = nav.acquire_interface()
    print("[NavMesh] Baking...")
    inav.start_navmesh_baking_and_wait()
    nm = inav.get_navmesh()
    if nm is None:
        print("[NavMesh] FAILED")
    else:
        print(f"[NavMesh] OK  test_pt={nm.query_random_point()}")
    return nm

"""Bring up PeopleSimulation for one episode.

Returns the PeopleSimulation instance on success, None on failure (caller
should `continue` to the next episode).

NOTE: the character barrier is LEFT CLOSED when this returns. The caller
must call `open_character_barrier()` right before entering the step loop.
"""

def setup_people_episode(evaluator,
                         ep_path: str,
                         dynamic_target: bool = False,
                         wait_steps: int = 300,
                         verbose: bool = True):

    people_sim = PeopleSimulation(ep_path, enable_dynamic_target=dynamic_target)

    task = asyncio.ensure_future(people_sim.setup_async())
    waited = 0
    while not task.done() and waited < wait_steps:
        evaluator.simulation_app.update()
        waited += 1

    if not people_sim.is_ready:
        if verbose:
            print(f"[WARN] People simulation not ready for {ep_path}, skipping.")
        return None

    # Step 2: freeze characters
    _set_barrier(False)

    # Step 3: re-bind physics after timeline churn
    evaluator.world.reset()
    evaluator.robot.initialize()

    # Step 4: rewind + play
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_current_time(0.0)
    timeline.play()

    return people_sim


def open_character_barrier(ep_idx: int | None = None, verbose: bool = True):
    """Let characters start moving. Call right before the step loop."""
    _set_barrier(True)
    if verbose:
        tag = f"ep={ep_idx}: " if ep_idx is not None else ""
        print(f"[BARRIER] {tag}characters can now move")


def close_character_barrier():
    _set_barrier(False)


def _set_barrier(is_open: bool):
    carb.settings.get_settings().set(
        "/exts/people_sim/character_barrier_open", bool(is_open)
    )