"""
Isaac Sim evaluation script utilities.

This module intentionally keeps behavior identical to the eval_*.py scripts:
- cleanup order matches existing scripts to avoid camera/replicator shutdown issues
- lighting setup is opt-in (call setup_all_lighting when needed)
- signal handlers call cleanup and exit
"""

from __future__ import annotations

from typing import Optional, Callable, Any


def cleanup_simulation(
    *,
    env: Optional[Any] = None,
    simulation_app: Optional[Any] = None,
    stop_event: Optional[Any] = None,
    planning_thread_obj: Optional[Any] = None,
    fps_writer: Optional[list] = None,
) -> None:
    """Best-effort cleanup for Isaac Sim + env + cameras + writers."""
    print("[INFO] Starting cleanup...")
    try:
        # 1) stop planning thread
        try:
            if stop_event is not None:
                stop_event.set()
            if planning_thread_obj is not None and getattr(planning_thread_obj, "is_alive", lambda: False)():
                print("  Stopping planning thread...")
                planning_thread_obj.join(timeout=3)
        except Exception as e:
            print(f"  Warning: planning thread cleanup failed: {e}")

        # 2) close video writers
        if fps_writer is not None:
            print("  Closing video writers...")
            for writer in fps_writer:
                try:
                    writer.close()
                except Exception:
                    pass

        # 3) detach camera sensors annotators to avoid __del__ issues
        if env is not None:
            print("  Detaching camera sensors...")
            try:
                unwrapped_env = env.unwrapped if hasattr(env, "unwrapped") else env
                if hasattr(unwrapped_env, "scene") and hasattr(unwrapped_env.scene, "sensors"):
                    camera_sensor = unwrapped_env.scene.sensors.get("camera_sensor")
                    if camera_sensor is not None:
                        if hasattr(camera_sensor, "_annotators"):
                            for annotator in camera_sensor._annotators:
                                try:
                                    annotator.detach()
                                except Exception:
                                    pass
                        camera_sensor._annotators = []
                        camera_sensor._sensor_prims = []
            except Exception as e:
                print(f"  Warning: Camera cleanup failed: {e}")

        # 4) close env
        if env is not None:
            print("  Closing environment...")
            try:
                unwrapped_env = env.unwrapped if hasattr(env, "unwrapped") else env
                if hasattr(unwrapped_env, "scene"):
                    try:
                        unwrapped_env.scene.reset()
                    except Exception:
                        pass
                env.close()
            except Exception as e:
                print(f"  Warning: Error closing env: {e}")

        # 5) clear simulation context
        print("  Clearing simulation context...")
        try:
            from isaaclab.sim import SimulationContext

            sim_context = SimulationContext.instance()
            if sim_context is not None:
                sim_context.clear_all_callbacks()
                SimulationContext.clear_instance()
        except Exception as e:
            print(f"  Warning: SimulationContext cleanup failed: {e}")

        # 6) clean Isaac Sim components
        if simulation_app is not None:
            print("  Cleaning Isaac Sim components...")
            try:
                import omni.replicator.core as rep

                rep.orchestrator.stop()
            except Exception as e:
                print(f"  Warning: Replicator cleanup failed: {e}")

            try:
                import carb

                settings = carb.settings.get_settings()
                settings.set("/exts/omni.syntheticdata/enabled", False)
                simulation_app.update()
            except Exception as e:
                print(f"  Warning: SyntheticData disable failed: {e}")

            try:
                import omni.usd

                context = omni.usd.get_context()
                if context:
                    context.close_stage()
                    for _ in range(3):
                        simulation_app.update()
            except Exception as e:
                print(f"  Warning: Stage cleanup failed: {e}")

            import gc

            gc.collect()

            print("  Closing simulation app...")
            try:
                simulation_app.close()
            except Exception as e:
                print(f"  Warning: Simulation close error: {e}")

        # 7) clear CUDA cache
        try:
            import torch

            if torch.cuda.is_available():
                print("  Clearing CUDA cache...")
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
        except Exception as e:
            print(f"  Warning: CUDA cleanup failed: {e}")

        print("[INFO] Cleanup complete")
    except Exception as e:
        print(f"[ERROR] Cleanup failed: {e}")
        import traceback

        traceback.print_exc()


def register_signal_handlers(
    *,
    get_env: Callable[[], Optional[Any]],
    get_simulation_app: Callable[[], Optional[Any]],
    stop_event: Optional[Any] = None,
    get_planning_thread_obj: Optional[Callable[[], Optional[Any]]] = None,
    get_fps_writer: Optional[Callable[[], Optional[list]]] = None,
) -> None:
    """Register SIGINT/SIGTERM handlers that run cleanup then exit."""
    import signal
    import sys

    def _handler(sig, frame):
        print("\n" + "=" * 50)
        print("Received interrupt signal, shutting down...")
        print("=" * 50)
        cleanup_simulation(
            env=get_env(),
            simulation_app=get_simulation_app(),
            stop_event=stop_event,
            planning_thread_obj=get_planning_thread_obj() if get_planning_thread_obj else None,
            fps_writer=get_fps_writer() if get_fps_writer else None,
        )
        print("Exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def setup_all_lighting(env: Any) -> None:
    import omni.usd
    import carb
    from pxr import UsdLux, Gf

    settings = carb.settings.get_settings()
    stage = omni.usd.get_context().get_stage()

    # 1) Viewport Camera Light
    settings.set("/rtx/useViewLightingMode", True)
    settings.set("/rtx/viewLightingMode", 1)

    # 2) ambient and exposure
    settings.set("/rtx/sceneDb/ambientLightIntensity", 3.0)
    settings.set("/rtx/post/tonemap/enable", True)
    settings.set("/rtx/post/tonemap/exposure", 1.0)

    # 3) add per-robot camera light
    camera_sensor = env.unwrapped.scene.sensors["camera_sensor"]
    for i, camera_prim in enumerate(camera_sensor._sensor_prims):
        camera_path = str(camera_prim.GetPath())
        parent_path = "/".join(camera_path.split("/")[:-1])
        light_path = f"{parent_path}/CameraLight"

        if not stage.GetPrimAtPath(light_path):
            sphere_light = UsdLux.SphereLight.Define(stage, light_path)
            sphere_light.CreateIntensityAttr(1000.0)
            sphere_light.CreateRadiusAttr(0.1)
            sphere_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
            print(f"[INFO] ✓ Added light to robot camera {i}")

    print("[INFO] ✓ All lighting setup complete")

