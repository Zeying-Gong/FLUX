from __future__ import annotations

from typing import Optional, Any


def launch_simulation_app(
    *,
    headless: bool,
    enable_cameras: bool,
    experience: Optional[str] = None,
    multi_gpu: bool = False,
    num_gpus: int = 1,
    gpu_id: Optional[int] = None,
) -> Any:
    """Create IsaacLab AppLauncher and return simulation app.

    Notes:
    - Keep Isaac Sim imports AFTER this is called in eval scripts.
    - If multi_gpu is False, gpu_id selects a CUDA device string (cuda:{gpu_id}).
    """
    from isaaclab.app import AppLauncher

    launcher_kwargs = {
        "headless": headless,
        "enable_cameras": enable_cameras,
    }
    if experience is not None:
        launcher_kwargs["experience"] = experience

    if multi_gpu:
        launcher_kwargs["multi_gpu"] = True
    else:
        if gpu_id is not None:
            launcher_kwargs["device"] = f"cuda:{gpu_id}"

    app_launcher = AppLauncher(**launcher_kwargs)
    simulation_app = app_launcher.app

    if multi_gpu:
        try:
            import carb

            settings = carb.settings.get_settings()
            settings.set("/renderer/multiGpu/enabled", True)
            settings.set("/renderer/multiGpu/maxGpuCount", num_gpus)
            settings.set("/renderer/multiGpu/autoEnable", True)
            print(f"[INFO] Multi-GPU enabled with {num_gpus} GPUs")
        except Exception as e:
            print(f"[WARN] Failed enabling multi-gpu renderer settings: {e}")
    else:
        if gpu_id is not None:
            print(f"[INFO] Single GPU mode: GPU {gpu_id}")

    return simulation_app

