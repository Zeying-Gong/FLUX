"""Image-goal navigation (native IsaacSim). Refactored to use utils_tasks.*"""
from utils_tasks.sim_launcher import (
    make_common_parser, launch_sim, register_signal_handlers,
    start_planning_thread, stop_event,
    EpisodeRunner, load_episode_from_npy,
    build_navigation_metrics, print_episode_metrics,
)

parser = make_common_parser(
    "Image-Goal navigation (native IsaacSim)",
    default_scene_dir="/workspace/FLUX/assets/n1_eval_scenes/cluttered_easy",
)
args_cli = parser.parse_args()

simulation_app = launch_sim(headless=True)

import os, sys
import numpy as np

from utils_tasks.basic_utils         import find_usd_path, write_metrics
from utils_tasks.visualization_utils import VisualizationManager
from utils_tasks.client_utils        import navigator_reset, imagegoal_step
from utils_tasks.evaluator           import IsaacSimEvaluator
from wheeled_robots.controllers.differential_controller import DifferentialController
from configs.robots import DINGO_WHEEL_RADIUS, DINGO_WHEEL_BASE


def main():
    task_name = "imagegoal"
    scene_list = sorted(os.listdir(args_cli.scene_dir))
    scene_name = scene_list[args_cli.scene_index]
    scene_path = os.path.join(args_cli.scene_dir, scene_name) + "/"
    usd_path, npy_path = find_usd_path(scene_path, "imagegoal")

    samples = np.load(npy_path)
    num_episodes = min(args_cli.num_episodes, len(samples))
    print(f"[INFO] {len(samples)} samples in npy, running {num_episodes} episodes.")

    evaluator = IsaacSimEvaluator(simulation_app, usd_path,
                                  scene_scale=args_cli.scene_scale)
    evaluator.enable_goal_camera()
    evaluator.setup()

    register_signal_handlers(
        get_env=lambda: None,
        get_simulation_app=lambda: simulation_app,
        stop_event=stop_event,
    )

    print("[INFO] Preheating simulation...")
    for _ in range(10):
        evaluator.world.step(render=False)
        simulation_app.update()

    algo = navigator_reset(
        evaluator.cam_intrinsic, batch_size=1,
        stop_threshold=args_cli.stop_threshold, port=args_cli.port,
    )
    if algo == "fallback_algo":
        print(f"[ERROR] Algorithm server not found. Start: python server.py --port {args_cli.port}")
        evaluator.close()
        sys.exit(1)
    print(f"[INFO] Algorithm connected, algo={algo}")

    planner_thread = start_planning_thread(
        plan_fn=imagegoal_step, port=args_cli.port,
        speed=args_cli.speed, goal_required=True,
    )

    diff_ctrl = DifferentialController(
        name="dingo_ctrl",
        wheel_radius=DINGO_WHEEL_RADIUS, wheel_base=DINGO_WHEEL_BASE,
    )

    save_dir = "./metrics/%s_%s_%s/%s/" % (
        task_name, algo, args_cli.scene_dir.split("/")[-1], scene_name,
    )
    os.makedirs(save_dir, exist_ok=True)
    evaluation_metrics: list[dict] = []
    vis_manager = VisualizationManager(history_size=5)

    for ep_idx in range(num_episodes):
        print(f"\n{'='*60}\n  Episode {ep_idx}/{num_episodes}  scene={scene_name}\n{'='*60}")

        start_pos, start_yaw, goal_world = load_episode_from_npy(npy_path, ep_idx)

        # snapshot the goal image at the goal pose
        evaluator.place_goal_camera(goal_world, yaw=start_yaw)
        goal_image       = evaluator.get_goal_rgb()
        goal_image_batch = goal_image[None]

        evaluator.reset_robot(start_pos, start_yaw)
        navigator_reset(env_id=0, port=args_cli.port)
        initial_dist = float(np.linalg.norm(start_pos - goal_world))

        runner = EpisodeRunner(
            evaluator, vis_manager, diff_ctrl,
            max_steps=args_cli.max_steps, save_dir=save_dir,
            ep_idx=ep_idx, algo=algo, task_name=task_name, scene_name=scene_name,
            goal_required=True,
        )
        runner.reset_vis_from_current_pose()
        runner.prepare_planning()

        while not runner.done and runner.step_count < args_cli.max_steps:
            obs = runner.observe()

            runner.push_plan(obs, goal_image_batch)

            traj_w, all_traj_w, all_vals = runner.pop_plan()
            joint_vels, v, w = runner.tick_mpc(traj_w, obs)

            dist_to_target = float(np.linalg.norm(obs.cam_pos[:2] - goal_world))

            if traj_w is not None:
                rgb_with_goal = np.concatenate(
                    (obs.rgb[0], goal_image), axis=1
                )
                runner.write_vis_frame(
                    obs, traj_w, all_traj_w, all_vals, v, w,
                    goal_position=goal_world,
                    dist_label_value=dist_to_target,
                    rgb_override=rgb_with_goal,
                    extra_vis_kwargs={"people_positions": None,
                                      "people_positions_dict": None},
                )

            runner.step_world(joint_vels, has_plan=(traj_w is not None))
            runner.record_position(obs)
            runner.check_termination(dist_to_target)

        runner.close_video()

        final_dist = float(np.linalg.norm(runner.last_cam_pos[:2] - goal_world))
        ep_metrics = build_navigation_metrics(
            ep_idx=ep_idx, success=runner.success,
            initial_dist=initial_dist, final_dist=final_dist,
            step_count=runner.step_count, step_dt=evaluator.step_dt,
            traj_length=runner.traj_length,
        )
        evaluation_metrics.append(ep_metrics)
        print_episode_metrics(ep_idx, ep_metrics)
        write_metrics(evaluation_metrics, save_dir + "metric.csv")

    print(f"\n[INFO] All {num_episodes} episodes done. Metrics → {save_dir}metric.csv")
    stop_event.set()
    planner_thread.join(timeout=5)
    evaluator.close()


if __name__ == "__main__":
    main()