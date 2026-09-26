"""Social navigation (native IsaacSim). Refactored to use utils_tasks.*"""
from utils_tasks.sim_launcher import make_common_parser, launch_sim, register_signal_handlers, start_planning_thread, stop_event, \
    EpisodeRunner, load_episode, build_navigation_metrics, print_episode_metrics
args_cli = make_common_parser("Social navigation (native IsaacSim)").parse_args()

simulation_app = launch_sim(headless=False)

import os, sys
import numpy as np

from utils_tasks.basic_utils         import find_usd_path, write_metrics
from utils_tasks.visualization_utils import VisualizationManager
from utils_tasks.client_utils        import navigator_reset, pointgoal_step
from utils_tasks.evaluator           import IsaacSimEvaluator
from utils_tasks.people_runtime      import (
    bake_navmesh, setup_people_episode, open_character_barrier,
)
from socialnav_metrics            import SocialMetricsTracker, get_people_positions
from wheeled_robots.controllers.differential_controller import DifferentialController
from configs.robots import DINGO_WHEEL_RADIUS, DINGO_WHEEL_BASE


def main():
    task_name = "socialnav"
    scene_list = sorted(os.listdir(args_cli.scene_dir))
    scene_name = scene_list[args_cli.scene_index]
    scene_path = os.path.join(args_cli.scene_dir, scene_name) + "/"
    usd_path, _ = find_usd_path(scene_path, "pointgoal")

    episode_files = []
    for ep_id in range(args_cli.num_episodes):
        p = os.path.join(scene_path, f"episode_{ep_id}.json")
        if not os.path.exists(p):
            raise RuntimeError(f"Missing episode file: {p}")
        episode_files.append(p)
    print(f"[INFO] {len(episode_files)} episode files validated.")

    evaluator = IsaacSimEvaluator(simulation_app, usd_path,
                                  scene_scale=args_cli.scene_scale)
    evaluator.setup()

    register_signal_handlers(
        get_env=lambda: None,
        get_simulation_app=lambda: simulation_app,
        stop_event=stop_event,
    )

    bake_navmesh(simulation_app)

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
        plan_fn=pointgoal_step, port=args_cli.port,
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

    for ep_idx in range(args_cli.num_episodes):
        print(f"\n{'='*60}\n  Episode {ep_idx}/{args_cli.num_episodes}  scene={scene_name}\n{'='*60}")

        ep_path = episode_files[ep_idx]
        start_pos, start_yaw, goal_world = load_episode(ep_path)
        evaluator.reset_robot(start_pos, start_yaw)
        navigator_reset(env_id=0, port=args_cli.port)
        initial_dist = float(np.linalg.norm(start_pos - goal_world))

        # people (static obstacles in this task)
        people_sim = setup_people_episode(evaluator, ep_path, dynamic_target=False)
        if people_sim is None:
            continue

        runner = EpisodeRunner(
            evaluator, vis_manager, diff_ctrl,
            max_steps=args_cli.max_steps, save_dir=save_dir,
            ep_idx=ep_idx, algo=algo, task_name=task_name, scene_name=scene_name,
            goal_required=True,
        )
        runner.reset_vis_from_current_pose()
        runner.prepare_planning()

        social_tracker = SocialMetricsTracker()
        open_character_barrier(ep_idx)

        while not runner.done and runner.step_count < args_cli.max_steps:
            obs = runner.observe()

            rel_vec = np.array([goal_world[0] - obs.cam_pos[0],
                                goal_world[1] - obs.cam_pos[1], 0.0])
            goal_cam = (obs.cam_rot.T @ rel_vec)[:2][None]
            runner.push_plan(obs, goal_cam)

            traj_w, all_traj_w, all_vals = runner.pop_plan()
            joint_vels, v, w = runner.tick_mpc(traj_w, obs)

            # social metrics
            people_positions, people_char_paths, pos_flag = get_people_positions()
            if pos_flag:
                social_tracker.update(obs.cam_pos, people_positions, evaluator.step_dt)
            else:
                people_positions = people_char_paths = []

            dist_to_target = float(np.linalg.norm(obs.cam_pos[:2] - goal_world))

            if traj_w is not None:
                runner.write_vis_frame(
                    obs, traj_w, all_traj_w, all_vals, v, w,
                    goal_position=goal_world,
                    dist_label_value=dist_to_target,
                    extra_vis_kwargs={
                        "people_positions": people_positions,
                        "people_positions_dict": people_char_paths,
                    },
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
            soc=social_tracker.get_metrics(),
        )
        evaluation_metrics.append(ep_metrics)
        print_episode_metrics(ep_idx, ep_metrics)
        write_metrics(evaluation_metrics, save_dir + "metric.csv")

    print(f"\n[INFO] All {args_cli.num_episodes} episodes done. Metrics → {save_dir}metric.csv")
    stop_event.set()
    planner_thread.join(timeout=5)
    evaluator.close()


if __name__ == "__main__":
    main()