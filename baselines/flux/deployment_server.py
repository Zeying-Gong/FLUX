"""
FLUX Deployment Server
Flask-based API for real-robot navigation inference.
"""

import argparse
import json
import os
import time
import math
from datetime import datetime

import numpy as np
import cv2
import imageio
from flask import Flask, jsonify, request
from PIL import Image

from policy_agent import GRPO_Agent
import sys
sys.path.insert(0, '/workspace/FLUX')

from utils_tasks.visualization_utils import VisualizationManager

app = Flask(__name__)

# ============ Global State ============
agent = None
vis_manager = VisualizationManager(history_size=5)
_flux_fps_writer = None
_flux_fps_rgbd_writer = None

# Navigation state
goal_world_pos = [7.0, 5.0, 0.0]  # Default goal in world frame # []
is_first_step = True
start_time = time.time()
step_idx = 0

# ============ Utilities ============

def local_to_world(robot_pose, local_goal):
    """Transform local goal to world coordinates."""
    rx, ry, ryaw = robot_pose[0], robot_pose[1], robot_pose[2]
    gx, gy, gz = local_goal
    
    world_x = rx + (gx * math.cos(ryaw) - gy * math.sin(ryaw))
    world_y = ry + (gx * math.sin(ryaw) + gy * math.cos(ryaw))
    world_z = gz # Assume z stays same for now
    
    return [world_x, world_y, world_z]

def world_to_local(robot_pose, world_goal):
    """Transform world goal to robot local coordinates."""
    rx, ry, ryaw = robot_pose[0], robot_pose[1], robot_pose[2]
    wx, wy, _ = world_goal
    
    dx = wx - rx
    dy = wy - ry
    
    local_x = dx * math.cos(ryaw) + dy * math.sin(ryaw)
    local_y = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
    
    return np.array([[local_x, local_y, 0.0]])

def reconstruct_trajectory(delta_xyt):
    """Reconstruct global trajectory from incremental steps."""
    B, T, _ = delta_xyt.shape
    xy = np.zeros((B, T + 1, 2))
    delta_xy = delta_xyt[:, :, :2]
    xy[:, 1:] = np.cumsum(delta_xy, axis=1)
    return xy

def process_dp_actions(dp_actions, use_discrete=False):
    """Process raw model actions into executable trajectories or discrete commands."""
    # Unnormalize: model predicts dx, dy in range [-1, 1] scaled by 4
    actions = dp_actions.copy()
    actions[:, :, :2] /= 4.0
    
    trajectories = reconstruct_trajectory(actions)
    mean_trajectory = np.mean(trajectories, axis=0) # (T+1, 2)
    
    if not use_discrete:
        return mean_trajectory
        
    # Optional: Discrete action mapping (move forward, turn left/right)
    # can be implemented here if needed for specific robot controllers
    return mean_trajectory

# ============ Request Handlers ============

@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    global step_idx, start_time, is_first_step, goal_world_pos
    global _flux_fps_writer, _flux_fps_rgbd_writer
    
    try:
        t_start = time.time()
        
        # 1. Parse Input
        image_file = request.files['image']
        depth_file = request.files['depth']
        meta_data = json.loads(request.form['json'])
        robot_pose = np.array(json.loads(request.form['robot_orientation'])) # [x, y, yaw]
        
        # 2. Reset Logic
        if meta_data.get('reset', False):
            print("[FLUX] Resetting agent and navigation state...")
            agent.reset(1, -3.0)
            is_first_step = True
            step_idx = 0
            start_time = time.time()
            # If reset provides a new goal, update it
            # goal_world_pos = ... 

        # 3. Coordinate Transformation
        if is_first_step:
            # Initialize goal in world frame based on first seen robot pose
            # if the input goal was relative.
            goal_world_pos = local_to_world(robot_pose, [7.0, 0.2, 0.0])
            is_first_step = False
            print(f"[FLUX] Goal locked at world: {goal_world_pos}")

        # Calculate local goal for model
        dx = goal_world_pos[0] - robot_pose[0]
        dy = goal_world_pos[1] - robot_pose[1]
        ryaw = robot_pose[2]
        
        local_x = dx * math.cos(ryaw) + dy * math.sin(ryaw)
        local_y = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
        point_goal = np.array([[local_x, local_y, 0.0]])
        
        # 4. Decode Observations
        # RGB: (H, W, 3) → (1, H, W, 3)
        img_pil = Image.open(image_file.stream).convert('RGB')
        img_np = np.asarray(img_pil).astype(np.uint8)
        img_np = img_np.reshape((1, img_np.shape[0], img_np.shape[1], 3))  # (1, H, W, 3)

        # Depth: (H, W) → (1, H, W, 1)
        depth_pil = Image.open(depth_file.stream).convert('I')
        depth_np = np.asarray(depth_pil).astype(np.float32) / 10000.0
        depth_np = depth_np[:, :, np.newaxis]                              # (H, W, 1)
        depth_np = depth_np.reshape((1, depth_np.shape[0], depth_np.shape[1], 1))  # (1, H, W, 1)
        
        # 5. Model Inference
        step_idx += 1
        exec_traj, all_trajs, all_vals, traj_mask, traj_depth_mask = agent.step_pointgoal(point_goal, img_np, depth_np)
        
        # 6. Post-processing & Visualization
        depth_vis = np.clip(depth_np[0, :, :, 0], 0.1, 3.0)
        depth_vis = ((depth_vis - 0.1) / 2.9 * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        rgbd_frame = np.hstack([img_np[0], depth_color])

        # visualize_trajectory 返回单张已拼好的图
        debug_full = vis_manager.visualize_trajectory(
            img_np[0], depth_np[0], args.camera_intrinsic,
            exec_traj[0], robot_pose=robot_pose,
            all_trajectories_points=all_trajs[0],
            all_trajectories_values=all_vals[0],
        )

        # traj_mask / traj_depth_mask 单独拼在左边（可选）
        # 统一高度后横向拼接
        h = debug_full.shape[0]
        traj_mask_resized = cv2.resize(traj_mask, 
            (int(traj_mask.shape[1] * h / traj_mask.shape[0]), h))
        traj_depth_resized = cv2.resize(traj_depth_mask,
            (int(traj_depth_mask.shape[1] * h / traj_depth_mask.shape[0]), h))

        debug_full = np.concatenate((traj_mask_resized, traj_depth_resized, debug_full), axis=1)

        Image.fromarray(debug_full).save('./output_flux.png')
        if _flux_fps_writer: _flux_fps_writer.append_data(debug_full)
        if _flux_fps_rgbd_writer: _flux_fps_rgbd_writer.append_data(rgbd_frame)
        
        # 7. Prepare Response
        output_traj = process_dp_actions(exec_traj, use_discrete=False)
        
        t_end = time.time()
        print(f"[FLUX] Step {step_idx} | Latency: {t_end-t_start:.3f}s | Goal Dist: {np.linalg.norm(point_goal[0][:2]):.2f}m")
        
        return jsonify({
            "trajectory": output_traj.tolist(),
            "status": "success"
        })

    except Exception as e:
        print(f"[FLUX Error] {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5801)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="/workspace/FLUX/checkpoints/flux_v1.ckpt")
    parser.add_argument("--save_video", action="store_true", default=True)
    args = parser.parse_args()

    # Camera Intrinsics
    args.camera_intrinsic = np.array([
        [386.5, 0.0, 328.9, 0.0],
        [0.0, 386.5, 244.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    print(f"[FLUX] Initializing Agent on {args.device}...")
    agent = GRPO_Agent(
        args.model_path,
        args.camera_intrinsic,
        image_size=224,
        memory_size=8,
        predict_size=24,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        device=args.device
    )
    agent.reset(1, -3.0)

    if args.save_video:
        log_dir = "/workspace/FLUX/logs"
        os.makedirs(log_dir, exist_ok=True)          # ← 先建目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _flux_fps_writer = imageio.get_writer(f"{log_dir}/flux_{timestamp}_debug.mp4", fps=5)
        _flux_fps_rgbd_writer = imageio.get_writer(f"{log_dir}/flux_{timestamp}_rgbd.mp4", fps=5)
        print(f"[FLUX] Video logging enabled: {log_dir}/flux_{timestamp}_*.mp4")

    print(f"[FLUX] Server starting at http://{args.host}:{args.port}")
    try:
        app.run(host=args.host, port=args.port, threaded=False)
    finally:
        if _flux_fps_writer: _flux_fps_writer.close()
        if _flux_fps_rgbd_writer: _flux_fps_rgbd_writer.close()
        print("[FLUX] Server stopped.")
