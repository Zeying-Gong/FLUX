"""
flux/server.py

推理服务器 —— 与 NavDP 接口完全一致，
加载 RL 微调后的权重（flux/policy_network.py 的 CFM_RL_Policy）。

用法：
    python server.py --port 8892 --checkpoint ./checkpoints_rl/rl_final.ckpt [--cfm-steps 5]

然后在 eval 脚本中把 --port 改成 8892 即可，其他不变。
"""

import os
import sys
import argparse
import datetime
import time
import json
import numpy as np
import cv2
import imageio
from PIL import Image
from flask import Flask, request, jsonify

# 把当前目录加到 path，使 policy_network / policy_agent 可以 import
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from policy_network import CFM_RL_Policy

# ---- 参数 ----
parser = argparse.ArgumentParser(description="flux inference server")
parser.add_argument("--port", type=int, default=8892, help="服务器端口（默认 8892）")
parser.add_argument("--checkpoint", type=str, default="checkpoints/flux_v1.ckpt", help="RL 微调后的 checkpoint 路径")
parser.add_argument("--cfm-steps", type=int, default=5, help="ODE 积分步数")
parser.add_argument("--normalization-config", type=str, default=None, help="归一化配置文件路径")
parser.add_argument("--device", type=str, default="cuda:0", help="推理设备")
args = parser.parse_known_args()[0]

app = Flask(__name__)

# ---- 全局状态 ----
_policy: CFM_RL_Policy = None
_memory_queues: list = None
_batch_size: int = 1
_stop_threshold: float = -3.0
_fps_writer = None
_image_size: int = 224
_memory_size: int = 8


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------

def _build_policy():
    global _policy
    _policy = CFM_RL_Policy(
        image_size=_image_size,
        memory_size=_memory_size,
        predict_size=24,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        device=args.device,
        normalization_config=args.normalization_config,
        cfm_num_steps=args.cfm_steps,
    )
    import torch
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    incompatible = _policy.load_state_dict(ckpt, strict=False)
    print(f"[flux server] Loaded: {args.checkpoint}")
    print(f"  missing={len(incompatible.missing_keys)}, "
          f"unexpected={len(incompatible.unexpected_keys)}")
    _policy.to(args.device)
    _policy.eval()


def _process_image(images: np.ndarray) -> np.ndarray:
    """预处理 RGB 图像：resize + pad + 归一化，与 policy_agent 一致"""
    assert len(images.shape) == 4
    H, W = images.shape[1], images.shape[2]
    prop = _image_size / max(H, W)
    result = []
    for img in images:
        r = cv2.resize(img, (-1, -1), fx=prop, fy=prop)
        ph = max((_image_size - r.shape[0]) // 2, 0)
        pw = max((_image_size - r.shape[1]) // 2, 0)
        r = np.pad(r, ((ph, ph), (pw, pw), (0, 0)), constant_values=0)
        r = cv2.resize(r, (_image_size, _image_size))
        result.append(r.astype(np.float32) / 255.0)
    return np.array(result)


def _process_depth(depths: np.ndarray) -> np.ndarray:
    depths = depths.copy()
    depths[depths == np.inf] = 0
    H, W = depths.shape[1], depths.shape[2]
    prop = _image_size / max(H, W)
    result = []
    for depth in depths:
        r = cv2.resize(depth, (-1, -1), fx=prop, fy=prop)
        ph = max((_image_size - r.shape[0]) // 2, 0)
        pw = max((_image_size - r.shape[1]) // 2, 0)
        r = np.pad(r, ((ph, ph), (pw, pw)), constant_values=0)
        r = cv2.resize(r, (_image_size, _image_size))
        r[r > 5.0] = 0
        r[r < 0.1] = 0
        result.append(r[:, :, np.newaxis])
    return np.array(result)


def _process_pointgoal(goals: np.ndarray) -> np.ndarray:
    clip = goals.clip(-10, 10)
    clip[:, 0] = np.clip(clip[:, 0], 0, 10)
    return clip


def _get_input_images(proc_images: np.ndarray) -> np.ndarray:
    """维护 memory queue，返回 (batch, memory_size, H, W, 3)"""
    input_images = []
    for i in range(len(_memory_queues)):
        if len(_memory_queues[i]) < _memory_size:
            _memory_queues[i].append(proc_images[i])
            buf = np.array(_memory_queues[i])
            buf = np.pad(buf, ((_memory_size - buf.shape[0], 0), (0,0), (0,0), (0,0)))
        else:
            del _memory_queues[i][0]
            _memory_queues[i].append(proc_images[i])
            buf = np.array(_memory_queues[i])
        input_images.append(buf)
    return np.array(input_images)


def _decode_image_depth(image_file, depth_file, declared_batch_size: int = None):
    image = Image.open(image_file.stream).convert("RGB")
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    depth = Image.open(depth_file.stream).convert("I")
    depth = np.asarray(depth).astype(np.float32) / 10000.0

    # 用 declared_batch_size，如果没传则用全局 _batch_size
    batch_size = declared_batch_size if declared_batch_size is not None else _batch_size

    H_single_img = image.shape[0] // batch_size
    H_single_dep = depth.shape[0] // batch_size

    image = image.reshape((batch_size, H_single_img, image.shape[1], 3))
    depth = depth.reshape((batch_size, H_single_dep, depth.shape[1], 1))
    return image, depth, batch_size  # 把实际 batch_size 也返回出去

# ------------------------------------------------------------------
# Flask 路由（与 NavDP server 完全一致）
# ------------------------------------------------------------------

@app.route("/navigator_reset", methods=["POST"])
def navigator_reset():
    global _policy, _memory_queues, _batch_size, _stop_threshold, _fps_writer

    intrinsic = np.array(request.get_json().get("intrinsic"))
    _stop_threshold = float(request.get_json().get("stop_threshold"))
    _batch_size = int(request.get_json().get("batch_size"))

    if _policy is None:
        _build_policy()

    _memory_queues = [[] for _ in range(_batch_size)]

    global _fps_writer
    if _fps_writer is not None:
        try:
            _fps_writer.close()
        except Exception:
            pass
    fmt = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    _fps_writer = imageio.get_writer(f"{fmt}_fps_flux_pointgoal.mp4", fps=7)

    return jsonify({"algo": "flux"})


@app.route("/navigator_reset_env", methods=["POST"])
def navigator_reset_env():
    env_id = int(request.get_json().get("env_id"))
    if _memory_queues is not None and env_id < len(_memory_queues):
        _memory_queues[env_id] = []
    return jsonify({"algo": "flux"})


@app.route("/pointgoal_step", methods=["POST"])
def pointgoal_step():
    import torch

    goal_data = json.loads(request.form.get("goal_data"))
    # goal_data 里的列表长度就是真实 batch_size
    actual_batch_size = len(goal_data["goal_x"])

    image, depth, actual_batch_size = _decode_image_depth(
        request.files["image"], request.files["depth"], actual_batch_size
    )
    # ... 后续用 actual_batch_size 替换所有 _batch_size ...
    goal = np.stack(
        (np.array(goal_data["goal_x"]),
         np.array(goal_data["goal_y"]),
         np.zeros(actual_batch_size)),
        axis=1
    )

    proc_images  = _process_image(image)
    proc_depths  = _process_depth(depth)
    input_images = _get_input_images(proc_images)
    input_goals  = _process_pointgoal(goal)

    with torch.no_grad():
        all_trajectory, critic_scores, _ = _policy.forward_train_pointgoal(
            input_goals, input_images, proc_depths, sample_num=16
        )

    all_trajectory_np = all_trajectory.cpu().numpy()
    all_values_np     = critic_scores.cpu().numpy()

    best_idx     = all_values_np.argmax(axis=1)
    execute_traj = all_trajectory_np[np.arange(actual_batch_size), best_idx]

    if all_values_np.max() < _stop_threshold:
        execute_traj[:, :, 0] = 0.0
        execute_traj[:, :, 1] = np.sign(execute_traj[:, :, 1].mean())

    return jsonify({
        "trajectory":     execute_traj.tolist(),
        "all_trajectory": all_trajectory_np.tolist(),
        "all_values":     all_values_np.tolist(),
    })


@app.route("/nogoal_step", methods=["POST"])
def nogoal_step():
    import torch

    # nogoal 没有 goal_data，从图像高度推断
    # 客户端垂直拼接了 batch_size 帧，高度是单帧的整数倍
    # 这里先按 _batch_size 解码，如果测试时需要动态 batch 再改
    image, depth, actual_batch_size = _decode_image_depth(
        request.files["image"], request.files["depth"], _batch_size
    )

    proc_images  = _process_image(image)
    proc_depths  = _process_depth(depth)
    input_images = _get_input_images(proc_images)

    with torch.no_grad():
        all_trajectory, critic_scores, _ = _policy.forward_train_nogoal(
            input_images, proc_depths, sample_num=16
        )

    all_trajectory_np = all_trajectory.cpu().numpy()
    all_values_np     = critic_scores.cpu().numpy()

    best_idx = np.zeros(actual_batch_size, dtype=np.int64)
    for b in range(actual_batch_size):
        traj_lens   = np.linalg.norm(all_trajectory_np[b, :, -1, :2], axis=-1)
        valid_mask  = traj_lens > 0.3
        masked_scores = all_values_np[b].copy()
        if valid_mask.any():
            masked_scores[~valid_mask] = -1e9
        best_idx[b] = masked_scores.argmax()

    execute_traj = all_trajectory_np[np.arange(actual_batch_size), best_idx]

    if all_values_np.max() < _stop_threshold:
        execute_traj[:, :, 0] = 0.0
        execute_traj[:, :, 1] = np.sign(execute_traj[:, :, 1].mean())

    return jsonify({
        "trajectory":     execute_traj.tolist(),
        "all_trajectory": all_trajectory_np.tolist(),
        "all_values":     all_values_np.tolist(),
    })


@app.route("/imagegoal_step", methods=["POST"])
def imagegoal_step():
    import torch

    image, depth, actual_batch_size = _decode_image_depth(
        request.files["image"], request.files["depth"], _batch_size
    )

    goal_file  = request.files["goal"]
    goal_image = Image.open(goal_file.stream).convert("RGB")
    goal_image = np.asarray(goal_image)
    goal_image = cv2.cvtColor(goal_image, cv2.COLOR_RGB2BGR)
    H_single   = goal_image.shape[0] // actual_batch_size
    goal_image = goal_image.reshape((actual_batch_size, H_single, goal_image.shape[1], 3))

    proc_images  = _process_image(image)
    proc_depths  = _process_depth(depth)
    proc_goals   = _process_image(goal_image)
    input_images = _get_input_images(proc_images)

    with torch.no_grad():
        all_trajectory, critic_scores, _, _ = _policy.predict_imagegoal_action(
            proc_goals, input_images, proc_depths, sample_num=16
        )

    all_trajectory_np = np.array(all_trajectory)
    all_values_np     = np.array(critic_scores)

    best_idx     = all_values_np.argmax(axis=1)
    execute_traj = all_trajectory_np[np.arange(actual_batch_size), best_idx]

    if all_values_np.max() < _stop_threshold:
        execute_traj[:, :, 0] = 0.0
        execute_traj[:, :, 1] = np.sign(execute_traj[:, :, 1].mean())

    return jsonify({
        "trajectory":     execute_traj.tolist(),
        "all_trajectory": all_trajectory_np.tolist(),
        "all_values":     all_values_np.tolist(),
    })
    
if __name__ == "__main__":
    print(f"[flux server] Starting on port {args.port}, device={args.device}")
    print(f"[flux server] Checkpoint: {args.checkpoint}")
    app.run(host="127.0.0.1", port=args.port)