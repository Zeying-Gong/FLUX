"""
CFM 导航模型 HTTP 服务器（NavDP 的 CFM 版本）
提供与 NavDP 相同的 Flask API，推理使用 CFM ODE 采样
"""
from PIL import Image
from flask import Flask, request, jsonify
from policy_agent import CFM_Agent
import numpy as np
import cv2
import imageio
import time
import datetime
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8889, help="服务器端口（默认 8889，与 navdp 区分）")
parser.add_argument("--checkpoint", type=str, required=True, help="CFM 微调权重路径（.ckpt / pytorch_model.bin）")
parser.add_argument("--normalization-config", type=str, default=None, help="归一化配置 JSON（与训练一致）")
parser.add_argument("--cfm-steps", type=int, default=5, help="ODE 积分步数")
args = parser.parse_known_args()[0]

def _resolve_norm_config(path):
    if not path:
        return None
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    for base in [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]:
        cand = os.path.normpath(os.path.join(base, path))
        if os.path.isfile(cand):
            return cand
    return path

app = Flask(__name__)
cfm_navigator = None
cfm_fps_writer = None

@app.route("/navigator_reset", methods=['POST'])
def cfm_reset():
    global cfm_navigator, cfm_fps_writer
    intrinsic = np.array(request.get_json().get('intrinsic'))
    threshold = np.array(request.get_json().get('stop_threshold'))
    batchsize = np.array(request.get_json().get('batch_size'))

    if cfm_navigator is None:
        norm_cfg = _resolve_norm_config(args.normalization_config)
        cfm_navigator = CFM_Agent(
            intrinsic,
            image_size=224,
            memory_size=8,
            predict_size=24,
            temporal_depth=16,
            heads=8,
            token_dim=384,
            navi_model=args.checkpoint,
            normalization_config=norm_cfg,
            cfm_num_steps=args.cfm_steps,
            device='cuda:0',
        )
        cfm_navigator.reset(batchsize, threshold)
    else:
        cfm_navigator.reset(batchsize, threshold)

    if cfm_fps_writer is None:
        format_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
        cfm_fps_writer = imageio.get_writer("{}_fps_cfm_pointgoal.mp4".format(format_time), fps=7)
    else:
        cfm_fps_writer.close()
        format_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
        cfm_fps_writer = imageio.get_writer("{}_fps_cfm_pointgoal.mp4".format(format_time), fps=7)

    return jsonify({"algo": "cfm"})

@app.route("/navigator_reset_env", methods=['POST'])
def cfm_reset_env():
    global cfm_navigator
    cfm_navigator.reset_env(int(request.get_json().get('env_id')))
    return jsonify({"algo": "cfm"})

def _decode_image_depth(image_file, depth_file, batch_size):
    image = Image.open(image_file.stream).convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    depth = Image.open(depth_file.stream).convert('I')
    depth = np.asarray(depth)[:, :, np.newaxis].astype(np.float32) / 10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    return image, depth

@app.route("/pointgoal_step", methods=['POST'])
def cfm_step_pointgoal():
    global cfm_navigator, cfm_fps_writer
    start = time.time()
    goal_data = json.loads(request.form.get('goal_data'))
    goal = np.stack((np.array(goal_data['goal_x']), np.array(goal_data['goal_y']), np.zeros_like(np.array(goal_data['goal_x']))), axis=1)
    batch_size = cfm_navigator.batch_size
    image, depth = _decode_image_depth(request.files['image'], request.files['depth'], batch_size)
    t1 = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = cfm_navigator.step_pointgoal(goal, image, depth)
    t2 = time.time()
    cfm_fps_writer.append_data(trajectory_mask)
    print("phase1:%f, phase2:%f, phase3:%f, all:%f" % (t1 - start, t2 - t1, time.time() - t2, time.time() - start))
    return jsonify({'trajectory': execute_trajectory.tolist(), 'all_trajectory': all_trajectory.tolist(), 'all_values': all_values.tolist()})

@app.route("/pixelgoal_step", methods=['POST'])
def cfm_step_pixelgoal():
    global cfm_navigator, cfm_fps_writer
    start = time.time()
    goal_data = json.loads(request.form.get('goal_data'))
    goal = np.stack((np.array(goal_data['goal_x']), np.array(goal_data['goal_y'])), axis=1)
    batch_size = cfm_navigator.batch_size
    image, depth = _decode_image_depth(request.files['image'], request.files['depth'], batch_size)
    t1 = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = cfm_navigator.step_pixelgoal(goal, image, depth)
    t2 = time.time()
    cfm_fps_writer.append_data(trajectory_mask)
    print("phase1:%f, phase2:%f, phase3:%f, all:%f" % (t1 - start, t2 - t1, time.time() - t2, time.time() - start))
    return jsonify({'trajectory': execute_trajectory.tolist(), 'all_trajectory': all_trajectory.tolist(), 'all_values': all_values.tolist()})

@app.route("/imagegoal_step", methods=['POST'])
def cfm_step_imagegoal():
    global cfm_navigator, cfm_fps_writer
    start = time.time()
    batch_size = cfm_navigator.batch_size
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_file = request.files['goal']
    image, depth = _decode_image_depth(image_file, depth_file, batch_size)
    goal = Image.open(goal_file.stream).convert('RGB')
    goal = np.asarray(goal)
    goal = cv2.cvtColor(goal, cv2.COLOR_RGB2BGR)
    goal = goal.reshape((batch_size, -1, goal.shape[1], 3))
    t1 = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = cfm_navigator.step_imagegoal(goal, image, depth)
    t2 = time.time()
    cfm_fps_writer.append_data(trajectory_mask)
    print("phase1:%f, phase2:%f, phase3:%f, all:%f" % (t1 - start, t2 - t1, time.time() - t2, time.time() - start))
    return jsonify({'trajectory': execute_trajectory.tolist(), 'all_trajectory': all_trajectory.tolist(), 'all_values': all_values.tolist()})

@app.route("/nogoal_step", methods=['POST'])
def cfm_step_nogoal():
    global cfm_navigator, cfm_fps_writer
    start = time.time()
    batch_size = cfm_navigator.batch_size
    image, depth = _decode_image_depth(request.files['image'], request.files['depth'], batch_size)
    t1 = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = cfm_navigator.step_nogoal(image, depth)
    t2 = time.time()
    cfm_fps_writer.append_data(trajectory_mask)
    print("phase1:%f, phase2:%f, phase3:%f, all:%f" % (t1 - start, t2 - t1, time.time() - t2, time.time() - start))
    return jsonify({'trajectory': execute_trajectory.tolist(), 'all_trajectory': all_trajectory.tolist(), 'all_values': all_values.tolist()})

@app.route("/navdp_step_ip_mixgoal", methods=['POST'])
def cfm_step_ip_mixgoal():
    global cfm_navigator, cfm_fps_writer
    start = time.time()
    batch_size = cfm_navigator.batch_size
    point_data = json.loads(request.form.get('goal_data'))
    point_goal = np.stack((np.array(point_data['goal_x']), np.array(point_data['goal_y']), np.zeros_like(np.array(point_data['goal_x']))), axis=1)
    image_goal_file = request.files['image_goal']
    image_goal = Image.open(image_goal_file.stream).convert('RGB')
    image_goal = np.asarray(image_goal)
    image_goal = cv2.cvtColor(image_goal, cv2.COLOR_RGB2BGR).reshape((batch_size, -1, image_goal.shape[1], 3))
    image, depth = _decode_image_depth(request.files['image'], request.files['depth'], batch_size)
    t1 = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = cfm_navigator.step_point_image_goal(point_goal, image_goal, image, depth)
    t2 = time.time()
    cfm_fps_writer.append_data(trajectory_mask)
    print("phase1:%f, phase2:%f, phase3:%f, all:%f" % (t1 - start, t2 - t1, time.time() - t2, time.time() - start))
    return jsonify({'trajectory': execute_trajectory.tolist(), 'all_trajectory': all_trajectory.tolist(), 'all_values': all_values.tolist()})

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=args.port)
