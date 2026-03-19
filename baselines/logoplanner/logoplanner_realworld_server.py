from PIL import Image
from flask import Flask, request, jsonify
from policy_agent import LoGoPlanner_Agent
import numpy as np
import cv2
import imageio
import time
import datetime
import json
import os
from deployment.mpc_controller import Mpc_controller
from deployment.visualization import plot_trajectory
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--port",type=int,default=8888)
parser.add_argument("--checkpoint",type=str,default="./logoplanner_policy.ckpt")
args = parser.parse_known_args()[0]

app = Flask(__name__)
navdp_navigator = None
navdp_fps_writer = None

@app.route("/navigator_reset",methods=['POST'])
def navdp_reset():
    global navdp_navigator,navdp_fps_writer
    intrinsic = np.array(request.get_json().get('intrinsic'))
    threshold = np.array(request.get_json().get('stop_threshold'))
    batchsize = np.array(request.get_json().get('batch_size'))
    if navdp_navigator is None:
        navdp_navigator = LoGoPlanner_Agent(intrinsic,
                                image_size=224,
                                memory_size=8,
                                predict_size=24,
                                temporal_depth=16,
                                heads=8,
                                token_dim=384,
                                navi_model=args.checkpoint,
                                device='cuda:0')
        navdp_navigator.reset(batchsize,threshold)
    else:
        navdp_navigator.reset(batchsize,threshold)

    if navdp_fps_writer is None:
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        navdp_fps_writer = imageio.get_writer("{}_fps_pointgoal.mp4".format(format_time),fps=2)
    else:
        navdp_fps_writer.close()
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        navdp_fps_writer = imageio.get_writer("{}_fps_pointgoal.mp4".format(format_time),fps=2)
    return jsonify({"algo":"logoplanner"})

@app.route("/navigator_reset_env",methods=['POST'])
def navdp_reset_env():
    global navdp_navigator
    navdp_navigator.reset_env(int(request.get_json().get('env_id')))
    return jsonify({"algo":"logoplanner"})

@app.route("/pointgoal_step",methods=['POST'])
def navdp_step_xy():
    global navdp_navigator,navdp_fps_writer
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x,goal_y,np.zeros_like(goal_x)),axis=1)
    batch_size = navdp_navigator.batch_size
    
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.copy()
    depth[np.isnan(depth)] = 0
    depth[np.isinf(depth)] = 0
    depth[depth > 10000] = 0
    depth = depth.astype(np.float32)
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    # Depth dilation: fill zero values with minimum non-zero values in 80x80 window
    from scipy import ndimage
    zero_mask = (depth == 0)
    if np.any(zero_mask):
        window_size = 80
        filled_depth = depth.copy()
        non_zero_depth = depth.copy().astype(np.float32)
        non_zero_depth[depth == 0] = np.inf
        min_filtered = ndimage.minimum_filter(non_zero_depth, size=window_size)
        valid_fill_mask = zero_mask & (min_filtered != np.inf)
        filled_depth[valid_fill_mask] = min_filtered[valid_fill_mask]
        depth = filled_depth
    
    depth= depth / 1000.0
    depth[np.where(depth < 0)] = 0
    
    execute_trajectory, all_trajectory, all_values, trajectory_mask, sub_pointgoal_pd = navdp_navigator.step_pointgoal(goal,image,depth)
    
    navdp_fps_writer.append_data(trajectory_mask)

    execute_trajectory = execute_trajectory[0, 2:]  # Take the first in batch
    if abs(execute_trajectory[:, 1]).max() > 0.3:
        execute_trajectory[:, 1] *= 2.0
    execute_trajectory[:, 0] *= 1.0

    mpc = Mpc_controller(
        execute_trajectory,
        is_omnidirectional=True,
        vx_max=0.8, vy_max=0.8, wz_max=0.3
    )
    mpc.reset()
    
    init_ref_traj = execute_trajectory
    init_theta = mpc.compute_ref_theta(init_ref_traj)[0]
    init_state = np.array([execute_trajectory[0, 0], execute_trajectory[0, 1], init_theta])
    
    x_history = [init_state]
    u_history = []
    max_steps = 20
    
    for i in range(max_steps):
        current_pos = x_history[-1][:2]
        end_pos = execute_trajectory[-1, :2]
        if np.linalg.norm(current_pos - end_pos) < 0.05:
            break
        opt_u, opt_x = mpc.solve(x_history[-1])
        x_history.append(opt_x[1, :])
        u_history.append(opt_u[0, :])

    x_history = np.array(x_history)
    u_history = np.array(u_history)

    if len(u_history) == 0:
        u_history = np.zeros((10, 3))
        
    plot_trajectory(init_ref_traj, x_history, u_history, traj_name='real')
    u_history = u_history[0:]

    u_history[:, 2] = u_history[:, 2] * 180.0 / np.pi  # Convert to degrees
    return jsonify({'cmd_list': u_history.tolist()})


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=args.port)
