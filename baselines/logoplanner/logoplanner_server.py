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

from PIL import Image, ImageDraw, ImageFont
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--port",type=int,default=8888)
parser.add_argument("--checkpoint",type=str,default="./logoplanner_policy.ckpt")
args = parser.parse_known_args()[0]

app = Flask(__name__)
logoplanner_navigator = None
logoplanner_fps_writer = None

@app.route("/navigator_reset",methods=['POST'])
def logoplanner_reset():
    global logoplanner_navigator,logoplanner_fps_writer
    intrinsic = np.array(request.get_json().get('intrinsic'))
    threshold = np.array(request.get_json().get('stop_threshold'))
    batchsize = np.array(request.get_json().get('batch_size'))
    if logoplanner_navigator is None:
        logoplanner_navigator = LoGoPlanner_Agent(intrinsic,
                                            image_size=224,
                                            memory_size=8,
                                            predict_size=24,
                                            temporal_depth=16,
                                            heads=8,
                                            token_dim=384,
                                            navi_model=args.checkpoint,
                                            device='cuda:0')
        logoplanner_navigator.reset(batchsize,threshold)
    else:
        logoplanner_navigator.reset(batchsize,threshold)

    if logoplanner_fps_writer is None:
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        logoplanner_fps_writer = imageio.get_writer("{}_fps_pointgoal.mp4".format(format_time),fps=7)
    else:
        logoplanner_fps_writer.close()
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        logoplanner_fps_writer = imageio.get_writer("{}_fps_pointgoal.mp4".format(format_time),fps=7)
    return jsonify({"algo":"logoplanner"})

@app.route("/navigator_reset_env",methods=['POST'])
def logoplanner_reset_env():
    global logoplanner_navigator
    logoplanner_navigator.reset_env(int(request.get_json().get('env_id')))
    return jsonify({"algo":"logoplanner"})

@app.route("/pointgoal_step",methods=['POST'])
def logoplanner_step_xy():
    global logoplanner_navigator,logoplanner_fps_writer
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x,goal_y,np.zeros_like(goal_x)),axis=1)
    batch_size = logoplanner_navigator.batch_size
    
    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    phase2_time = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask, sub_pointgoal_pd = logoplanner_navigator.step_pointgoal(goal,image,depth)
    phase3_time = time.time()
    logoplanner_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, phase4_time-phase3_time, time.time() - start_time))

    return jsonify({'trajectory': execute_trajectory.tolist(),
                    'all_trajectory': all_trajectory.tolist(),
                    'all_values': all_values.tolist(),
                    'sub_pointgoal_pd': sub_pointgoal_pd.tolist()})

@app.route("/nogoal_step",methods=['POST'])
def logoplanner_step_nogoal():
    global logoplanner_navigator,logoplanner_fps_writer
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']
    batch_size = logoplanner_navigator.batch_size
    
    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    phase2_time = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = logoplanner_navigator.step_nogoal(image,depth)
    phase3_time = time.time()
    logoplanner_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, phase4_time-phase3_time, time.time() - start_time))
    return jsonify({'trajectory': execute_trajectory.tolist(),
                    'all_trajectory': all_trajectory.tolist(),
                    'all_values': all_values.tolist()})

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=args.port)
