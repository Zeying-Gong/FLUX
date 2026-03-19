"""
NavDP 导航模型 HTTP 服务器
用来给go2提供导航推理的 Flask API 接口
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
import sys
sys.path.append('/home/justin/GitRepos/NavDP/baselines/nomad/diffusion_policy')
sys.path.append('/home/justin/GitRepos/NavDP/baselines/flownav/depth_anything_v2')

from cfm_rf_rl.policy_agent import GRPO_Agent
from navdp.policy_agent import NavDP_Agent
from nomad.nomad_agent import NoMaDAgent
from lotus.lotus_policy_agent import NavDP_Agent_cross_modal
from flownav.flownav_agent import FlowNavAgent
from cfm_rf.policy_agent import CFM_Agent
import cv2
import imageio

from PIL import Image, ImageDraw, ImageFont
from navdp.visualization_utils import VisualizationManager
import math

app = Flask(__name__)
idx = 0
start_time = time.time()
output_dir = ''


# ============ 全局变量 ============
navdp_fps_writer = None  # 视频写入器，用于保存可视化结果
cfmrl_fps_writer = None
cfm_fps_writer = None 
nomad_fps_writer = None
flownav_fps_writer = None  
# navdp_depth_fps_writer = None  # 视频写入器，用于保存可视化结果
# navdp_trajvis_writer_1 = None  # 视频写入器，用于保存可视化结果
# navdp_trajvis_writer_2 = None  # 视频写入器，用于保存可视化结果
vis_manager = VisualizationManager(history_size=5)
goal_position = [7,0.2,0]
first = True

# navdp_fps_rgbd_writer = None 



def local_to_world(robot_orientation, goal_position):
    robot_x, robot_y, robot_z = robot_orientation[0], robot_orientation[1], 0
    yaw = robot_orientation[2]  # 只取yaw
    goal_x, goal_y, goal_z = goal_position

    # 将yaw转换为弧度
    yaw_radians = yaw

    # 计算世界坐标系中的位置
    world_x = robot_x + (goal_x * math.cos(yaw_radians) - goal_y * math.sin(yaw_radians))
    world_y = robot_y + (goal_x * math.sin(yaw_radians) + goal_y * math.cos(yaw_radians))
    world_z = robot_z + goal_z

    return (world_x, world_y, world_z)

def traj_to_actions(dp_actions, use_discrate_action=True):
    def reconstruct_xy_from_delta(delta_xyt):
        """
        Input:
            delta_xyt: [B, T, 3], dx, dy are position increments in global coordinates, dθ is heading difference (not used for position)
            start_xy: [B, 2] starting point
        Output:
            xy: [B, T+1, 2] reconstructed global trajectory
        """
        start_xy = np.zeros((len(delta_xyt), 2))
        delta_xy = delta_xyt[:, :, :2]  # Take dx, dy parts
        cumsum_xy = np.cumsum(delta_xy, axis=1)  # [B, T, 2]

        B = delta_xyt.shape[0]
        T = delta_xyt.shape[1]
        xy = np.zeros((B, T + 1, 2))
        xy[:, 0] = start_xy
        xy[:, 1:] = start_xy[:, None, :] + cumsum_xy

        return xy

    def trajectory_to_discrete_actions_close_to_goal(trajectory, step_size=0.25, turn_angle_deg=15, lookahead=4):
        actions = []
        yaw = 0.0
        pos = trajectory[0]
        turn_angle_rad = np.deg2rad(turn_angle_deg)
        traj = trajectory
        goal = trajectory[-1]

        def normalize_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        while np.linalg.norm(pos - goal) > 0.2:
            # Find the nearest trajectory point index to current position
            dists = np.linalg.norm(traj - pos, axis=1)
            nearest_idx = np.argmin(dists)
            # Look ahead a bit (not exceeding trajectory end)
            target_idx = min(nearest_idx + lookahead, len(traj) - 1)
            target = traj[target_idx]
            # Target direction
            target_dir = target - pos
            if np.linalg.norm(target_dir) < 1e-6:
                break
            target_yaw = np.arctan2(target_dir[1], target_dir[0])
            # Difference between current yaw and target yaw
            delta_yaw = normalize_angle(target_yaw - yaw)
            n_turns = int(round(delta_yaw / turn_angle_rad))
            if n_turns > 0:
                actions += [2] * n_turns
            elif n_turns < 0:
                actions += [3] * (-n_turns)
            yaw = normalize_angle(yaw + n_turns * turn_angle_rad)

            # Move forward one step
            next_pos = pos + step_size * np.array([np.cos(yaw), np.sin(yaw)])

            # If moving forward one step makes us farther from goal, stop
            if np.linalg.norm(next_pos - goal) > np.linalg.norm(pos - goal):
                break

            actions.append(1)
            pos = next_pos

        return actions

    # unnormalize
    dp_actions[:, :, :2] /= 4.0
    all_trajectory = reconstruct_xy_from_delta(dp_actions)
    trajectory = np.mean(all_trajectory, axis=0)
    if use_discrate_action:
        actions = trajectory_to_discrete_actions_close_to_goal(trajectory)
        return actions
    else:
        return trajectory
    
def visualize_images(original_image, depth_normalized, batch_size=1):
    # 解码 RGB 图像

    
    # 解码深度图
    depth_display = cv2.convertScaleAbs(depth_normalized * 255)  # 应用伪彩色映射
    depth_original = depth_display.copy()  # 保存原始深度图用于显示

    
    # 创建可视化窗口
    cv2.namedWindow('RGB Image', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Depth Image', cv2.WINDOW_AUTOSIZE)
    
    # 显示图像
    cv2.imshow('RGB Image', original_image)
    cv2.imshow('Depth Image', depth_original)
    
    print("按任意键关闭窗口...")
    cv2.waitKey(0)  # 等待按键
    
    # 清理资源
    cv2.destroyAllWindows()
    
    return None

def stepwithwriters(agents,writers,robot_orientation,image,depth):
    for i in range(len(writers)):
        print(f'{i}th agent')
        execute_trajectory, all_trajectory, all_values, trajectory_mask, trajectory_depth_mask, trajectory_best_mask, trajectory_white_mask = \
            agents[i].step_nogoal(image, depth)

        vis_resized, vis_resized_all = vis_manager.visualize_trajectory(
                        image[0], depth[0], np.array(
            [[386.5, 0.0, 328.9, 0.0], [0.0, 386.5, 244, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        ),
                        execute_trajectory[0],
                        robot_pose=robot_orientation,
                        all_trajectories_points=all_trajectory[0],
                        all_trajectories_values=all_values[0],
                    )

        
        exe_rgb = np.concatenate((trajectory_mask, vis_resized), axis=1)
        # print("trajectory_depth_mask", trajectory_depth_mask.shape, "vis_resized_all",vis_resized_all.shape, 'vis_resized', vis_resized.shape)
        all_depth = np.concatenate((trajectory_depth_mask, vis_resized_all), axis=1)
        all_best = np.concatenate((trajectory_best_mask, vis_resized), axis=1)
        all_white = np.concatenate((trajectory_white_mask, vis_resized), axis=1)
        all_vis = np.concatenate((exe_rgb, all_depth,all_best,all_white), axis=0)
        
        
        writers[i].append_data(all_vis)
    return execute_trajectory
    

@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    global idx, output_dir, start_time, navdp_fps_writer, cfmrl_fps_writer, cfm_fps_writer, nomad_fps_writer,flownav_fps_writer, vis_manager,goal_position, first
    print('get')
    # ===== 接收输入数据 =====
    image_file = request.files['image']  # RGB 图像文件
    depth_file = request.files['depth']  # 深度图文件
    json_data = request.form['json']
    data = json.loads(json_data)
    robot_orientation = request.form['robot_orientation']
    robot_orientation = np.array(json.loads(robot_orientation))
    
    if first:
        # goal_position[0][0] += robot_orientation[0]
        # goal_position[0][1] += robot_orientation[1]
        goal_position = local_to_world(robot_orientation, goal_position)
        first = False
    
    # print('robot_orientation', robot_orientation)
    batch_size = cfmrl_agent.batch_size
    
    phase1_time = time.time()
    
    # ===== 解码 RGB 图像 =====
    image = Image.open(image_file.stream)  # 从字节流打开
    image = image.convert('RGB')  # 确保是 RGB 格式
    image = np.asarray(image)  # 转 numpy 数组
    # image = image[..., ::-1]
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # PIL是RGB，OpenCV是BGR
    
    
    # ===== 解码深度图 =====
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')  # 转成 32位整数
    depth = np.asarray(depth)[:, :, np.newaxis]  # 添加通道维度
    depth = depth.astype(np.float32) / 10000.0  # 从 uint16 转回米（发送时乘了10000）
    # visualize_images(image, depth, batch_size=1)
        
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))  # reshape 成 (batch, H, W, 1)
    image = image.reshape((batch_size, -1, image.shape[1], 3))  # reshape 成 (batch, H, W, 3)
    
    phase2_time = time.time()

    # camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # instruction = "Turn around and walk out of this office. Turn towards your slight right at the chair. Move forward to the walkway and go near the red bin. You can see an open door on your right side, go inside the open door. Stop at the computer monitor"
    policy_init = data['reset']
    if policy_init:
        start_time = time.time()
        idx = 0
        output_dir = 'output/runs' + datetime.now().strftime('%m-%d-%H%M')
        os.makedirs(output_dir, exist_ok=True)
        print("init reset model!!!")
        cfmrl_agent.reset(1, -3.0)
        navdp_agent.reset(1, -3.0)
        cfm_agent.reset(1,-3.0)
        flownav_navigator.reset(1)
        nomad_navigator.reset(1)

    idx += 1

    look_down = False
    t0 = time.time()
    dual_sys_output = {'output_trajectory':None,'output_pixel':None}

    # point_goal = np.array([[goal_position[0] - robot_orientation[0],
    #     goal_position[1] - robot_orientation[1],0]])
    
    dx = goal_position[0] - robot_orientation[0]
    dy = goal_position[1] - robot_orientation[1]
    yaw = robot_orientation[2]
    # print(f"robot yaw {yaw}")

    # 旋转到机器人局部坐标系
    local_x =  dx * math.cos(yaw) + dy * math.sin(yaw)
    local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)

    point_goal = np.array([[local_x, local_y, 0]])
    
    # print('point_goal', point_goal, 'goal position', goal_position)
    depth_image_clipped = np.clip(depth[0], 0.1, 3)
            
    # 归一化到[0, 1]范围
    depth_image_normalized = (depth_image_clipped - 0.1) / (3 - 0.1)  # 线性归一化
    
    # 将深度图像转换为RGB格式
    depth_image_rgb = cv2.cvtColor((depth_image_normalized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    rgbd = np.concatenate((image[0], depth_image_rgb), axis=0)
    
    execute_trajectory = stepwithwriters([nomad_navigator,flownav_navigator,navdp_agent,cfm_agent,cfmrl_agent],[nomad_fps_writer, flownav_fps_writer, navdp_fps_writer, cfm_fps_writer, cfmrl_fps_writer],robot_orientation,image,depth)
    
    
    dual_sys_output['output_trajectory'] = traj_to_actions(execute_trajectory, use_discrate_action=False)
    
    json_output = {}
    
    json_output['trajectory'] = dual_sys_output['output_trajectory'].tolist()
    if dual_sys_output['output_pixel'] is not None:
        json_output['pixel_goal'] = dual_sys_output['output_pixel'] 

    t1 = time.time()
    generate_time = t1 - t0
    print(f"dual sys step {generate_time}")
    print(f"json_output {json_output}")
    return jsonify(json_output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="./checkpoints_rl_socialnav/rl_checkpoint_update02700.ckpt")
    parser.add_argument("--resize_w", type=int, default=224)
    parser.add_argument("--resize_h", type=int, default=224)
    parser.add_argument("--num_history", type=int, default=8)
    args = parser.parse_args()

    args.camera_intrinsic = np.array(
        [[386.5, 0.0, 328.9, 0.0], [0.0, 386.5, 244, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )

    cfmrl_agent = GRPO_Agent(
            "./cfm_rf_rl/checkpoints_rl_socialnav/rl_checkpoint_update02700.ckpt",
            args.camera_intrinsic ,
            image_size=224,        # 输入图像尺寸
            memory_size=8,         # 历史帧数（时序记忆）
            predict_size=24,       # 预测轨迹点数
            temporal_depth=16,     # Transformer 层数
            heads=8,               # 注意力头数
            token_dim=384,         # Token 维度,  # 模型权重路径
            device='cuda:0'        # GPU 设备
        )
    
    cfm_agent = CFM_Agent(
            args.camera_intrinsic ,
            image_size=224,        # 输入图像尺寸
            memory_size=8,         # 历史帧数（时序记忆）
            predict_size=24,       # 预测轨迹点数
            temporal_depth=16,     # Transformer 层数
            heads=8,               # 注意力头数
            token_dim=384,         # Token 维度
            navi_model="./cfm_rf/checkpoints/checkpoint-11710navdp.ckpt",  # 模型权重路径
            device='cuda:0'        # GPU 设备
        )
    
    navdp_agent = NavDP_Agent(
            args.camera_intrinsic ,
            image_size=224,        # 输入图像尺寸
            memory_size=8,         # 历史帧数（时序记忆）
            predict_size=24,       # 预测轨迹点数
            temporal_depth=16,     # Transformer 层数
            heads=8,               # 注意力头数
            token_dim=384,         # Token 维度
            navi_model="./navdp/checkpoints/navdp-cross-modal.ckpt",  # 模型权重路径
            device='cuda:0'        # GPU 设备
        )
    
    nomad_navigator = NoMaDAgent(args.camera_intrinsic,
                                model_path='/home/justin/GitRepos/NavDP/baselines/nomad/nomad.pth',
                                model_config_path='/home/justin/GitRepos/NavDP/baselines/nomad/configs/nomad.yaml',
                                robot_config_path='/home/justin/GitRepos/NavDP/baselines/nomad/configs/robot_config.yaml',
                                data_config_path='/home/justin/GitRepos/NavDP/baselines/nomad/configs/data_config.yaml',
                                device='cuda:0' )
    
    # lotus_agent = NavDP_Agent_cross_modal(args.camera_intrinsic,
    #                             image_size=224,
    #                             memory_size=8,
    #                             predict_size=24,
    #                             temporal_depth=16,
    #                             heads=8,
    #                             token_dim=384,
    #                             navi_model='/home/justin/GitRepos/NavDP/baselines/lotus/checkpoint-40460navdp.ckpt',
    #                             device='cuda:0')
    
    flownav_navigator = FlowNavAgent(
            image_intrinsic=args.camera_intrinsic,
            model_config_path='./flownav/configs/flownav_navdp.yaml',
            robot_config_path='./flownav/configs/robot_config.yaml',
            device='cuda:0',
            policy_weights_path='/home/justin/GitRepos/NavDP/baselines/flownav/flownav_weights.pth',
            depth_weights_path='/home/justin/GitRepos/NavDP/baselines/flownav/depth_anything_v2_vits.pth',
        )
    flownav_navigator.reset(1)
    
    nomad_navigator.reset(1)
        
    format_time = datetime.fromtimestamp(time.time())
    format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
    navdp_fps_writer = imageio.get_writer("{}_navdp_fps_goal.mp4".format(format_time), fps=3)
    cfmrl_fps_writer = imageio.get_writer("{}_cfmrl_fps_goal.mp4".format(format_time), fps=3)
    flownav_fps_writer = imageio.get_writer("{}_flownav_fps_goal.mp4".format(format_time), fps=3)
    cfm_fps_writer = imageio.get_writer("{}_cfm_fps_goal.mp4".format(format_time), fps=3)
    nomad_fps_writer = imageio.get_writer("{}_nomad_fps_goal.mp4".format(format_time), fps=3)
    # navdp_fps_rgbd_writer = imageio.get_writer("{}_fps_rgbd_nongoal.mp4".format(format_time), fps=3)

    cfmrl_agent.reset(1, -3.0) 
    navdp_agent.reset(1, -3.0)
    cfm_agent.reset(1,-3.0) 
    flownav_navigator.reset(1)
    nomad_navigator.reset(1)

    app.run(host='0.0.0.0', port=5801)
    navdp_fps_writer.close()
    cfmrl_fps_writer.close()
    nomad_fps_writer.close()
    flownav_fps_writer.close()
    cfm_fps_writer.close()
    # navdp_fps_rgbd_writer.close()
    # navdp_depth_fps_writer.close()
    # navdp_trajvis_writer_1.close()
    # navdp_trajvis_writer_2.close()
    print('end')
    
    
