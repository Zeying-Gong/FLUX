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

from policy_agent import NavDP_Agent
import cv2
import imageio

from PIL import Image, ImageDraw, ImageFont
from visualization_utils import VisualizationManager
import math

app = Flask(__name__)
idx = 0
start_time = time.time()
output_dir = ''


# ============ 全局变量 ============
navdp_fps_writer = None  # 视频写入器，用于保存可视化结果
# navdp_depth_fps_writer = None  # 视频写入器，用于保存可视化结果
# navdp_trajvis_writer_1 = None  # 视频写入器，用于保存可视化结果
# navdp_trajvis_writer_2 = None  # 视频写入器，用于保存可视化结果
vis_manager = VisualizationManager(history_size=5)
goal_position = [4,0,0]
first = True

navdp_fps_rgbd_writer = None 



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
    
@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    global idx, output_dir, start_time, navdp_fps_writer, navdp_depth_fps_writer, vis_manager,goal_position, first
    
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
    batch_size = agent.batch_size
    
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
        # agent.reset()
        agent.reset(1, -3.0)

    idx += 1

    look_down = False
    t0 = time.time()
    dual_sys_output = {'output_trajectory':None,'output_pixel':None}

    # point_goal = np.array([[goal_position[0] - robot_orientation[0],
    #     goal_position[1] - robot_orientation[1],0]])
    
    dx = goal_position[0] - robot_orientation[0]
    dy = goal_position[1] - robot_orientation[1]
    yaw = robot_orientation[2]
    print(f"robot yaw {yaw}")

    # 旋转到机器人局部坐标系
    local_x =  dx * math.cos(yaw) + dy * math.sin(yaw)
    local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)

    point_goal = np.array([[local_x, local_y, 0]])
    
    print('point_goal', point_goal, 'goal position', goal_position)
    depth_image_clipped = np.clip(depth[0], 0.1, 3)
            
    # 归一化到[0, 1]范围
    depth_image_normalized = (depth_image_clipped - 0.1) / (3 - 0.1)  # 线性归一化
    
    # 将深度图像转换为RGB格式
    depth_image_rgb = cv2.cvtColor((depth_image_normalized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # print(f"depth {depth.shape} {depth_image_normalized.shape} depth_image_normalized {depth_image_rgb.shape} image[0] {image[0].shape}")
    rgbd = np.concatenate((image[0], depth_image_rgb), axis=0)
    
    execute_trajectory, all_trajectory, all_values, trajectory_mask, trajectory_depth_mask = \
        agent.step_pointgoal(point_goal, image, depth)
        # agent.step_nogoal(image, depth) 
        # agent.step_pointgoal(point_goal, image, depth) #agent.step_nogoal(image, depth) # agent.step_pointgoal(goal, image, depth)
    # print('execute_trajectory', execute_trajectory.shape, 'all_trajectory', all_trajectory.shape)
        
    
    
    # camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
    #     camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
    #     camera_rot_quat = camera_rot_quat[:,[1, 2, 3, 0]]
    #     camera_rot = R.from_quat(camera_rot_quat).as_matrix()
    # robot_vel = env.unwrapped.scene.articulations['robot'].data.root_lin_vel_w[0, :2].norm().cpu().numpy()
    #     robot_ang_vel = env.unwrapped.scene.articulations['robot'].data.root_ang_vel_w[0, 2].cpu().numpy()
    # x0 = np.stack([camera_pos[:,0], camera_pos[:,1], np.arctan2(camera_rot[:,1,0], camera_rot[:,0,0]), [robot_vel], [robot_ang_vel]],axis=-1)
    
    vis_resized, vis_resized_all = vis_manager.visualize_trajectory(
                    image[0], depth[0], np.array(
        [[386.5, 0.0, 328.9, 0.0], [0.0, 386.5, 244, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    ),
                    execute_trajectory[0],
                    robot_pose=robot_orientation,
                    all_trajectories_points=all_trajectory[0],
                    all_trajectories_values=all_values[0],
                    goal_pos=point_goal[0]
                )
    # vis_resized, vis_resized_all = vis_manager.visualize_trajectory_global_with_people(
    #                 image[0], depth[0], np.array(
    #     [[386.5, 0.0, 328.9, 0.0], [0.0, 386.5, 244, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    # ),
    #                 execute_trajectory[0],
    #                 robot_pose=robot_orientation,
    #                 all_trajectories_points=all_trajectory[0],
    #                 all_trajectories_values=all_values[0],
    #                 goal_position=point_goal[0]
    #             )
    
    exe_rgb = np.concatenate((trajectory_mask, vis_resized), axis=1)
    print("trajectory_depth_mask", trajectory_depth_mask.shape, "vis_resized_all",vis_resized_all.shape, 'vis_resized', vis_resized.shape)
    all_depth = np.concatenate((trajectory_depth_mask, vis_resized_all), axis=1)
    all_vis = np.concatenate((exe_rgb, all_depth), axis=0)
    
    # all_vis = all_vis[..., ::-1]
    image = Image.fromarray(all_vis, 'RGB')  # 使用 RGBA 模式

    # 保存为 PNG 图像
    image.save('./output_image.png')
    
    navdp_fps_writer.append_data(all_vis)
    navdp_fps_rgbd_writer.append_data(rgbd)
    # navdp_depth_fps_writer.append_data(trajectory_depth_mask)
    
    # print("vis_resized",vis_resized.shape)
    # navdp_trajvis_writer_1.append_data(vis_resized)
    # navdp_trajvis_writer_2.append_data(vis_resized_all)
    
    dual_sys_output['output_trajectory'] = traj_to_actions(execute_trajectory, use_discrate_action=False)
    
    json_output = {}
    
    json_output['trajectory'] = dual_sys_output['output_trajectory'].tolist()
    if dual_sys_output['output_pixel'] is not None:
        json_output['pixel_goal'] = dual_sys_output['output_pixel'] 

    t1 = time.time()
    generate_time = t1 - t0
    print(f"dual sys step {generate_time}")
    print(f"json_output {json_output}")
    # return None
    return jsonify(json_output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="./checkpoints/navdp-cross-modal.ckpt")
    parser.add_argument("--resize_w", type=int, default=224)
    parser.add_argument("--resize_h", type=int, default=224)
    parser.add_argument("--num_history", type=int, default=8)
    args = parser.parse_args()

    args.camera_intrinsic = np.array(
        [[386.5, 0.0, 328.9, 0.0], [0.0, 386.5, 244, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    # agent = InternVLAN1AsyncAgent(args)
    agent = NavDP_Agent(
            args.camera_intrinsic ,
            image_size=224,        # 输入图像尺寸
            memory_size=8,         # 历史帧数（时序记忆）
            predict_size=24,       # 预测轨迹点数
            temporal_depth=16,     # Transformer 层数
            heads=8,               # 注意力头数
            token_dim=384,         # Token 维度
            navi_model=args.model_path,  # 模型权重路径
            device='cuda:0'        # GPU 设备
        )
    
    format_time = datetime.fromtimestamp(time.time())
    format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
    navdp_fps_writer = imageio.get_writer("{}_fps_nongoal.mp4".format(format_time), fps=3)
    navdp_fps_rgbd_writer = imageio.get_writer("{}_fps_rgbd_nongoal.mp4".format(format_time), fps=3)
    # navdp_depth_fps_writer = imageio.get_writer("{}_depth_fps_nongoal.mp4".format(format_time), fps=7)
    # navdp_trajvis_writer_1 = imageio.get_writer("{}_singletraj_fps_nongoal.mp4".format(format_time), fps=7)
    # navdp_trajvis_writer_2 = imageio.get_writer("{}_alltraj_fps_nongoal.mp4".format(format_time), fps=7)
    # agent.step(
    #     np.zeros((480, 640, 3)),
    #     np.zeros((480, 640)),
    #     np.eye(4),
    #     "hello",
    # )
    agent.reset(1, -3.0) 

    app.run(host='0.0.0.0', port=5801)
    navdp_fps_writer.close()
    navdp_fps_rgbd_writer.close()
    # navdp_depth_fps_writer.close()
    # navdp_trajvis_writer_1.close()
    # navdp_trajvis_writer_2.close()
    print('end')
    
    
