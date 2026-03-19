"""
NavDP 导航模型 HTTP 服务器
提供导航推理的 Flask API 接口
"""
from PIL import Image
from flask import Flask, request, jsonify
from policy_agent import NavDP_Agent
import numpy as np
import cv2
import imageio
import time
import datetime
import json
import os

from PIL import Image, ImageDraw, ImageFont
import argparse

# ============ 解析命令行参数 ============
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8888)  # 服务器端口号
parser.add_argument("--checkpoint", type=str, default="checkpoints/navdp-cross-modal.ckpt")  # 模型权重路径
args = parser.parse_known_args()[0]

# ============ 创建 Flask 应用 ============
app = Flask(__name__)

# ============ 全局变量 ============
navdp_navigator = None  # NavDP Agent 对象（模型推理接口），首次请求时初始化
navdp_fps_writer = None  # 视频写入器，用于保存可视化结果

@app.route("/navigator_reset", methods=['POST'])
def navdp_reset():
    """
    初始化或重置导航器
    这是客户端第一次调用的接口，用于创建 NavDP Agent
    
    输入 JSON:
        - intrinsic: 相机内参矩阵 (3x3)
        - stop_threshold: 停止阈值，当所有轨迹价值低于此值时停止前进
        - batch_size: 批次大小（并行环境数）
    
    返回:
        {"algo": "navdp"}
    """
    global navdp_navigator, navdp_fps_writer
    
    # ===== 接收参数 =====
    intrinsic = np.array(request.get_json().get('intrinsic'))  # 相机内参
    threshold = np.array(request.get_json().get('stop_threshold'))  # 停止阈值
    batchsize = np.array(request.get_json().get('batch_size'))  # 批次大小
    
    # ===== 创建或重置 NavDP Agent =====
    if navdp_navigator is None:
        # 第一次调用，创建 NavDP Agent 并加载模型权重
        navdp_navigator = NavDP_Agent(
            intrinsic,
            image_size=224,        # 输入图像尺寸
            memory_size=8,         # 历史帧数（时序记忆）
            predict_size=24,       # 预测轨迹点数
            temporal_depth=16,     # Transformer 层数
            heads=8,               # 注意力头数
            token_dim=384,         # Token 维度
            navi_model=args.checkpoint,  # 模型权重路径
            device='cuda:0'        # GPU 设备
        )
        navdp_navigator.reset(batchsize, threshold)  # 重置状态（清空历史队列）
    else:
        # 后续调用，只重置状态
        navdp_navigator.reset(batchsize, threshold)
    
    # ===== 创建或重新创建视频写入器 =====
    if navdp_fps_writer is None:
        # 第一次创建视频写入器
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        navdp_fps_writer = imageio.get_writer("{}_fps_pointgoal.mp4".format(format_time), fps=7)
    else:
        # 已有视频写入器，关闭旧的，创建新的
        navdp_fps_writer.close()
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        navdp_fps_writer = imageio.get_writer("{}_fps_pointgoal.mp4".format(format_time), fps=7)
    
    return jsonify({"algo": "navdp"})

@app.route("/navigator_reset_env", methods=['POST'])
def navdp_reset_env():
    """
    重置单个环境的导航器状态
    当某个环境的 episode 结束时调用，只清空该环境的历史帧队列
    
    输入 JSON:
        - env_id: 环境索引
    
    返回:
        {"algo": "navdp"}
    """
    global navdp_navigator
    navdp_navigator.reset_env(int(request.get_json().get('env_id')))  # 清空指定环境的历史队列
    return jsonify({"algo": "navdp"})

@app.route("/pointgoal_step", methods=['POST'])
def navdp_step_xy():
    """
    点目标导航推理接口
    给定当前 RGB-D 观测和目标点坐标，输出导航轨迹
    
    输入:
        - files['image']: RGB 图像 (JPEG格式)
        - files['depth']: 深度图 (PNG格式，16位，单位：厘米*10000)
        - form['goal_data']: JSON，包含 goal_x, goal_y (相对坐标，单位：米)
    
    返回 JSON:
        - trajectory: 最优轨迹 (batch, 24, 3)
        - all_trajectory: 所有16条候选轨迹 (batch, 16, 24, 3)
        - all_values: 所有16条轨迹的价值 (batch, 16)
    """
    global navdp_navigator, navdp_fps_writer
    start_time = time.time()
    
    # ===== 接收输入数据 =====
    image_file = request.files['image']  # RGB 图像文件
    depth_file = request.files['depth']  # 深度图文件
    goal_data = json.loads(request.form.get('goal_data'))  # 目标点数据
    goal_x = np.array(goal_data['goal_x'])  # 目标 x 坐标
    goal_y = np.array(goal_data['goal_y'])  # 目标 y 坐标
    goal = np.stack((goal_x, goal_y, np.zeros_like(goal_x)), axis=1)  # 拼成 (batch, 3)，z=0
    batch_size = navdp_navigator.batch_size
    
    phase1_time = time.time()
    
    # ===== 解码 RGB 图像 =====
    image = Image.open(image_file.stream)  # 从字节流打开
    image = image.convert('RGB')  # 确保是 RGB 格式
    image = np.asarray(image)  # 转 numpy 数组
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # PIL是RGB，OpenCV是BGR
    image = image.reshape((batch_size, -1, image.shape[1], 3))  # reshape 成 (batch, H, W, 3)
    
    # ===== 解码深度图 =====
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')  # 转成 32位整数
    depth = np.asarray(depth)[:, :, np.newaxis]  # 添加通道维度
    depth = depth.astype(np.float32) / 10000.0  # 从 uint16 转回米（发送时乘了10000）
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))  # reshape 成 (batch, H, W, 1)
    
    phase2_time = time.time()
    
    # ===== 核心推理：调用 NavDP Agent =====
    execute_trajectory, all_trajectory, all_values, trajectory_mask, trajectory_depth_mask = \
        navdp_navigator.step_pointgoal(goal, image, depth)
    # execute_trajectory: (batch, 24, 3) 最优轨迹
    # all_trajectory: (batch, 16, 24, 3) 所有候选轨迹
    # all_values: (batch, 16) 价值评分
    # trajectory_mask: 可视化图像
    
    phase3_time = time.time()
    
    # ===== 保存可视化视频 =====
    navdp_fps_writer.append_data(trajectory_mask)
    
    phase4_time = time.time()
    
    # ===== 打印各阶段耗时 =====
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f" % (
        phase1_time - start_time,   # 数据接收和解码
        phase2_time - phase1_time,  # （无额外操作）
        phase3_time - phase2_time,  # 模型推理
        phase4_time - phase3_time,  # 保存视频
        time.time() - start_time    # 总耗时
    ))
    
    # ===== 返回结果 =====
    return jsonify({
        'trajectory': execute_trajectory.tolist(),      # 最优轨迹
        'all_trajectory': all_trajectory.tolist(),      # 所有候选
        'all_values': all_values.tolist()               # 所有价值
    })


@app.route("/pixelgoal_step", methods=['POST'])
def navdp_step_pixel():
    """
    像素目标导航推理接口
    给定当前 RGB-D 观测和图像中的目标像素坐标，输出导航轨迹
    
    输入:
        - files['image']: RGB 图像 (JPEG格式)
        - files['depth']: 深度图 (PNG格式，16位)
        - form['goal_data']: JSON，包含 goal_x, goal_y (图像坐标，单位：像素)
    
    返回 JSON:
        - trajectory: 最优轨迹 (batch, 24, 3)
        - all_trajectory: 所有16条候选轨迹 (batch, 16, 24, 3)
        - all_values: 所有16条轨迹的价值 (batch, 16)
    """
    global navdp_navigator, navdp_fps_writer
    
    start_time = time.time()
    
    # ===== 接收输入数据 =====
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])  # 目标像素 x 坐标
    goal_y = np.array(goal_data['goal_y'])  # 目标像素 y 坐标
    goal = np.stack((goal_x, goal_y), axis=1)  # 拼成 (batch, 2)
    batch_size = navdp_navigator.batch_size
    
    phase1_time = time.time()
    
    # ===== 解码 RGB 图像 =====
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    # ===== 解码深度图 =====
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:, :, np.newaxis]
    depth = depth.astype(np.float32) / 10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    phase2_time = time.time()
    
    # ===== 核心推理：像素目标导航 =====
    execute_trajectory, all_trajectory, all_values, trajectory_mask = \
        navdp_navigator.step_pixelgoal(goal, image, depth)
    # goal: 像素坐标 (x, y)
    
    phase3_time = time.time()
    
    # ===== 保存可视化视频 =====
    navdp_fps_writer.append_data(trajectory_mask)
    
    phase4_time = time.time()
    
    # ===== 打印耗时 =====
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f" % (
        phase1_time - start_time,
        phase2_time - phase1_time,
        phase3_time - phase2_time,
        phase4_time - phase3_time,
        time.time() - start_time
    ))
    
    return jsonify({
        'trajectory': execute_trajectory.tolist(),
        'all_trajectory': all_trajectory.tolist(),
        'all_values': all_values.tolist()
    })

@app.route("/imagegoal_step", methods=['POST'])
def navdp_step_image():
    """
    图像目标导航推理接口
    给定当前 RGB-D 观测和目标位置的 RGB 图像，输出导航轨迹
    
    输入:
        - files['image']: 当前 RGB 图像 (JPEG格式)
        - files['depth']: 当前深度图 (PNG格式，16位)
        - files['goal']: 目标位置的 RGB 图像 (JPEG格式)
    
    返回 JSON:
        - trajectory: 最优轨迹 (batch, 24, 3)
        - all_trajectory: 所有16条候选轨迹 (batch, 16, 24, 3)
        - all_values: 所有16条轨迹的价值 (batch, 16)
    """
    global navdp_navigator, navdp_fps_writer
    start_time = time.time()
    
    # ===== 接收输入数据 =====
    image_file = request.files['image']  # 当前观测图像
    depth_file = request.files['depth']  # 当前深度图
    goal_file = request.files['goal']    # 目标图像
    batch_size = navdp_navigator.batch_size
    
    phase1_time = time.time()
    
    # ===== 解码当前 RGB 图像 =====
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("image.jpg", image)  # 保存用于调试
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    # ===== 解码目标 RGB 图像 =====
    goal = Image.open(goal_file.stream)
    goal = goal.convert('RGB')
    goal = np.asarray(goal)
    goal = cv2.cvtColor(goal, cv2.COLOR_RGB2BGR)
    cv2.imwrite("goal.jpg", goal)  # 保存用于调试
    goal = goal.reshape((batch_size, -1, goal.shape[1], 3))
    
    # ===== 解码深度图 =====
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:, :, np.newaxis]
    depth = depth.astype(np.float32) / 10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    phase2_time = time.time()
    
    # ===== 核心推理：图像目标导航 =====
    execute_trajectory, all_trajectory, all_values, trajectory_mask = \
        navdp_navigator.step_imagegoal(goal, image, depth)
    # goal: 目标图像
    # image: 当前图像
    # depth: 当前深度
    
    phase3_time = time.time()
    
    # ===== 保存可视化视频 =====
    navdp_fps_writer.append_data(trajectory_mask)
    
    phase4_time = time.time()
    
    # ===== 打印耗时 =====
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f" % (
        phase1_time - start_time,
        phase2_time - phase1_time,
        phase3_time - phase2_time,
        phase4_time - phase3_time,
        time.time() - start_time
    ))
    
    return jsonify({
        'trajectory': execute_trajectory.tolist(),
        'all_trajectory': all_trajectory.tolist(),
        'all_values': all_values.tolist()
    })

@app.route("/nogoal_step", methods=['POST'])
def navdp_step_nogoal():
    """
    无目标探索推理接口
    只给定当前 RGB-D 观测，自主探索（无明确目标）
    
    输入:
        - files['image']: RGB 图像 (JPEG格式)
        - files['depth']: 深度图 (PNG格式，16位)
    
    返回 JSON:
        - trajectory: 最优轨迹 (batch, 24, 3)
        - all_trajectory: 所有16条候选轨迹 (batch, 16, 24, 3)
        - all_values: 所有16条轨迹的价值 (batch, 16)
    """
    global navdp_navigator, navdp_fps_writer
    start_time = time.time()
    
    # ===== 接收输入数据 =====
    image_file = request.files['image']
    depth_file = request.files['depth']
    batch_size = navdp_navigator.batch_size
    
    phase1_time = time.time()
    
    # ===== 解码 RGB 图像 =====
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    # ===== 解码深度图 =====
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:, :, np.newaxis]
    depth = depth.astype(np.float32) / 10000.0  # 转回米
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    phase2_time = time.time()
    
    # ===== 核心推理：无目标探索 =====
    execute_trajectory, all_trajectory, all_values, trajectory_mask, trajectory_depth_mask = \
        navdp_navigator.step_nogoal(image, depth)
    # 注意：无目标任务不需要 goal 输入，目标编码为全0
    
    phase3_time = time.time()
    
    # ===== 保存可视化视频 =====
    navdp_fps_writer.append_data(trajectory_mask)
    
    phase4_time = time.time()
    
    # ===== 打印耗时 =====
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f" % (
        phase1_time - start_time,
        phase2_time - phase1_time,
        phase3_time - phase2_time,
        phase4_time - phase3_time,
        time.time() - start_time
    ))
    
    return jsonify({
        'trajectory': execute_trajectory.tolist(),
        'all_trajectory': all_trajectory.tolist(),
        'all_values': all_values.tolist()
    })

@app.route("/navdp_step_ip_mixgoal", methods=['POST'])
def navdp_step_ip_mixgoal():
    """
    混合目标导航推理接口（点目标 + 图像目标）
    同时给定点目标坐标和目标图像，融合两种目标信息进行导航
    
    输入:
        - files['image']: 当前 RGB 图像 (JPEG格式)
        - files['depth']: 当前深度图 (PNG格式，16位)
        - files['image_goal']: 目标位置的 RGB 图像 (JPEG格式)
        - form['goal_data']: JSON，包含 goal_x, goal_y (点目标坐标，单位：米)
    
    返回 JSON:
        - trajectory: 最优轨迹 (batch, 24, 3)
        - all_trajectory: 所有16条候选轨迹 (batch, 16, 24, 3)
        - all_values: 所有16条轨迹的价值 (batch, 16)
    """
    global navdp_navigator, navdp_fps_writer
    start_time = time.time()
    
    # ===== 接收输入数据 =====
    image_file = request.files['image']
    depth_file = request.files['depth']
    batch_size = navdp_navigator.batch_size
    
    # ===== 解析点目标 =====
    point_goal_data = json.loads(request.form.get('goal_data'))
    point_goal_x = np.array(point_goal_data['goal_x'])
    point_goal_y = np.array(point_goal_data['goal_y'])
    point_goal = np.stack((point_goal_x, point_goal_y, np.zeros_like(point_goal_x)), axis=1)  # (batch, 3)
    
    # ===== 解析图像目标 =====
    image_goal_file = request.files['image_goal']
    image_goal = Image.open(image_goal_file.stream)
    image_goal = image_goal.convert('RGB')
    image_goal = np.asarray(image_goal)
    image_goal = cv2.cvtColor(image_goal, cv2.COLOR_RGB2BGR)
    cv2.imwrite("goal.jpg", image_goal)
    image_goal = image_goal.reshape((batch_size, -1, image_goal.shape[1], 3))
    
    phase1_time = time.time()
    
    # ===== 解码当前观测 =====
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:, :, np.newaxis]
    depth = depth.astype(np.float32) / 10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    phase2_time = time.time()
    
    # ===== 核心推理：混合目标导航 =====
    execute_trajectory, all_trajectory, all_values, trajectory_mask = \
        navdp_navigator.step_point_image_goal(point_goal, image_goal, image, depth)
    # point_goal: 点目标坐标
    # image_goal: 图像目标
    # 两种目标会被分配到不同的 goal slot 中
    
    phase3_time = time.time()
    
    # ===== 保存可视化视频 =====
    navdp_fps_writer.append_data(trajectory_mask)
    
    phase4_time = time.time()
    
    # ===== 打印耗时 =====
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f" % (
        phase1_time - start_time,
        phase2_time - phase1_time,
        phase3_time - phase2_time,
        phase4_time - phase3_time,
        time.time() - start_time
    ))
    
    return jsonify({
        'trajectory': execute_trajectory.tolist(),
        'all_trajectory': all_trajectory.tolist(),
        'all_values': all_values.tolist()
    })
    

if __name__ == "__main__":
    # ===== 启动 Flask 服务器 =====
    # host='127.0.0.1' 表示只监听本地连接
    # port 从命令行参数获取，默认 8888
    app.run(host='127.0.0.1', port=args.port)