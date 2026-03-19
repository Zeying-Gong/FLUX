"""
NavDP 客户端工具
功能：封装与 NavDP Server 的 HTTP 通信

主要函数：
- navigator_reset: 重置导航Agent
- nogoal_step: 无目标探索推理
- pointgoal_step: 点目标导航推理
- imagegoal_step: 图像目标导航推理

数据编码格式：
- RGB: JPEG 格式
- Depth: PNG 16位（单位：厘米*10000）
- Goal: JSON 格式
"""
import requests
import numpy as np
import cv2
import io
import json
import time

def navigator_reset(intrinsic=None, stop_threshold=-0.5, batch_size=1, port=8888, env_id=None):
    """
    重置导航 Agent
    
    功能：
    1. 全局重置（env_id=None）：初始化 NavDP Agent，设置相机内参和batch大小
    2. 单环境重置（env_id=int）：清空指定环境的历史观测队列
    
    Args:
        intrinsic: 相机内参矩阵 (3x3)，用于轨迹投影
        stop_threshold: 探索转向的停止阈值（Critic价值低于此值时停止前进）
        batch_size: 批次大小（并行环境数）
        port: NavDP Server 端口
        env_id: 环境ID（None表示全局重置，int表示单环境重置）
    
    Returns:
        algo: 算法名称（如 "navdp"）
    """
    print("http://localhost:%d/navigator_reset" % port)
    
    if env_id is None:
        # ===== 全局重置：初始化 Agent =====
        url = "http://localhost:%d/navigator_reset" % port
        response = requests.post(url, json={
            'intrinsic': intrinsic.tolist(),
            'stop_threshold': stop_threshold,
            'batch_size': batch_size
        })
    else:
        # ===== 单环境重置：清空历史队列 =====
        url = "http://localhost:%d/navigator_reset_env" % port
        response = requests.post(url, json={'env_id': env_id})
    
    return json.loads(response.text)['algo']

def nogoal_step(rgb_images,depth_images,port=8888):
    concat_images = np.concatenate([img for img in rgb_images],axis=0)
    concat_depths = np.concatenate([img for img in depth_images],axis=0)
    url = "http://localhost:%d/nogoal_step"%port
    _, rgb_image = cv2.imencode('.jpg', concat_images)
    image_bytes = io.BytesIO()
    image_bytes.write(rgb_image)
    
    depth_image = np.clip(concat_depths*10000.0,0,65535.0).astype(np.uint16)
    _, depth_image = cv2.imencode('.png', depth_image)
    depth_bytes = io.BytesIO()
    depth_bytes.write(depth_image)
    
    files = {
        'image': ('image.jpg', image_bytes.getvalue(), 'image/jpeg'),
        'depth': ('depth.png', depth_bytes.getvalue(), 'image/png'),
    }
    data = {
        'depth_time':time.time(),
        'rgb_time':time.time(),
    }
    response = requests.post(url, files=files, data=data)
    trajectory = json.loads(response.text)['trajectory']
    all_trajectory = json.loads(response.text)['all_trajectory']
    all_value = json.loads(response.text)['all_values']
    return np.array(trajectory),np.array(all_trajectory),np.array(all_value)

def pointgoal_step(point_goals, rgb_images, depth_images, port=8888):
    """
    点目标导航推理
    
    Args:
        point_goals: 目标点坐标 (batch, 2)，[x, y] in camera frame，单位：米
        rgb_images: RGB 图像 (batch, H, W, 3)，值域 [0, 255]
        depth_images: 深度图 (batch, H, W)，单位：米
        port: NavDP Server 端口
    
    Returns:
        trajectory: 最优轨迹 (batch, 24, 3)，[Δx, Δy, Δθ]
        all_trajectory: 所有候选轨迹 (batch, 16, 24, 3)
        all_value: Critic 价值 (batch, 16)
    
    数据编码：
    - RGB: JPEG 压缩
    - Depth: PNG 16位，深度值 = 米 * 10000（厘米*100）
    - Goal: JSON格式
    """
    # ===== 步骤1：拼接 batch 维度 =====
    concat_images = np.concatenate([img for img in rgb_images], axis=0)
    concat_depths = np.concatenate([img for img in depth_images], axis=0)
    
    # ===== 步骤2：编码 RGB（JPEG）=====
    url = "http://localhost:%d/pointgoal_step" % port
    _, rgb_image = cv2.imencode('.jpg', concat_images)
    image_bytes = io.BytesIO()
    image_bytes.write(rgb_image)
    
    # ===== 步骤3：编码 Depth（PNG 16位）=====
    # 深度转换：米 → 厘米*100 → uint16
    # 例如：0.5米 → 5000 → uint16(5000)
    depth_image = np.clip(concat_depths * 10000.0, 0, 65535.0).astype(np.uint16)
    _, depth_image = cv2.imencode('.png', depth_image)
    depth_bytes = io.BytesIO()
    depth_bytes.write(depth_image)
    
    # ===== 步骤4：准备 HTTP 请求 =====
    files = {
        'image': ('image.jpg', image_bytes.getvalue(), 'image/jpeg'),
        'depth': ('depth.png', depth_bytes.getvalue(), 'image/png'),
    }
    data = {
        'goal_data': json.dumps({
            'goal_x': point_goals[:, 0].tolist(),  # 目标 x 坐标列表
            'goal_y': point_goals[:, 1].tolist()   # 目标 y 坐标列表
        }),
        'depth_time': time.time(),
        'rgb_time': time.time(),
    }
    
    # ===== 步骤5：发送请求并解析响应 =====
    response = requests.post(url, files=files, data=data)
    trajectory = json.loads(response.text)['trajectory']
    all_trajectory = json.loads(response.text)['all_trajectory']
    all_value = json.loads(response.text)['all_values']
    
    # 可选返回：子目标（用于调试）
    if 'sub_pointgoal_pd' in json.loads(response.text):
        sub_pointgoal_pd = json.loads(response.text)['sub_pointgoal_pd']
        return np.array(trajectory), np.array(all_trajectory), np.array(all_value), sub_pointgoal_pd
    else:
        return np.array(trajectory), np.array(all_trajectory), np.array(all_value)

def imagegoal_step(image_goals,rgb_images,depth_images,port=8888):
    concat_images = np.concatenate([img for img in rgb_images],axis=0)
    concat_depths = np.concatenate([img for img in depth_images],axis=0)
    concat_goals = np.concatenate([img for img in image_goals],axis=0)
    
    url = "http://localhost:%d/imagegoal_step"%port
    _, rgb_image = cv2.imencode('.jpg', concat_images)
    image_bytes = io.BytesIO()
    image_bytes.write(rgb_image)
    
    _, goal_image = cv2.imencode('.jpg', concat_goals)
    goal_bytes = io.BytesIO()
    goal_bytes.write(goal_image)
    
    depth_image = np.clip(concat_depths*10000.0,0,65535.0).astype(np.uint16)
    _, depth_image = cv2.imencode('.png', depth_image)
    depth_bytes = io.BytesIO()
    depth_bytes.write(depth_image)
    
    files = {
        'image': ('image.jpg', image_bytes.getvalue(), 'image/jpeg'),
        'goal': ('goal.jpg', goal_bytes.getvalue(), 'image/jpeg'),
        'depth': ('depth.png', depth_bytes.getvalue(), 'image/png'),
    }
    data = {
        'depth_time':time.time(),
        'rgb_time':time.time(),
    }
    response = requests.post(url, files=files, data=data)
    trajectory = json.loads(response.text)['trajectory']
    all_trajectory = json.loads(response.text)['all_trajectory']
    all_value = json.loads(response.text)['all_values']
    return np.array(trajectory),np.array(all_trajectory),np.array(all_value)





