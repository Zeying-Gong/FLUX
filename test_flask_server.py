import requests
import numpy as np
from PIL import Image
import io
import json

PORT = 8888
# 服务器配置
SERVER_URL = f"http://localhost:{PORT}"

def create_dummy_image(height=480, width=640, batch_size=1):
    """创建测试用的RGB图像"""
    img_array = np.random.randint(0, 255, (height * batch_size, width, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='RGB')
    return img

def create_dummy_depth(height=480, width=640, batch_size=1):
    """创建测试用的深度图像 (单通道，uint16格式)"""
    # 深度值范围 0-50000 (对应 0-5米，因为除以10000)
    depth_array = np.random.randint(5000, 30000, (height * batch_size, width), dtype=np.uint16)
    img = Image.fromarray(depth_array, mode='I;16')
    return img

def test_navigator_reset():
    """测试 /navigator_reset 接口"""
    print("=" * 50)
    print("测试 navigator_reset 接口...")
    
    # 构造相机内参矩阵 (3x3)
    intrinsic = [[320, 0, 320],
                 [0, 320, 240],
                 [0, 0, 1]]
    
    payload = {
        'intrinsic': intrinsic,
        'stop_threshold': 0.5,
        'batch_size': 1
    }
    
    try:
        response = requests.post(f"{SERVER_URL}/navigator_reset", json=payload)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_navigator_reset_env(env_id=0):
    """测试 /navigator_reset_env 接口"""
    print("=" * 50)
    print(f"测试 navigator_reset_env 接口 (env_id={env_id})...")
    
    payload = {
        'env_id': env_id  # 添加必需的参数
    }
    
    try:
        response = requests.post(f"{SERVER_URL}/navigator_reset_env", json=payload)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_pointgoal_step(batch_size=1):
    """测试 /pointgoal_step 接口"""
    print("=" * 50)
    print(f"测试 pointgoal_step 接口 (batch_size={batch_size})...")
    
    # 创建测试图像
    rgb_image = create_dummy_image(batch_size=batch_size)
    depth_image = create_dummy_depth(batch_size=batch_size)
    
    # 保存到内存中的字节流
    rgb_bytes = io.BytesIO()
    rgb_image.save(rgb_bytes, format='PNG')
    rgb_bytes.seek(0)
    
    depth_bytes = io.BytesIO()
    depth_image.save(depth_bytes, format='PNG')
    depth_bytes.seek(0)
    
    # 构造目标点数据
    goal_data = {
        'goal_x': [2.5] * batch_size,  # x方向2.5米
        'goal_y': [1.0] * batch_size   # y方向1.0米
    }
    
    # 准备文件和表单数据
    files = {
        'image': ('test_rgb.png', rgb_bytes, 'image/png'),
        'depth': ('test_depth.png', depth_bytes, 'image/png')
    }
    
    data = {
        'goal_data': json.dumps(goal_data)
    }
    
    try:
        response = requests.post(f"{SERVER_URL}/pointgoal_step", files=files, data=data)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"返回键: {result.keys()}")
            print(f"trajectory shape: {np.array(result['trajectory']).shape}")
            print(f"all_trajectory shape: {np.array(result['all_trajectory']).shape}")
            print(f"all_values shape: {np.array(result['all_values']).shape}")
            print(f"trajectory 示例: {np.array(result['trajectory'])[:3]}")
        else:
            print(f"错误响应: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_nogoal_step(batch_size=1):
    """测试 /nogoal_step 接口 (新增)"""
    print("=" * 50)
    print(f"测试 nogoal_step 接口 (batch_size={batch_size})...")
    
    # 创建测试图像
    rgb_image = create_dummy_image(batch_size=batch_size)
    depth_image = create_dummy_depth(batch_size=batch_size)
    
    # 保存到内存中的字节流
    rgb_bytes = io.BytesIO()
    rgb_image.save(rgb_bytes, format='PNG')
    rgb_bytes.seek(0)
    
    depth_bytes = io.BytesIO()
    depth_image.save(depth_bytes, format='PNG')
    depth_bytes.seek(0)
    
    # 准备文件（nogoal 不需要 goal_data）
    files = {
        'image': ('test_rgb.png', rgb_bytes, 'image/png'),
        'depth': ('test_depth.png', depth_bytes, 'image/png')
    }
    
    try:
        response = requests.post(f"{SERVER_URL}/nogoal_step", files=files)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"返回键: {result.keys()}")
            print(f"trajectory shape: {np.array(result['trajectory']).shape}")
            print(f"all_trajectory shape: {np.array(result['all_trajectory']).shape}")
            print(f"all_values shape: {np.array(result['all_values']).shape}")
            print(f"trajectory 示例: {np.array(result['trajectory'])[:3]}")
            
            # 检查轨迹是否合理
            trajectory = np.array(result['trajectory'])
            if trajectory.shape[0] > 0:
                print(f"✅ 生成了 {trajectory.shape[1]} 个轨迹点")
                print(f"   轨迹范围: x=[{trajectory[0,:,0].min():.2f}, {trajectory[0,:,0].max():.2f}], "
                      f"y=[{trajectory[0,:,1].min():.2f}, {trajectory[0,:,1].max():.2f}]")
            else:
                print("⚠️ 未生成轨迹点")
        else:
            print(f"错误响应: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imagegoal_step(batch_size=1):
    """测试 /imagegoal_step 接口 (可选，如果你有的话)"""
    print("=" * 50)
    print(f"测试 imagegoal_step 接口 (batch_size={batch_size})...")
    
    # 创建测试图像
    rgb_image = create_dummy_image(batch_size=batch_size)
    depth_image = create_dummy_depth(batch_size=batch_size)
    goal_image = create_dummy_image(batch_size=batch_size)  # 目标图像
    
    # 保存到内存中的字节流
    rgb_bytes = io.BytesIO()
    rgb_image.save(rgb_bytes, format='PNG')
    rgb_bytes.seek(0)
    
    depth_bytes = io.BytesIO()
    depth_image.save(depth_bytes, format='PNG')
    depth_bytes.seek(0)
    
    goal_bytes = io.BytesIO()
    goal_image.save(goal_bytes, format='PNG')
    goal_bytes.seek(0)
    
    # 准备文件
    files = {
        'image': ('test_rgb.png', rgb_bytes, 'image/png'),
        'depth': ('test_depth.png', depth_bytes, 'image/png'),
        'goal_image': ('test_goal.png', goal_bytes, 'image/png')
    }
    
    try:
        response = requests.post(f"{SERVER_URL}/imagegoal_step", files=files)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"返回键: {result.keys()}")
            print(f"trajectory shape: {np.array(result['trajectory']).shape}")
            print(f"all_trajectory shape: {np.array(result['all_trajectory']).shape}")
            print(f"all_values shape: {np.array(result['all_values']).shape}")
        else:
            print(f"错误响应: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_batch_processing():
    """测试批处理 (多个环境)"""
    print("\n" + "=" * 50)
    print("测试批处理能力...")
    
    batch_sizes = [1, 2, 4]
    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")
        
        # 测试 nogoal_step 的批处理
        success = test_nogoal_step(batch_size=batch_size)
        if not success:
            print(f"⚠️ Batch size {batch_size} 测试失败")
            return False
    
    return True

def main():
    print("开始测试 IPlanner Navigation Server")
    print(f"服务器地址: {SERVER_URL}\n")
    
    # 测试顺序
    test_results = {}
    
    # 1. 重置导航器
    print("\n[测试 1/5] navigator_reset")
    test_results['navigator_reset'] = test_navigator_reset()
    
    # 2. 重置环境
    print("\n[测试 2/5] navigator_reset_env")
    test_results['navigator_reset_env'] = test_navigator_reset_env()
    
    # 3. 测试点目标导航
    print("\n[测试 3/5] pointgoal_step")
    test_results['pointgoal_step'] = test_pointgoal_step(batch_size=1)
    
    # 4. 测试无目标探索导航 (新增)
    print("\n[测试 4/5] nogoal_step")
    test_results['nogoal_step'] = test_nogoal_step(batch_size=1)
    
    # 5. 测试批处理
    # print("\n[测试 5/5] batch_processing")
    # test_results['batch_processing'] = test_batch_processing()
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print("=" * 50)
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:25s}: {status}")
    
    print("=" * 50)
    
    all_passed = all(test_results.values())
    if all_passed:
        print("✅ 所有测试通过!")
    else:
        failed_tests = [name for name, result in test_results.items() if not result]
        print(f"❌ {len(failed_tests)} 个测试失败: {', '.join(failed_tests)}")
        print("\n请检查:")
        print("  1. 服务器是否正常运行")
        print("  2. 模型是否正确加载")
        print("  3. 服务器日志中的错误信息")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)