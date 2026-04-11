import argparse
import io
import json

import numpy as np
import requests
from PIL import Image

# Overwritten in main() from CLI --port (default 9999)
SERVER_URL = "http://localhost:9999"


# ============================================================
# 工具函数
# ============================================================

def create_dummy_image(height=480, width=640, batch_size=1):
    """创建测试用的RGB图像（垂直拼接 batch）"""
    img_array = np.random.randint(0, 255, (height * batch_size, width, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='RGB')
    return img


def create_dummy_depth(height=480, width=640, batch_size=1):
    """创建测试用的深度图（单通道 uint16，值域 0-65535 对应 0-6.5m）"""
    depth_array = np.random.randint(5000, 30000, (height * batch_size, width), dtype=np.uint16)
    img = Image.fromarray(depth_array, mode='I;16')
    return img


def _encode_rgb(pil_image: Image.Image, fmt='JPEG') -> io.BytesIO:
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    buf.seek(0)
    return buf


def _encode_depth_png(pil_image: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    pil_image.save(buf, format='PNG')
    buf.seek(0)
    return buf


def _print_trajectory_info(result: dict):
    traj = np.array(result['trajectory'])
    all_traj = np.array(result['all_trajectory'])
    all_val = np.array(result['all_values'])
    print(f"  trajectory      shape : {traj.shape}")
    print(f"  all_trajectory  shape : {all_traj.shape}")
    print(f"  all_values      shape : {all_val.shape}")
    if traj.ndim >= 2 and traj.shape[0] > 0:
        pts = traj[0] if traj.ndim == 3 else traj
        print(f"  trajectory 前3点      : {pts[:3]}")
        print(f"  轨迹 x 范围: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]  "
              f"y 范围: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]")


# ============================================================
# 单项测试
# ============================================================

def test_navigator_reset():
    """测试 /navigator_reset —— 全局初始化"""
    print("=" * 55)
    print("测试 navigator_reset ...")

    intrinsic = [[320, 0, 320],
                 [0, 320, 240],
                 [0, 0, 1]]

    payload = {
        'intrinsic': intrinsic,
        'stop_threshold': 0.5,
        'batch_size': 1
    }

    try:
        resp = requests.post(f"{SERVER_URL}/navigator_reset", json=payload, timeout=10)
        print(f"  状态码 : {resp.status_code}")
        print(f"  响应   : {resp.json()}")
        return resp.status_code == 200
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return False


def test_navigator_reset_env(env_id: int = 0):
    """测试 /navigator_reset_env —— 单环境 reset"""
    print("=" * 55)
    print(f"测试 navigator_reset_env (env_id={env_id}) ...")

    payload = {'env_id': env_id}

    try:
        resp = requests.post(f"{SERVER_URL}/navigator_reset_env", json=payload, timeout=10)
        print(f"  状态码 : {resp.status_code}")
        print(f"  响应   : {resp.json()}")
        return resp.status_code == 200
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return False


def test_pointgoal_step(batch_size: int = 1):
    """测试 /pointgoal_step"""
    print("=" * 55)
    print(f"测试 pointgoal_step (batch_size={batch_size}) ...")

    rgb_buf   = _encode_rgb(_encode_rgb.__class__ and create_dummy_image(batch_size=batch_size), 'PNG')
    depth_buf = _encode_depth_png(create_dummy_depth(batch_size=batch_size))

    # 重新生成，上面写法有误
    rgb_buf   = _encode_rgb(create_dummy_image(batch_size=batch_size), 'PNG')

    goal_data = {
        'goal_x': [2.5] * batch_size,
        'goal_y': [1.0] * batch_size
    }

    files = {
        'image': ('rgb.png',   rgb_buf,   'image/png'),
        'depth': ('depth.png', depth_buf, 'image/png'),
    }
    data = {'goal_data': json.dumps(goal_data)}

    try:
        resp = requests.post(f"{SERVER_URL}/pointgoal_step", files=files, data=data, timeout=30)
        print(f"  状态码 : {resp.status_code}")
        if resp.status_code == 200:
            _print_trajectory_info(resp.json())
        else:
            print(f"  错误响应: {resp.text[:300]}")
        return resp.status_code == 200
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return False


def test_nogoal_step(batch_size: int = 1):
    """测试 /nogoal_step（无目标探索）"""
    print("=" * 55)
    print(f"测试 nogoal_step (batch_size={batch_size}) ...")

    rgb_buf   = _encode_rgb(create_dummy_image(batch_size=batch_size), 'PNG')
    depth_buf = _encode_depth_png(create_dummy_depth(batch_size=batch_size))

    files = {
        'image': ('rgb.png',   rgb_buf,   'image/png'),
        'depth': ('depth.png', depth_buf, 'image/png'),
    }

    try:
        resp = requests.post(f"{SERVER_URL}/nogoal_step", files=files, timeout=30)
        print(f"  状态码 : {resp.status_code}")
        if resp.status_code == 200:
            _print_trajectory_info(resp.json())
        else:
            print(f"  错误响应: {resp.text[:300]}")
        return resp.status_code == 200
    except Exception as e:
        import traceback
        print(f"  ❌ 错误: {e}")
        traceback.print_exc()
        return False


def test_imagegoal_step(batch_size: int = 1):
    """
    测试 /imagegoal_step（图像目标导航）

    字段说明（与 client_utils.imagegoal_step 保持一致）：
        image : 当前 RGB 观测（JPEG，垂直拼接 batch）
        goal  : 目标 RGB 图像（JPEG，垂直拼接 batch）
        depth : 深度图（PNG uint16，值 * 10000 = 米）
        depth_time / rgb_time : 时间戳（float，表单字段）
    """
    print("=" * 55)
    print(f"测试 imagegoal_step (batch_size={batch_size}) ...")

    import time

    # ----- 编码图像 -----
    # 当前观测：JPEG（与 client 一致）
    rgb_pil = create_dummy_image(batch_size=batch_size)
    rgb_buf = io.BytesIO()
    rgb_arr = np.array(rgb_pil)
    import cv2
    _, rgb_enc = cv2.imencode('.jpg', rgb_arr)
    rgb_buf.write(rgb_enc)
    rgb_buf.seek(0)

    # 目标图像：JPEG
    goal_pil = create_dummy_image(batch_size=batch_size)
    goal_buf = io.BytesIO()
    goal_arr = np.array(goal_pil)
    _, goal_enc = cv2.imencode('.jpg', goal_arr)
    goal_buf.write(goal_enc)
    goal_buf.seek(0)

    # 深度图：PNG uint16（client 端 *10000 后 clip 到 65535）
    depth_arr = np.random.uniform(0.5, 5.0, (480 * batch_size, 640)).astype(np.float32)
    depth_u16 = np.clip(depth_arr * 10000.0, 0, 65535).astype(np.uint16)
    _, depth_enc = cv2.imencode('.png', depth_u16)
    depth_buf = io.BytesIO()
    depth_buf.write(depth_enc)
    depth_buf.seek(0)

    # ----- 发送请求 -----
    files = {
        'image': ('image.jpg', rgb_buf.getvalue(),  'image/jpeg'),
        'goal':  ('goal.jpg',  goal_buf.getvalue(), 'image/jpeg'),
        'depth': ('depth.png', depth_buf.getvalue(),'image/png'),
    }
    data = {
        'depth_time': time.time(),
        'rgb_time':   time.time(),
    }

    try:
        resp = requests.post(f"{SERVER_URL}/imagegoal_step", files=files, data=data, timeout=30)
        print(f"  状态码 : {resp.status_code}")

        if not resp.text.strip():
            print("  ❌ 服务器返回空响应（server 可能正在 reset 或尚未就绪）")
            return False

        if resp.status_code == 200:
            result = resp.json()
            _print_trajectory_info(result)
        else:
            print(f"  错误响应: {resp.text[:300]}")

        return resp.status_code == 200

    except json.JSONDecodeError as e:
        print(f"  ❌ JSON 解析失败: {e}")
        print(f"     原始响应: '{resp.text[:200]}'")
        return False
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return False


def test_batch_processing(endpoint: str = 'nogoal'):
    """测试批处理能力（batch_size = 1/2/4）"""
    print("=" * 55)
    print(f"测试批处理（endpoint={endpoint}_step）...")

    fn_map = {
        'nogoal':    test_nogoal_step,
        'pointgoal': test_pointgoal_step,
        'imagegoal': test_imagegoal_step,
    }
    fn = fn_map.get(endpoint, test_nogoal_step)

    for bs in [1, 2, 4]:
        print(f"\n  --- batch_size = {bs} ---")
        ok = fn(batch_size=bs)
        if not ok:
            print(f"  ⚠️ batch_size={bs} 测试失败，终止批处理测试")
            return False
    return True


# ============================================================
# 主入口
# ============================================================

def main(port: int = 9999):
    global SERVER_URL
    SERVER_URL = f"http://localhost:{port}"
    print("=" * 55)
    print("FLUX Navigation Server 接口测试")
    print(f"服务器地址: {SERVER_URL}")
    print("=" * 55)

    results: dict[str, bool] = {}

    steps = [
        ("1/6  navigator_reset",     lambda: test_navigator_reset()),
        ("2/6  navigator_reset_env", lambda: test_navigator_reset_env(env_id=0)),
        ("3/6  pointgoal_step",      lambda: test_pointgoal_step(batch_size=1)),
        ("4/6  nogoal_step",         lambda: test_nogoal_step(batch_size=1)),
        ("5/6  imagegoal_step",      lambda: test_imagegoal_step(batch_size=1)),
        ("6/6  batch_processing",    lambda: test_batch_processing(endpoint='nogoal')),
    ]

    for label, fn in steps:
        print(f"\n[测试 {label}]")
        results[label] = fn()

    # ----- 汇总 -----
    print("\n" + "=" * 55)
    print("测试结果汇总:")
    print("=" * 55)
    for name, ok in results.items():
        mark = "✅ 通过" if ok else "❌ 失败"
        print(f"  {name:35s}: {mark}")
    print("=" * 55)

    failed = [n for n, ok in results.items() if not ok]
    if not failed:
        print("✅ 所有测试通过!")
    else:
        print(f"❌ {len(failed)} 个测试失败: {', '.join(failed)}")
        print("\n排查建议:")
        print("  1. 确认服务器已启动且模型加载完成")
        print("  2. 检查服务器日志中的具体报错")
        print("  3. 确认各接口字段名与 server 端一致")
        print("     imagegoal_step: image / goal / depth (非 goal_image)")

    return len(failed) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FLUX navigation HTTP server")
    parser.add_argument("--port", "-p", type=int, default=9999,
                        help="Server port (default: 9999)")
    parser.add_argument("--test", "-t",
                        choices=["all", "reset", "pointgoal", "nogoal", "imagegoal", "batch"],
                        default="all",
                        help="单独运行某项测试 (default: all)")
    args = parser.parse_args()

    if args.test == "all":
        ok = main(port=args.port)
    else:
        SERVER_URL = f"http://localhost:{args.port}"
        fn_map = {
            "reset":     test_navigator_reset,
            "pointgoal": lambda: test_pointgoal_step(batch_size=1),
            "nogoal":    lambda: test_nogoal_step(batch_size=1),
            "imagegoal": lambda: test_imagegoal_step(batch_size=1),
            "batch":     lambda: test_batch_processing(endpoint='nogoal'),
        }
        ok = fn_map[args.test]()

    exit(0 if ok else 1)