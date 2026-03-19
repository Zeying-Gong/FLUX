#!/usr/bin/env python

import threading
import queue
import requests
import json
import time
import argparse
from io import BytesIO

import numpy as np
import cv2
import pyrealsense2 as rs
from PIL import Image

# Import LeKiwi modules from your original code
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig
from lerobot.robots.lekiwi.lekiwi import LeKiwi

class LeKiwiClient:
    def __init__(self, args):
        self.args = args
        self.stop_event = threading.Event()
        self.cmd_queue = queue.Queue()  # Removed maxsize - we handle clearing explicitly
        self.pipeline = None
        self.intrinsic_matrix = None

    def init_realsense(self):
        """Initialize Realsense pipeline and get intrinsic parameters"""
        print("Initializing Realsense...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # Configure streams (match server's expected resolution)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        try:
            self.pipeline.start(config)
        except Exception as e:
            print(f"Realsense initialization failed: {e}")
            return False
        
        # Get intrinsic parameters (required for server initialization)
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError("Failed to get color frame for intrinsics")
            
            intr = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
            self.intrinsic_matrix = np.array([
                [intr.fx, 0, intr.ppx],
                [0, intr.fy, intr.ppy],
                [0, 0, 1]
            ])
            print(f"Realsense Intrinsics:\n{self.intrinsic_matrix}")
            return True
        except Exception as e:
            print(f"Failed to get Realsense intrinsics: {e}")
            self.pipeline.stop()
            return False

    def communication_worker(self):
        """Thread: Capture Realsense data, send to server, receive command lists"""
        # First initialize server with navigator_reset
        if not self._init_server():
            self.stop_event.set()
            return

        # Prepare goal data (batch_size compatible)
        goal_data = {
            "goal_x": [self.args.goal_x] * self.args.batch_size,
            "goal_y": [self.args.goal_y] * self.args.batch_size
        }

        send_interval = 1.0 / self.args.send_fps  # Control send frequency
        last_send_time = time.time()

        while not self.stop_event.is_set():
            # Control send frequency
            current_time = time.time()
            if current_time - last_send_time < send_interval:
                time.sleep(0.001)
                continue
            last_send_time = current_time

            # Capture Realsense frames
            color_img, depth_img = self._capture_realsense_frames()
            if color_img is None or depth_img is None:
                continue

            # Convert to PIL images (server-compatible format)
            pil_color = self._convert_color_to_pil(color_img)
            pil_depth = self._convert_depth_to_pil(depth_img)
            frame_capture_time = time.time()

            # Send to server and get commands
            cmd_list = self._send_to_server(pil_color, pil_depth, goal_data)
            print(f"Received {len(cmd_list)} commands from server")
            send_time = time.time()
            print(f"Server communication costs {send_time - frame_capture_time:.3f}s")
            if cmd_list:
                self._update_cmd_queue(cmd_list)

    def _init_server(self):
        """Send initial reset request to server"""
        print(f"Initializing server: {self.args.server_url}/navigator_reset")
        reset_data = {
            "intrinsic": self.intrinsic_matrix.tolist(),
            "stop_threshold": self.args.stop_threshold,
            "batch_size": self.args.batch_size
        }
        
        try:
            response = requests.post(
                f"{self.args.server_url}/navigator_reset",
                json=reset_data,
                timeout=50
            )
            response.raise_for_status()
            print(f"Server initialized: {response.json()}")
            return True
        except Exception as e:
            print(f"Server initialization failed: {e}")
            return False

    def _capture_realsense_frames(self):
        """Capture and preprocess Realsense frames"""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=500)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                print("No Realsense frames received")
                return None, None

            # Convert to numpy arrays
            color_img = np.asanyarray(color_frame.get_data())  # BGR format
            depth_img = np.asanyarray(depth_frame.get_data())  # Z16 (mm)

            return color_img, depth_img
        except Exception as e:
            print(f"Frame capture failed: {e}")
            return None, None

    @staticmethod
    def _convert_color_to_pil(color_img):
        """Convert BGR numpy array to RGB PIL Image"""
        rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_img)

    @staticmethod
    def _convert_depth_to_pil(depth_img):
        """Convert Z16 depth array to 16-bit PIL Image (server-compatible)"""
        return Image.fromarray(depth_img.astype(np.uint16), mode='I;16')

    def _send_to_server(self, pil_color, pil_depth, goal_data):
        """Send multipart request to server and return command list"""
        # Prepare image buffers
        color_buf = BytesIO()
        pil_color.save(color_buf, format='jpeg')
        color_buf.seek(0)

        depth_buf = BytesIO()
        pil_depth.save(depth_buf, format='PNG')
        depth_buf.seek(0)

        # Prepare request data
        files = {
            'image': ('color.png', color_buf, 'image/jpeg'),
            'depth': ('depth.png', depth_buf, 'image/png')
        }
        data = {'goal_data': json.dumps(goal_data)}

        try:
            response = requests.post(
                f"{self.args.server_url}/pointgoal_step",
                files=files,
                data=data,
                timeout=1000000
            )
            response.raise_for_status()
            result = response.json()
            return result.get('cmd_list', [])
        except Exception as e:
            print(f"Server communication failed: {e}")
            return []
        finally:
            color_buf.close()
            depth_buf.close()

    def _update_cmd_queue(self, cmd_list):
        """Update command queue - discard ALL old commands and keep only the latest"""
        if not cmd_list:
            return

        try:
            # Clear all existing commands in the queue (thread-safe)
            while not self.cmd_queue.empty():
                try:
                    self.cmd_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Put the latest command list
            self.cmd_queue.put_nowait(cmd_list)
            print(f"Updated queue with latest {len(cmd_list)} commands (discarded old ones)")
        except Exception as e:
            print(f"Command queue update failed: {e}")

    def execution_worker(self):
        """Thread: Execute commands from queue on LeKiwi robot - prioritize LATEST commands"""
        # Initialize LeKiwi robot
        robot = LeKiwi(LeKiwiConfig())
        try:
            print("Connecting to LeKiwi robot...")
            robot.connect()
            print("LeKiwi connected successfully")
        except Exception as e:
            print(f"LeKiwi connection failed: {e}")
            self.stop_event.set()
            return

        last_cmd_time = time.time()
        watchdog_active = False
        current_cmds = []  # Current command list to execute
        cmd_idx = 0        # Current position in command list

        while not self.stop_event.is_set():
            loop_start = time.time()
            print('cmd index: ', cmd_idx, len(current_cmds))

            # Step 1: Check for NEW command list (highest priority)
            new_cmds = self._get_latest_cmds()
            if new_cmds:
                # Abort old commands - switch to new list immediately
                current_cmds = new_cmds
                cmd_idx = 0  # Reset to start of new command list
                last_cmd_time = time.time()
                watchdog_active = False
                print(f"Switched to new command list (size: {len(current_cmds)})")

            # Step 2: Execute next command in current list (if available)
            if current_cmds and cmd_idx < len(current_cmds):
                cmd = current_cmds[cmd_idx]
                self._execute_cmd(robot, cmd)
                cmd_idx += 1  # Move to next command in list
                last_cmd_time = time.time()
                watchdog_active = False
            else:
                # Step 3: Watchdog - stop robot if no commands for timeout
                if (time.time() - last_cmd_time > self.args.watchdog_timeout / 1000 
                    and not watchdog_active):
                    print("Watchdog timeout - stopping robot")
                    robot.stop_base()
                    watchdog_active = True

            # Control loop frequency (maintain exec_freq)
            elapsed = time.time() - loop_start
            time.sleep(max(1/self.args.exec_freq - elapsed, 0.001))

        # Cleanup robot connection
        print("Stopping robot...")
        robot.stop_base()
        robot.disconnect()
        print("LeKiwi disconnected")

    def _get_latest_cmds(self):
        """Get latest command list (discard intermediate lists if multiple)"""
        latest_cmds = None
        try:
            # Get all commands in queue (only keep the last one)
            while not self.cmd_queue.empty():
                latest_cmds = self.cmd_queue.get_nowait()
            return latest_cmds
        except queue.Empty:
            return None

    def _execute_cmd(self, robot, cmd):
        """Execute single velocity command (vx, vy, wz)"""
        vx, vy, wz = cmd
        print(f"Executing command: vx={vx:.3f}, vy={vy:.3f}, wz={wz:.3f}")
        
        # Base movement command
        base_action = {'x.vel': vx, 'y.vel': vy, 'theta.vel': wz}
        
        # Fixed arm position (adjust as needed)
        arm_action = {
            'arm_shoulder_pan.pos': -4.618768328445739,
            'arm_shoulder_lift.pos': -90.0,
            'arm_elbow_flex.pos': 0.0,
            'arm_wrist_flex.pos': 90.744630071599047,
            'arm_wrist_roll.pos': 0.890109890109898,
            'arm_gripper.pos': 0.0
        }

        try:
            # Send combined action to robot
            robot.send_action({**base_action, **arm_action})
            # Command execution duration (configurable)
            time.sleep(self.args.cmd_exec_delay)
        except Exception as e:
            print(f"Command execution failed: {e}")

    def run(self):
        """Start client and run until stopped"""
        # Initialize Realsense first (critical dependency)
        if not self.init_realsense():
            print("Failed to initialize Realsense - exiting")
            return

        # Start worker threads
        comm_thread = threading.Thread(target=self.communication_worker, daemon=True)
        exec_thread = threading.Thread(target=self.execution_worker, daemon=True)

        comm_thread.start()
        exec_thread.start()
        print("Client started. Press Ctrl+C to stop.")

        # Main thread: Wait for stop signal (Ctrl+C)
        try:
            while not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping client...")
            self.stop_event.set()

        # Cleanup resources
        comm_thread.join(timeout=2.0)
        exec_thread.join(timeout=2.0)
        self.pipeline.stop()
        print("Client stopped cleanly")

def parse_args():
    parser = argparse.ArgumentParser(description="LeKiwi Client with Realsense + Server Communication")
    
    # Server configuration
    parser.add_argument("--server-url", type=str, default="http://192.168.1.100:8888",
                        help="Flask server URL (e.g., http://192.168.1.100:8888)")
    parser.add_argument("--send-fps", type=int, default=7, help="Frame send frequency to server")
    
    # Navigation goals
    parser.add_argument("--goal-x", type=float, default=10.0, help="Goal X position (meters)")
    parser.add_argument("--goal-y", type=float, default=0.0, help="Goal Y position (meters)")
    parser.add_argument("--stop-threshold", type=float, default=-1.0, help="Server stop threshold")
    parser.add_argument("--batch-size", type=int, default=1, help="Server batch size (must match)")
    
    # Robot control
    parser.add_argument("--cmd-exec-delay", type=float, default=0.1,
                        help="Time to execute each velocity command (seconds)")
    parser.add_argument("--exec-freq", type=int, default=50, help="Command execution loop frequency")
    parser.add_argument("--watchdog-timeout", type=int, default=100000,
                        help="Watchdog timeout (milliseconds) - stops robot if no commands")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    client = LeKiwiClient(args)
    client.run()
