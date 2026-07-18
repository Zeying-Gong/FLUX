"""
Follower Behavior Script v2 — 对齐 Habitat Social Nav 的 turn-or-go 状态机

Tracking-only variant. The untouched datagen baseline is kept in
``cylinder_follow_v2.py``; online collection experiments must use this file so
their navigation, safety, and camera-only changes do not alter that baseline.

核心改动（相比 v1）：
- 删除解耦速度模型（decoupled speed），改用 Habitat 的 turn-or-go
- 路径点选择：取 path_points[1]（第二个点），而非 path_points[-2]
- 对齐 Habitat oracle：导航到 human 本体，视觉 bbox 居中作为软约束
- 动态避障：注册到 GlobalCharacterPositionManager + 旋转绕行
- 弯道优化：小角度偏差时边走边转，避免 stop-turn-go 卡顿
"""

import omni.ext
import omni.kit.commands
from pxr import Gf, Sdf, UsdGeom, Usd
import omni.usd
import carb
import math
import numpy as np
from omni.kit.scripting import BehaviorScript


class FollowerInitializationError(RuntimeError):
    """Raised when required follower setup attributes are missing."""

# Isaac Sim API
try:
    import omni.anim.navigation.core as nav
    from omni.anim.people.scripts.global_character_position_manager import (
        GlobalCharacterPositionManager,
    )
    ANIM_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Isaac Sim animation modules not available: {e}")
    nav = None
    GlobalCharacterPositionManager = None
    ANIM_MODULES_AVAILABLE = False


class FollowerBehavior(BehaviorScript):
    """
    Habitat 风格的 turn-or-go 跟随行为脚本。

    状态机：
      TOO_CLOSE  → 后退
      AT_GOAL    → 停止
      NAVIGATING → 转向 或 直走（从不同时）
    """

    def on_init(self):
        self.follower_prim_path = str(self.prim_path)

        # Habitat 对齐参数（来自 hssd_spot_human_social_nav.yaml）
        self.my_radius = 0.20
        self.follow_distance = 1.5    # 兼容旧配置；控制实际使用 safe_dis_max/path_stop_distance
        self.too_close_distance = 1.0 # 低于该距离必须后退，避免贴近碰撞
        self.safe_dis_min = 1.2       # 慢速带外沿：1.0~1.2m 减速贴近
        self.safe_dis_max = 2.0       # 追赶加速带外沿：>=2.0m 达到最高追赶速度
        self.danger_dis = 0.8         # Habitat DANGER_DISTANCE
        self.turn_thresh = 0.1        # 转向阈值（弧度，约5.7度）
        self.turn_and_go_thresh = 0.5 # 边走边转阈值（~28度，仅 differential 模式）
        self.forward_velocity = 2.5  # Habitat oracle 前向速度上限（m/s）
        self.lateral_velocity = 1.5  # Habitat oracle 侧向速度上限（m/s）
        self.turn_velocity = 2.0      # Habitat oracle yaw 速度上限（rad/s）

        # 加速度限制（防止瞬间跳变导致画面抖动）
        self.max_linear_accel = 3.0   # 最大线速度变化率（m/s²）
        self.max_lateral_accel = 3.0  # 最大侧向速度变化率（m/s²）
        self.max_angular_accel = 4.0  # 最大角速度变化率（rad/s²）
        self.prev_linear_vel = 0.0    # 上一帧线速度
        self.prev_angular_vel = 0.0   # 上一帧角速度
        self.prev_body_action = np.zeros(3, dtype=np.float32)

        # 运动学模型："differential" = 差速轮（只有 vx + ω）
        #              "omnidirectional" = 万向轮（vx + vy + ω 解耦）
        self.motion_type = "omnidirectional"

        # 内部状态
        self.navigation_interface = None
        self.navmesh = None
        self.target_prim = None
        self.follower_prim = None

        # 当前 NavMesh 路线。正常控制每帧重算；缓存仅用于 debug 和短暂 fallback。
        self.cached_path = []          # 缓存的路径点列表
        self.cached_waypoint_idx = 1   # 当前追踪的路径点索引
        self.path_replan_dist = 1.2    # 目标移动超过此距离时重新规划
        self.waypoint_reach_dist = 0.3 # 到达路径点的判定距离
        self.last_target_pos = None    # 上次规划时的目标位置
        self.max_navmesh_snap = 0.25   # 候选点投影到 navmesh 的最大允许偏差（米）
        self.target_navmesh_snap = 1.0 # 目标 SkelRoot 在货架转角可允许更远投影
        self.planning_radius = 0.20    # Overridden by follower:planning_radius when configured.
        self.path_stop_distance = self.too_close_distance  # 进入目标半径后暂停；NavMesh 路线仍规划到目标
        self._current_nav_waypoint = None
        self._last_nav_path_distance = None
        self._last_route_path_distance = None
        self._last_tracking_distance = None
        self._last_effective_path_stop_distance = self.path_stop_distance
        self._last_path_detour_active = False
        self._path_replanned_this_frame = False
        self._path_query_failed = False
        self._nav_dominant = False
        self._snap_slide_count = 0
        self.snap_slide_hold_threshold = 3
        self._previous_projection_origin = None
        self._projection_cycle_count = 0
        self.target_pos_history = []   # Recent observed human trail for path fallback.
        self.target_pos_history_limit = 300
        self._corridor_center_offset = 0.0
        self._consecutive_path_failures = 0
        self._prev_target_motion_pos = None
        self.human_in_frame = False
        self.human_center_x = 0.5
        self.last_seen_center_x = 0.5
        self.visual_confidence = 0.0
        self.prev_error_t = 0.0
        self._raw_human_angle_diff = 0.0
        self._last_target_bbox_frame = -1
        self._follower_center_z = None
        self.debug_draw_path = True
        self.debug_draw_path_thickness = 4.0
        self.debug_draw_trail = True
        self.debug_draw_trail_max_points = 300
        self.debug_draw_trail_min_distance = 0.05
        self._debug_trail_points = []
        self._debug_draw_iface = None
        self._debug_draw_unavailable_logged = False

        # 相机同步（刚性绑定 cylinder）
        self.camera_prim = None
        self.camera_offset = [0.0, 0.0, 0.0]
        self.camera_rotation = [0.0, 0.0, 0.0]

        # 动态避障（robot/follower 主动避开非目标角色）
        self.character_manager = None
        self.dynamic_avoidance_enabled = False
        self.avoidance_radius = 2.4   # 开始检测避障的距离
        self.dynamic_avoidance_radius = 0.5
        self.character_obstacle_radius = 0.5
        self.avoidance_time_horizon = 1.0
        self.avoidance_clearance_margin = 0.12
        self.orca_time_horizon = 2.0
        self.orca_neighbor_distance = 1.2
        self.orca_clearance_margin = 0.1
        self._dynamic_obstacle_history = {}
        self.positions_over_time = []  # 位置历史（估算速度用）
        self.delta_time_list = []

        # 调试日志（用于定位“停住不动/朝向异常”）
        self.debug_enabled = True
        self.debug_period_sec = 0.5
        self._debug_time = 0.0
        self._next_periodic_log_time = 0.0
        self._last_state = None
        self._event_last_log_time = {}

        # 卡住检测（期望在移动，但位移几乎为零）
        self._last_follower_pos = None
        self._last_stuck_tracking_distance = None
        self._last_stuck_waypoint_distance = None
        self._last_stuck_waypoint_idx = None
        self._stuck_elapsed = 0.0
        self._last_stuck_waypoint_skip_time = -999.0
        self.stuck_move_eps = 0.015
        self.stuck_warn_sec = 0.8
        self.stuck_recovery_enabled = True
        self.stuck_recovery_sec = 1.2
        self.stuck_recovery_cooldown = 1.0
        self.stuck_recovery_distance = 0.8
        self._recovery_waypoint = None
        self._recovery_until = 0.0
        self._recovery_count = 0
        self._last_recovery_time = -999.0
        self._last_recovery_reason = ""

        # 上一帧指令速度（用于卡住检测日志）
        self._last_cmd_linear = 0.0
        self._last_cmd_lateral = 0.0
        self._last_cmd_angular = 0.0
        self._lazy_on_play_attempted = False

        # 从 prim 属性读取目标与可配置控制参数，要求必须显式配置目标。
        try:
            self.target_skelroot_path = self._read_target_from_attribute()
        except FollowerInitializationError as e:
            carb.log_error(f"[FollowerBehavior v2] Initialization failed: {e}")
            self.target_skelroot_path = None

        carb.log_info(
            f"[FollowerBehavior v2] on_init for {self.prim_path}, "
            f"target={self.target_skelroot_path}, motion={self.motion_type}"
        )

    # ------------------------------------------------------------------
    # 属性读取 & 自动检测（保留 v1 逻辑）
    # ------------------------------------------------------------------

    def _read_target_from_attribute(self):
        """从 prim 属性读取目标 skelroot 路径；缺失时直接报错。"""
        follower_prim = self.stage.GetPrimAtPath(self.prim_path)
        if not follower_prim.IsValid():
            raise FollowerInitializationError(
                f"Follower prim invalid: {self.prim_path}"
            )

        target_attr = follower_prim.GetAttribute("follower:target_skelroot_path")
        if not target_attr:
            raise FollowerInitializationError(
                "Missing required attr: follower:target_skelroot_path"
            )

        target_path = target_attr.Get()
        if not target_path:
            raise FollowerInitializationError(
                "Empty required attr: follower:target_skelroot_path"
            )

        target_prim = self.stage.GetPrimAtPath(str(target_path))
        if not target_prim.IsValid():
            raise FollowerInitializationError(
                f"Target prim invalid: {target_path}"
            )

        radius_attr = follower_prim.GetAttribute("follower:radius")
        if radius_attr and radius_attr.Get():
            self.my_radius = float(radius_attr.Get())

        follow_distance_attr = follower_prim.GetAttribute("follower:follow_distance")
        if follow_distance_attr and follow_distance_attr.Get() is not None:
            self.follow_distance = float(follow_distance_attr.Get())

        motion_attr = follower_prim.GetAttribute("follower:motion_type")
        if motion_attr and motion_attr.Get():
            val = str(motion_attr.Get()).lower()
            if val in ("differential", "omnidirectional"):
                self.motion_type = val

        safe_min_attr = follower_prim.GetAttribute("follower:safe_distance_min")
        if safe_min_attr and safe_min_attr.Get() is not None:
            self.safe_dis_min = float(safe_min_attr.Get())

        safe_max_attr = follower_prim.GetAttribute("follower:safe_distance_max")
        if safe_max_attr and safe_max_attr.Get() is not None:
            self.safe_dis_max = float(safe_max_attr.Get())
        if self.safe_dis_max < self.safe_dis_min:
            carb.log_warn(
                "[FollowerBehavior v2] safe_distance_max < safe_distance_min; "
                "clamping max to min"
            )
            self.safe_dis_max = self.safe_dis_min
        too_close_attr = follower_prim.GetAttribute("follower:too_close_distance")
        if too_close_attr and too_close_attr.Get() is not None:
            self.too_close_distance = float(too_close_attr.Get())
        self.too_close_distance = float(np.clip(
            self.too_close_distance,
            0.0,
            self.safe_dis_min,
        ))
        self.follow_distance = float(np.clip(
            self.follow_distance,
            self.safe_dis_min,
            self.safe_dis_max,
        ))
        self.path_stop_distance = self.too_close_distance

        max_snap_attr = follower_prim.GetAttribute("follower:max_navmesh_snap")
        if max_snap_attr and max_snap_attr.Get() is not None:
            self.max_navmesh_snap = float(max_snap_attr.Get())

        target_snap_attr = follower_prim.GetAttribute("follower:target_navmesh_snap")
        if target_snap_attr and target_snap_attr.Get() is not None:
            self.target_navmesh_snap = float(target_snap_attr.Get())
        self.target_navmesh_snap = max(float(self.target_navmesh_snap), float(self.max_navmesh_snap))

        planning_radius_attr = follower_prim.GetAttribute("follower:planning_radius")
        if planning_radius_attr and planning_radius_attr.Get() is not None:
            self.planning_radius = float(planning_radius_attr.Get())
        self.planning_radius = max(float(self.planning_radius), float(self.my_radius))

        dynamic_enabled_attr = follower_prim.GetAttribute(
            "follower:dynamic_avoidance_enabled"
        )
        if dynamic_enabled_attr and dynamic_enabled_attr.Get() is not None:
            self.dynamic_avoidance_enabled = bool(dynamic_enabled_attr.Get())

        dynamic_radius_attr = follower_prim.GetAttribute(
            "follower:dynamic_avoidance_radius"
        )
        if dynamic_radius_attr and dynamic_radius_attr.Get() is not None:
            self.dynamic_avoidance_radius = float(dynamic_radius_attr.Get())
        self.dynamic_avoidance_radius = max(
            float(self.dynamic_avoidance_radius),
            float(self.my_radius),
        )

        path_stop_attr = follower_prim.GetAttribute("follower:path_stop_distance")
        if path_stop_attr and path_stop_attr.Get() is not None:
            self.path_stop_distance = float(path_stop_attr.Get())
        self.path_stop_distance = max(float(self.path_stop_distance), 0.0)

        stuck_eps_attr = follower_prim.GetAttribute("follower:stuck_move_epsilon")
        if stuck_eps_attr and stuck_eps_attr.Get() is not None:
            self.stuck_move_eps = float(stuck_eps_attr.Get())

        stuck_warn_attr = follower_prim.GetAttribute("follower:stuck_warn_seconds")
        if stuck_warn_attr and stuck_warn_attr.Get() is not None:
            self.stuck_warn_sec = float(stuck_warn_attr.Get())

        recovery_enabled_attr = follower_prim.GetAttribute(
            "follower:stuck_recovery_enabled"
        )
        if recovery_enabled_attr and recovery_enabled_attr.Get() is not None:
            self.stuck_recovery_enabled = bool(recovery_enabled_attr.Get())

        recovery_sec_attr = follower_prim.GetAttribute(
            "follower:stuck_recovery_seconds"
        )
        if recovery_sec_attr and recovery_sec_attr.Get() is not None:
            self.stuck_recovery_sec = float(recovery_sec_attr.Get())

        recovery_cooldown_attr = follower_prim.GetAttribute(
            "follower:stuck_recovery_cooldown"
        )
        if recovery_cooldown_attr and recovery_cooldown_attr.Get() is not None:
            self.stuck_recovery_cooldown = float(recovery_cooldown_attr.Get())

        recovery_distance_attr = follower_prim.GetAttribute(
            "follower:stuck_recovery_distance"
        )
        if recovery_distance_attr and recovery_distance_attr.Get() is not None:
            self.stuck_recovery_distance = float(recovery_distance_attr.Get())

        debug_path_attr = follower_prim.GetAttribute("follower:debug_draw_path")
        if debug_path_attr and debug_path_attr.Get() is not None:
            self.debug_draw_path = bool(debug_path_attr.Get())

        debug_path_thickness_attr = follower_prim.GetAttribute(
            "follower:debug_draw_path_thickness"
        )
        if (
            debug_path_thickness_attr
            and debug_path_thickness_attr.Get() is not None
        ):
            self.debug_draw_path_thickness = float(debug_path_thickness_attr.Get())
        self.debug_draw_path_thickness = max(float(self.debug_draw_path_thickness), 1.0)

        debug_trail_attr = follower_prim.GetAttribute("follower:debug_draw_trail")
        if debug_trail_attr and debug_trail_attr.Get() is not None:
            self.debug_draw_trail = bool(debug_trail_attr.Get())

        debug_trail_max_attr = follower_prim.GetAttribute(
            "follower:debug_draw_trail_max_points"
        )
        if debug_trail_max_attr and debug_trail_max_attr.Get() is not None:
            self.debug_draw_trail_max_points = int(debug_trail_max_attr.Get())
        self.debug_draw_trail_max_points = max(
            int(self.debug_draw_trail_max_points), 2
        )

        debug_trail_min_attr = follower_prim.GetAttribute(
            "follower:debug_draw_trail_min_distance"
        )
        if debug_trail_min_attr and debug_trail_min_attr.Get() is not None:
            self.debug_draw_trail_min_distance = float(debug_trail_min_attr.Get())
        self.debug_draw_trail_min_distance = max(
            float(self.debug_draw_trail_min_distance), 0.0
        )

        return str(target_path)

    def _find_own_character_skelroot(self):
        """根据父级层级自动检测本圆柱对应的 SkelRoot。"""
        try:
            prim = self.stage.GetPrimAtPath(self.prim_path)
            if not prim.IsValid():
                return None

            manroot_prim = None
            current = prim
            while current and current.IsValid() and manroot_prim is None:
                if "ManRoot" in current.GetName():
                    manroot_prim = current
                    break
                parent = current.GetParent()
                if parent and parent.IsValid():
                    for sibling in parent.GetChildren():
                        if (
                            sibling.GetPath() != current.GetPath()
                            and "ManRoot" in sibling.GetName()
                        ):
                            manroot_prim = sibling
                            break
                current = parent

            if not manroot_prim:
                return None

            def find_skelroot(p):
                if p.GetTypeName() == "SkelRoot":
                    return str(p.GetPath())
                for child in p.GetChildren():
                    result = find_skelroot(child)
                    if result:
                        return result
                return None

            return find_skelroot(manroot_prim)

        except Exception as e:
            carb.log_error(f"[FollowerBehavior v2] Error in auto-detect: {e}")
            return None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def on_play(self):
        print("[FollowerBehavior v2] on_play called")

        if not self.target_skelroot_path:
            carb.log_error("[FollowerBehavior v2] Missing valid target_skelroot_path; aborting behavior")
            return

        # 导航接口
        try:
            if nav is not None:
                self.navigation_interface = nav.acquire_interface()
                if self.navigation_interface:
                    self.navmesh = self.navigation_interface.get_navmesh()
        except Exception as e:
            print(f"[FollowerBehavior v2] Cannot acquire nav interface: {e}")

        # 目标 prim
        self.target_prim = self.stage.GetPrimAtPath(self.target_skelroot_path)
        if not self.target_prim.IsValid():
            carb.log_error(
                f"[FollowerBehavior v2] Target not found: "
                f"{self.target_skelroot_path}"
            )
            return

        self.follower_prim = self.prim
        if not self.follower_prim.IsValid():
            carb.log_error(
                f"[FollowerBehavior v2] Follower not found: "
                f"{self.follower_prim_path}"
            )
            return
        follower_pos, _ = self._get_prim_pose(self.follower_prim)
        if follower_pos is not None:
            self._follower_center_z = float(follower_pos[2])
            self._debug_trail_points = [np.asarray(follower_pos, dtype=np.float32)]

        # 相机属性
        try:
            cam_attr = self.follower_prim.GetAttribute("follower:camera_path")
            if cam_attr and cam_attr.Get():
                cam_path = cam_attr.Get()
                self.camera_prim = self.stage.GetPrimAtPath(cam_path)
                if not self.camera_prim.IsValid():
                    self.camera_prim = None

            offset_attr = self.follower_prim.GetAttribute(
                "follower:camera_offset"
            )
            if offset_attr and offset_attr.Get():
                v = offset_attr.Get()
                self.camera_offset = [v[0], v[1], v[2]]

            rot_attr = self.follower_prim.GetAttribute(
                "follower:camera_rotation"
            )
            if rot_attr and rot_attr.Get():
                v = rot_attr.Get()
                self.camera_rotation = [v[0], v[1], v[2]]
        except Exception as e:
            carb.log_warn(
                f"[FollowerBehavior v2] Could not read camera attrs: {e}"
            )

        # 角色管理器（可选动态避障）
        try:
            if (
                self.dynamic_avoidance_enabled
                and GlobalCharacterPositionManager is not None
            ):
                self.character_manager = (
                    GlobalCharacterPositionManager.get_instance()
                )
        except Exception:
            self.character_manager = None

        print("[FollowerBehavior v2] Initialization complete")
        if self.debug_draw_path or self.debug_draw_trail:
            self._get_debug_draw_interface()
        self._dbg(
            "info",
            (
                f"on_play ready: motion={self.motion_type}, radius={self.my_radius}, "
                f"dynamic_enabled={self.dynamic_avoidance_enabled}, "
                f"dynamic_radius={self.dynamic_avoidance_radius}, "
                f"safe=[{self.safe_dis_min},{self.safe_dis_max}], "
                f"turn_thresh={self.turn_thresh}, turn_and_go={self.turn_and_go_thresh}, "
                f"navmesh={self.navmesh is not None}, manager={self.character_manager is not None}"
            ),
        )

    def on_stop(self):
        self.on_destroy()

    def on_destroy(self):
        # 从全局管理器注销，让其他角色不再避开已销毁的 cylinder
        if self.character_manager is not None:
            try:
                mgr = self.character_manager
                mgr._character_positions.pop(self.follower_prim_path, None)
                mgr._character_future_positions.pop(
                    self.follower_prim_path, None
                )
                mgr._character_radius.pop(self.follower_prim_path, None)
            except Exception:
                pass
        self.navigation_interface = None
        self.navmesh = None
        self.positions_over_time = []
        self.delta_time_list = []
        self.cached_path = []
        self.last_target_pos = None
        self._current_nav_waypoint = None
        self._recovery_waypoint = None
        self._debug_trail_points = []
        self._clear_debug_path()

    # ------------------------------------------------------------------
    # 核心循环：Habitat turn-or-go 状态机（带弯道优化 + 动态避障）
    # ------------------------------------------------------------------

    def on_update(self, _current_time: float, delta_time: float):
        # Kit can dynamically instantiate a newly bound BehaviorScript after
        # the timeline is already playing without delivering on_play(). In
        # that case on_init() has run, but target/follower/navmesh are unset.
        # Initialize lazily so dynamic oracle binding follows the same setup
        # path as a script present before the initial timeline play.
        if (not self.target_prim or not self.follower_prim) and not self._lazy_on_play_attempted:
            self._lazy_on_play_attempted = True
            self._log_event(
                "lazy_on_play",
                "on_update received before on_play; initializing now",
                cooldown=2.0,
                level="warn",
            )
            self.on_play()
        if not self.target_prim or not self.follower_prim:
            self._log_event("invalid_prims", "on_update skipped: invalid target/follower prim", cooldown=2.0, level="warn")
            return
        if delta_time <= 0:
            self._log_event("non_positive_dt", f"on_update skipped: delta_time={delta_time}", cooldown=2.0, level="warn")
            return

        self._debug_time += delta_time
        self._path_replanned_this_frame = False
        self._path_query_failed = False
        self._nav_dominant = False

        target_pos, target_rpy = self._get_prim_pose(self.target_prim)
        follower_pos, follower_rpy = self._get_prim_pose(self.follower_prim)
        if target_pos is None or target_rpy is None or follower_pos is None:
            self._log_event("pose_none", "pose query failed, skipping frame", cooldown=1.0, level="warn")
            return

        follower_yaw = follower_rpy[2]
        self._publish_diagnostics(snap_rejected=False)
        self._update_visual_tracking_from_bbox()

        # 可选发布 follower 位置；robot 侧局部避障不依赖 people 全局避障开关
        self._publish_position(follower_pos, delta_time)

        # 2D 向量：follower → human
        to_human_2d = np.array([
            target_pos[0] - follower_pos[0],
            target_pos[1] - follower_pos[1],
        ])
        dist_to_human = float(np.linalg.norm(to_human_2d))
        tracking_distance = dist_to_human
        self._last_tracking_distance = tracking_distance
        self.target_pos_history.append(target_pos.copy())
        if len(self.target_pos_history) > self.target_pos_history_limit:
            self.target_pos_history.pop(0)

        # follower 前向向量（Isaac Sim XY 平面）
        robot_forward_2d, _ = self._forward_left_vectors(follower_yaw)

        angle_to_human = self._get_angle(robot_forward_2d, to_human_2d)

        body_cmd = None
        next_waypoint = None
        force_nav_for_detour = False
        goal_stop_radius = self._goal_stop_radius()
        within_goal_stop_radius = dist_to_human <= goal_stop_radius
        target_closing = self._target_is_closing_on_follower(follower_pos)
        close_target_should_retreat = self._should_enter_too_close(
            dist_to_human,
            within_goal_stop_radius,
            target_closing,
        )
        if dist_to_human < self.safe_dis_min and close_target_should_retreat:
            detour_path = self._compute_follow_path(follower_pos, target_pos) or []
            if (
                dist_to_human >= self._too_close_distance()
                and detour_path
                and len(detour_path) >= 2
                and self._last_path_detour_active
                and self._last_nav_path_distance is not None
                and self._last_nav_path_distance > self.safe_dis_min
                and not target_closing
            ):
                self.cached_path = detour_path
                self.cached_waypoint_idx = self._initial_waypoint_idx_for_path(
                    follower_pos,
                    detour_path,
                )
                self.last_target_pos = target_pos.copy()
                tracking_distance = self._get_tracking_distance(dist_to_human)
                force_nav_for_detour = True
                self._log_event(
                    "too_close_detour_nav",
                    (
                        f"straight close but nav path detours: "
                        f"straight={dist_to_human:.3f}, "
                        f"path={self._last_nav_path_distance:.3f}; navigating"
                    ),
                    cooldown=0.8,
                    level="warn",
                )
            elif target_closing:
                self._log_event(
                    "too_close_target_closing",
                    (
                        f"target closing while too close: "
                        f"straight={dist_to_human:.3f}; retreating"
                    ),
                    cooldown=0.8,
                    level="warn",
                )

        too_close_exit_distance = self._too_close_distance() + max(
            0.04,
            self.my_radius * 0.15,
        )
        too_close_active = close_target_should_retreat
        if (
            not force_nav_for_detour
            and self._last_state == "TOO_CLOSE"
            and dist_to_human < too_close_exit_distance
        ):
            too_close_active = True

        # === 状态1: TOO_CLOSE — Habitat oracle retreat ===
        if too_close_active and not force_nav_for_detour:
            self._log_state_transition("TOO_CLOSE", dist_to_human, angle_to_human, None)
            # A retreat command must not inherit forward momentum from the
            # previous chase command. The follower is kinematic, so preserving
            # that momentum only delays braking and can cross the pedestrian's
            # proximity boundary before the next collection-frame check.
            self.prev_linear_vel = min(float(self.prev_linear_vel), 0.0)
            self.prev_body_action[0] = min(float(self.prev_body_action[0]), 0.0)
            body_cmd, next_waypoint = self._compute_habitat_retreat_command(
                follower_pos, target_pos, to_human_2d, dist_to_human
            )
        else:
            body_cmd, next_waypoint = self._compute_habitat_nav_command(
                follower_pos, follower_yaw, target_pos, to_human_2d, dist_to_human
            )
            tracking_distance = self._get_tracking_distance(dist_to_human)
            if self._should_pause_for_goal_radius(tracking_distance):
                body_cmd[:2] = 0.0
                body_cmd[2] = 0.0
                self._log_state_transition("AT_GOAL", tracking_distance, angle_to_human, None)
            elif tracking_distance < self.safe_dis_min:
                speed_scale = self._tracking_translation_scale(tracking_distance)
                body_cmd[0] *= speed_scale
                body_cmd[1] *= speed_scale
                if self.human_in_frame and self._is_human_stationary():
                    edge_dist = min(self.human_center_x, 1.0 - self.human_center_x)
                    if edge_dist < 0.25:
                        body_cmd[0] = 0.0
                        body_cmd[1] = 0.0
                        body_cmd[2] = float(np.clip(
                            self._raw_human_angle_diff * 5.0 * self.turn_velocity,
                            -self.turn_velocity,
                            self.turn_velocity,
                        ))
                moving = np.linalg.norm(body_cmd[:2]) > 0.03
                state_name = "TRACK_SAFE_DIST" if moving else "AT_GOAL"
                self._log_state_transition(state_name, tracking_distance, angle_to_human, None)
            else:
                speed_scale = self._tracking_translation_scale(tracking_distance)
                body_cmd[0] *= speed_scale
                body_cmd[1] *= speed_scale
                to_waypoint_2d = to_human_2d
                if next_waypoint is not None:
                    to_waypoint_2d = np.array([
                        next_waypoint[0] - follower_pos[0],
                        next_waypoint[1] - follower_pos[1],
                    ])
                angle_to_waypoint = self._get_angle(robot_forward_2d, to_waypoint_2d)
                self._log_state_transition("NAV_OMNI", tracking_distance, angle_to_human, angle_to_waypoint)

        self._apply_body_command(body_cmd, to_human_2d, delta_time, follower_yaw)
        updated_follower_pos, _ = self._get_prim_pose(self.follower_prim)
        if updated_follower_pos is None:
            updated_follower_pos = follower_pos

        self._record_debug_trail_point(updated_follower_pos)
        self._draw_debug_path(updated_follower_pos, target_pos, next_waypoint)
        self._sync_camera(target_pos, delta_time)
        self._log_periodic(dist_to_human, angle_to_human, updated_follower_pos, target_pos, next_waypoint)
        if next_waypoint is not None:
            self._current_nav_waypoint = next_waypoint.copy()
        self._update_stuck_detector(updated_follower_pos, delta_time, tracking_distance)
        self._publish_diagnostics(waypoint=next_waypoint)

    def _sync_camera(self, target_pos, delta_time):
        if self.camera_prim and self.camera_prim.IsValid():
            self._update_camera_transform(target_pos, delta_time)

    def _get_debug_draw_interface(self):
        if self._debug_draw_iface is not None:
            return self._debug_draw_iface
        try:
            from isaacsim.util.debug_draw import _debug_draw

            self._debug_draw_iface = _debug_draw.acquire_debug_draw_interface()
        except Exception as e:
            if not self._debug_draw_unavailable_logged:
                self._log_event(
                    "debug_draw_unavailable",
                    f"debug draw path disabled; isaacsim.util.debug_draw unavailable: {e}",
                    cooldown=10.0,
                    level="warn",
                )
                self._debug_draw_unavailable_logged = True
            self._debug_draw_iface = None
        return self._debug_draw_iface

    def _clear_debug_path(self):
        iface = self._debug_draw_iface
        if iface is None:
            return
        try:
            iface.clear_lines()
        except Exception:
            pass

    def _debug_point(self, pos, reference_z=None):
        point = np.asarray(pos, dtype=np.float32)
        z = float(point[2]) if point.shape[0] >= 3 else 0.0
        if reference_z is not None:
            z = float(reference_z)
        return (float(point[0]), float(point[1]), z + 0.08)

    def _record_debug_trail_point(self, follower_pos):
        if not self.debug_draw_trail:
            return
        point = np.asarray(follower_pos, dtype=np.float32).copy()
        if not self._debug_trail_points:
            self._debug_trail_points.append(point)
        else:
            last_point = self._debug_trail_points[-1]
            distance = float(np.linalg.norm(point[:2] - last_point[:2]))
            if distance >= float(self.debug_draw_trail_min_distance):
                self._debug_trail_points.append(point)

        max_points = max(int(self.debug_draw_trail_max_points), 2)
        if len(self._debug_trail_points) > max_points:
            self._debug_trail_points = self._debug_trail_points[-max_points:]

    def _debug_path_segments(self, follower_pos, target_pos=None, next_waypoint=None):
        if not self.debug_draw_path:
            return []

        reference_z = float(follower_pos[2])
        color = (1.0, 0.86, 0.05, 1.0)
        thickness = float(self.debug_draw_path_thickness)

        if self.cached_path and len(self.cached_path) >= 2:
            # Draw only the NavMesh route polyline. Do not prepend the current
            # follower pose or controller carrot: either connector is a straight
            # chord that can visually cut across shelf corners even when the
            # underlying cached route is valid.
            start_idx = min(
                max(int(self.cached_waypoint_idx) - 1, 0),
                len(self.cached_path) - 2,
            )
            points = []
            for point in self.cached_path[start_idx:]:
                candidate = np.asarray(point, dtype=np.float32)
                if not points or np.linalg.norm(candidate[:2] - points[-1][:2]) > 0.03:
                    points.append(candidate)
        else:
            return []

        return [
            (
                self._debug_point(points[idx], reference_z=reference_z),
                self._debug_point(points[idx + 1], reference_z=reference_z),
                color,
                thickness,
            )
            for idx in range(len(points) - 1)
        ]

    def _debug_trail_segments(self, reference_z):
        if not self.debug_draw_trail or len(self._debug_trail_points) < 2:
            return []

        color = (0.05, 0.75, 1.0, 1.0)
        thickness = max(float(self.debug_draw_path_thickness) * 0.7, 1.0)
        return [
            (
                self._debug_point(self._debug_trail_points[idx], reference_z=reference_z),
                self._debug_point(self._debug_trail_points[idx + 1], reference_z=reference_z),
                color,
                thickness,
            )
            for idx in range(len(self._debug_trail_points) - 1)
        ]

    def _draw_debug_path(self, follower_pos, target_pos, next_waypoint=None):
        if not self.debug_draw_path and not self.debug_draw_trail:
            return
        iface = self._get_debug_draw_interface()
        if iface is None:
            return

        reference_z = float(follower_pos[2])
        segments = self._debug_path_segments(
            follower_pos,
            target_pos,
            next_waypoint,
        )
        segments.extend(self._debug_trail_segments(reference_z))

        if not segments:
            self._clear_debug_path()
            return

        start_points = [segment[0] for segment in segments]
        end_points = [segment[1] for segment in segments]
        colors = [segment[2] for segment in segments]
        thicknesses = [segment[3] for segment in segments]
        try:
            iface.clear_lines()
            iface.draw_lines(start_points, end_points, colors, thicknesses)
        except Exception as e:
            self._log_event(
                "debug_path_draw_failed",
                f"debug path draw failed: {e}",
                cooldown=2.0,
                level="warn",
            )

    # ------------------------------------------------------------------
    # 调试日志工具
    # ------------------------------------------------------------------

    def _dbg(self, level, message):
        if not self.debug_enabled:
            return
        text = f"[FollowerBehavior v2][debug] {message}"
        if level == "warn":
            carb.log_warn(text)
        elif level == "error":
            carb.log_error(text)
        else:
            carb.log_info(text)

    def _log_event(self, key, message, cooldown=0.6, level="info"):
        if not self.debug_enabled:
            return
        now = self._debug_time
        last = self._event_last_log_time.get(key, -1e9)
        if now - last < cooldown:
            return
        self._event_last_log_time[key] = now
        self._dbg(level, message)

    def _log_state_transition(
        self, state_name, dist_to_human, angle_to_human, angle_to_waypoint
    ):
        if self._last_state == state_name:
            return
        self._last_state = state_name
        wp_text = "n/a" if angle_to_waypoint is None else f"{angle_to_waypoint:.3f}"
        self._dbg(
            "info",
            (
                f"state -> {state_name}, dist={dist_to_human:.3f}, "
                f"ang_human={angle_to_human:.3f}, ang_wp={wp_text}, "
                f"cmd_v={self._last_cmd_linear:.3f}, cmd_w={self._last_cmd_angular:.3f}"
            ),
        )
        self._publish_diagnostics()

    def _set_diagnostic_attr(self, name, value, value_type):
        if not self.follower_prim or not self.follower_prim.IsValid():
            return
        try:
            attr = self.follower_prim.GetAttribute(name)
            if not attr:
                attr = self.follower_prim.CreateAttribute(name, value_type)
            attr.Set(value)
        except Exception as e:
            self._log_event(
                "diagnostic_attr_error",
                f"failed to set {name}: {e}",
                cooldown=2.0,
                level="warn",
            )

    def _publish_diagnostics(self, waypoint=None, snap_rejected=None):
        self._set_diagnostic_attr(
            "follower:last_state",
            self._last_state or "UNKNOWN",
            Sdf.ValueTypeNames.String,
        )
        self._set_diagnostic_attr(
            "follower:last_cmd_linear",
            float(self._last_cmd_linear),
            Sdf.ValueTypeNames.Float,
        )
        self._set_diagnostic_attr(
            "follower:last_cmd_lateral",
            float(self._last_cmd_lateral),
            Sdf.ValueTypeNames.Float,
        )
        self._set_diagnostic_attr(
            "follower:last_cmd_angular",
            float(self._last_cmd_angular),
            Sdf.ValueTypeNames.Float,
        )
        self._set_diagnostic_attr(
            "follower:navmesh_available",
            bool(self.navmesh is not None),
            Sdf.ValueTypeNames.Bool,
        )
        self._set_diagnostic_attr(
            "follower:dynamic_avoidance_enabled",
            bool(self.dynamic_avoidance_enabled),
            Sdf.ValueTypeNames.Bool,
        )
        self._set_diagnostic_attr(
            "follower:nav_path_distance",
            float(self._last_nav_path_distance)
            if self._last_nav_path_distance is not None else -1.0,
            Sdf.ValueTypeNames.Float,
        )
        self._set_diagnostic_attr(
            "follower:tracking_distance",
            float(self._last_tracking_distance)
            if self._last_tracking_distance is not None else -1.0,
            Sdf.ValueTypeNames.Float,
        )
        self._set_diagnostic_attr(
            "follower:path_detour_active",
            bool(self._last_path_detour_active),
            Sdf.ValueTypeNames.Bool,
        )
        self._set_diagnostic_attr(
            "follower:path_replanned_this_frame",
            bool(self._path_replanned_this_frame),
            Sdf.ValueTypeNames.Bool,
        )
        self._set_diagnostic_attr(
            "follower:path_query_failed",
            bool(self._path_query_failed),
            Sdf.ValueTypeNames.Bool,
        )
        self._set_diagnostic_attr(
            "follower:human_visible",
            bool(self.human_in_frame),
            Sdf.ValueTypeNames.Bool,
        )
        self._set_diagnostic_attr(
            "follower:nav_dominant",
            bool(self._nav_dominant),
            Sdf.ValueTypeNames.Bool,
        )
        self._set_diagnostic_attr(
            "follower:snap_slide_count",
            int(self._snap_slide_count),
            Sdf.ValueTypeNames.Int,
        )
        self._set_diagnostic_attr(
            "follower:projection_cycle_count",
            int(self._projection_cycle_count),
            Sdf.ValueTypeNames.Int,
        )
        self._set_diagnostic_attr(
            "follower:effective_path_stop_distance",
            float(self._last_effective_path_stop_distance),
            Sdf.ValueTypeNames.Float,
        )
        self._set_diagnostic_attr(
            "follower:stuck_elapsed",
            float(self._stuck_elapsed),
            Sdf.ValueTypeNames.Float,
        )
        self._set_diagnostic_attr(
            "follower:recovery_active",
            bool(
                self._recovery_waypoint is not None
                and self._debug_time < self._recovery_until
            ),
            Sdf.ValueTypeNames.Bool,
        )
        self._set_diagnostic_attr(
            "follower:stuck_recovery_count",
            int(self._recovery_count),
            Sdf.ValueTypeNames.Int,
        )
        self._set_diagnostic_attr(
            "follower:last_recovery_reason",
            self._last_recovery_reason or "",
            Sdf.ValueTypeNames.String,
        )
        if waypoint is not None:
            self._current_nav_waypoint = waypoint.copy()
            self._set_diagnostic_attr(
                "follower:last_waypoint",
                Gf.Vec3f(float(waypoint[0]), float(waypoint[1]), float(waypoint[2])),
                Sdf.ValueTypeNames.Float3,
            )
        if snap_rejected is not None:
            self._set_diagnostic_attr(
                "follower:snap_rejected",
                bool(snap_rejected),
                Sdf.ValueTypeNames.Bool,
            )

    def _log_periodic(
        self, dist_to_human, angle_to_human, follower_pos, target_pos, waypoint
    ):
        if not self.debug_enabled:
            return
        if self._debug_time < self._next_periodic_log_time:
            return
        self._next_periodic_log_time = self._debug_time + self.debug_period_sec

        wp_text = "none"
        if waypoint is not None:
            wp_text = (
                f"({waypoint[0]:.2f},{waypoint[1]:.2f},{waypoint[2]:.2f})"
            )
        nav_dist = "none"
        if self._last_nav_path_distance is not None:
            nav_dist = f"{self._last_nav_path_distance:.3f}"
        track_dist = "none"
        if self._last_tracking_distance is not None:
            track_dist = f"{self._last_tracking_distance:.3f}"
        self._dbg(
            "info",
            (
                f"tick dist={dist_to_human:.3f}, track={track_dist}, "
                f"nav={nav_dist}, detour={self._last_path_detour_active}, "
                f"visible={self.human_in_frame}, navdom={self._nav_dominant}, "
                f"slides={self._snap_slide_count}, "
                f"ang_h={angle_to_human:.3f}, "
                f"follower=({follower_pos[0]:.2f},{follower_pos[1]:.2f}), "
                f"target=({target_pos[0]:.2f},{target_pos[1]:.2f}), "
                f"wp={wp_text}, cmd_v={self._last_cmd_linear:.3f}, "
                f"cmd_w={self._last_cmd_angular:.3f}"
            ),
        )

    def _distance_to_active_waypoint(self, follower_pos):
        waypoint = self._current_nav_waypoint
        if waypoint is None:
            if not self.cached_path or len(self.cached_path) < 2:
                return None
            idx = int(np.clip(
                self.cached_waypoint_idx,
                0,
                len(self.cached_path) - 1,
            ))
            waypoint = self.cached_path[idx]
        return float(np.linalg.norm(
            np.asarray(waypoint, dtype=np.float32)[:2] - follower_pos[:2]
        ))

    def _update_stuck_detector(self, follower_pos, delta_time, tracking_distance):
        if self._last_follower_pos is None:
            self._last_follower_pos = follower_pos.copy()
            self._last_stuck_tracking_distance = float(tracking_distance)
            self._last_stuck_waypoint_distance = self._distance_to_active_waypoint(
                follower_pos
            )
            self._last_stuck_waypoint_idx = int(self.cached_waypoint_idx)
            return

        moved = np.linalg.norm(follower_pos[:2] - self._last_follower_pos[:2])
        self._last_follower_pos = follower_pos.copy()
        previous_tracking = self._last_stuck_tracking_distance
        self._last_stuck_tracking_distance = float(tracking_distance)
        progress_delta = (
            float(previous_tracking) - float(tracking_distance)
            if previous_tracking is not None else 0.0
        )
        current_waypoint_idx = int(self.cached_waypoint_idx)
        waypoint_distance = self._distance_to_active_waypoint(follower_pos)
        previous_waypoint_distance = self._last_stuck_waypoint_distance
        waypoint_progress = None
        if (
            waypoint_distance is not None
            and previous_waypoint_distance is not None
            and self._last_stuck_waypoint_idx == current_waypoint_idx
        ):
            waypoint_progress = float(previous_waypoint_distance) - float(
                waypoint_distance
            )
        self._last_stuck_waypoint_distance = waypoint_distance
        self._last_stuck_waypoint_idx = current_waypoint_idx

        # 只有在“确实应该动”的情况下才计时
        command_speed = float(np.linalg.norm([
            self._last_cmd_linear,
            self._last_cmd_lateral,
        ]))
        should_move = (
            command_speed > 0.1
            and tracking_distance > self.safe_dis_max + 0.05
        )
        same_waypoint_no_progress = (
            waypoint_progress is not None
            and waypoint_progress < max(0.005, self.stuck_move_eps * 0.5)
        )
        no_route_progress = (
            previous_tracking is not None
            and progress_delta < max(0.01, self.stuck_move_eps * 0.5)
            and (
                same_waypoint_no_progress
                or (waypoint_progress is None
                    and moved < max(self.stuck_move_eps * 1.25,
                                    self.my_radius * 0.04))
            )
        )

        if should_move and (
            moved < self.stuck_move_eps
            or no_route_progress
        ):
            self._stuck_elapsed += delta_time
            if self._stuck_elapsed >= self.stuck_warn_sec:
                self._log_event(
                    "stuck",
                    (
                        f"possible_stuck moved={moved:.4f}, "
                        f"progress={progress_delta:.4f}, "
                        f"wp_progress={waypoint_progress if waypoint_progress is not None else 0.0:.4f}, "
                        f"stuck_t={self._stuck_elapsed:.2f}, "
                        f"cmd_v={self._last_cmd_linear:.3f}, "
                        f"cmd_w={self._last_cmd_angular:.3f}, "
                        f"track={tracking_distance:.3f}"
                    ),
                    cooldown=0.8,
                    level="warn",
                )
                if self._skip_stuck_waypoint(follower_pos):
                    self._stuck_elapsed = 0.0
                    return
            if self._stuck_elapsed >= self.stuck_recovery_sec:
                self._activate_stuck_recovery(follower_pos)
        else:
            self._stuck_elapsed = 0.0

    def _skip_stuck_waypoint(self, follower_pos):
        if not self.cached_path or len(self.cached_path) < 3:
            return False
        if self.cached_waypoint_idx >= len(self.cached_path) - 1:
            return False
        if self._debug_time - self._last_stuck_waypoint_skip_time < 0.8:
            return False

        old_idx = int(self.cached_waypoint_idx)
        points = [np.asarray(p, dtype=np.float32) for p in self.cached_path]
        if self._is_sharp_path_turn(points, old_idx):
            self._log_event(
                "waypoint_skip_sharp_blocked",
                (
                    f"hold sharp stuck waypoint {old_idx}; "
                    "using recovery instead of cutting corner"
                ),
                cooldown=0.8,
                level="warn",
            )
            return False

        next_idx = min(old_idx + 1, len(self.cached_path) - 1)
        min_next_dist = max(
            float(self.waypoint_reach_dist),
            float(self.my_radius) * 1.25,
        )
        while next_idx < len(self.cached_path) - 1:
            next_wp = np.asarray(self.cached_path[next_idx], dtype=np.float32)
            if np.linalg.norm(next_wp[:2] - follower_pos[:2]) >= min_next_dist:
                break
            next_idx += 1

        self.cached_waypoint_idx = next_idx
        self._last_stuck_waypoint_skip_time = self._debug_time
        self._last_stuck_tracking_distance = None
        self._last_stuck_waypoint_distance = None
        self._last_stuck_waypoint_idx = None
        self.prev_body_action[:2] = 0.0
        self.prev_linear_vel = 0.0
        waypoint = np.asarray(
            self.cached_path[self.cached_waypoint_idx],
            dtype=np.float32,
        )
        self._log_event(
            "waypoint_skip_stuck",
            (
                f"skip stuck waypoint {old_idx}->{self.cached_waypoint_idx}, "
                f"follower=({follower_pos[0]:.2f},{follower_pos[1]:.2f}), "
                f"next=({waypoint[0]:.2f},{waypoint[1]:.2f})"
            ),
            cooldown=0.2,
            level="warn",
        )
        self._publish_diagnostics(waypoint=waypoint)
        return True

    def _active_recovery_waypoint(self, follower_pos):
        if self._recovery_waypoint is None:
            return None
        if self._debug_time >= self._recovery_until:
            self._recovery_waypoint = None
            return None
        if np.linalg.norm(follower_pos[:2] - self._recovery_waypoint[:2]) < self.waypoint_reach_dist:
            self._recovery_waypoint = None
            self._last_recovery_reason = "recovery_waypoint_reached"
            return None
        return self._recovery_waypoint

    def _activate_stuck_recovery(self, follower_pos):
        if not self.stuck_recovery_enabled:
            return
        if self._debug_time - self._last_recovery_time < self.stuck_recovery_cooldown:
            return

        recovery_wp = self._find_stuck_recovery_waypoint(follower_pos)
        if recovery_wp is None:
            self._last_recovery_reason = "no_valid_recovery_waypoint"
            self._publish_diagnostics()
            return

        self._recovery_waypoint = recovery_wp
        self._recovery_until = self._debug_time + max(
            self.stuck_recovery_sec, self.stuck_recovery_cooldown
        )
        self._last_recovery_time = self._debug_time
        self._recovery_count += 1
        self._last_recovery_reason = "stuck_static_or_navmesh_local_escape"
        self.cached_path = []
        self.cached_waypoint_idx = 1
        self.last_target_pos = None
        self._last_stuck_tracking_distance = None
        self._last_stuck_waypoint_distance = None
        self._last_stuck_waypoint_idx = None
        self.prev_body_action[:2] = 0.0
        self.prev_linear_vel = 0.0
        self._publish_diagnostics(waypoint=recovery_wp)
        self._log_event(
            "stuck_recovery",
            (
                f"recovery wp=({recovery_wp[0]:.2f},{recovery_wp[1]:.2f}), "
                f"count={self._recovery_count}"
            ),
            cooldown=0.2,
            level="warn",
        )

    def _find_stuck_recovery_waypoint(self, follower_pos):
        if self.navmesh is None:
            return None

        _, follower_rpy = self._get_prim_pose(self.follower_prim)
        yaw = follower_rpy[2] if follower_rpy is not None else 0.0
        move_world = self._body_to_world_velocity(self.prev_body_action[:2], yaw)
        if np.linalg.norm(move_world) < 1e-6 and self._current_nav_waypoint is not None:
            move_world = self._current_nav_waypoint[:2] - follower_pos[:2]
        if np.linalg.norm(move_world) < 1e-6:
            forward, left = self._forward_left_vectors(yaw)
            move_world = forward + left

        move_dir = move_world / max(np.linalg.norm(move_world), 1e-6)
        left_dir = np.array([-move_dir[1], move_dir[0]])
        directions = [
            -move_dir,
            left_dir,
            -left_dir,
            -move_dir + left_dir,
            -move_dir - left_dir,
            move_dir + left_dir,
            move_dir - left_dir,
        ]
        distances = [
            self.stuck_recovery_distance,
            self.stuck_recovery_distance * 1.5,
            max(self.stuck_recovery_distance * 0.5, self.my_radius * 2.0),
        ]

        best = None
        best_score = -1.0
        for distance in distances:
            for direction in directions:
                norm = np.linalg.norm(direction)
                if norm < 1e-6:
                    continue
                direction = direction / norm
                candidate = follower_pos.copy()
                candidate[0] += direction[0] * distance
                candidate[1] += direction[1] * distance
                projected = self._project_to_navmesh(candidate)
                if projected is None:
                    continue
                displacement = np.linalg.norm(projected[:2] - follower_pos[:2])
                snap = np.linalg.norm(projected[:2] - candidate[:2])
                if displacement < max(self.waypoint_reach_dist, self.my_radius * 1.5):
                    continue
                if snap > max(self.max_navmesh_snap * 2.0, 0.35):
                    continue
                path = self._compute_path(follower_pos, projected)
                if not path:
                    continue
                score = displacement - 0.5 * snap
                if score > best_score:
                    best_score = score
                    best = projected
        return best

    # ------------------------------------------------------------------
    # Habitat 工具函数
    # ------------------------------------------------------------------

    @staticmethod
    def _forward_left_vectors(yaw):
        forward = np.array([math.sin(yaw), -math.cos(yaw)], dtype=np.float32)
        left = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        return forward, left

    @classmethod
    def _world_to_body_velocity(cls, world_vel_2d, yaw):
        forward, left = cls._forward_left_vectors(yaw)
        return np.array(
            [
                float(np.dot(world_vel_2d, forward)),
                float(np.dot(world_vel_2d, left)),
            ],
            dtype=np.float32,
        )

    @classmethod
    def _body_to_world_velocity(cls, body_vel_2d, yaw):
        forward, left = cls._forward_left_vectors(yaw)
        return forward * body_vel_2d[0] + left * body_vel_2d[1]

    def _estimate_target_velocity(self, target_pos, delta_time):
        if self._prev_target_motion_pos is None or delta_time <= 0:
            self._prev_target_motion_pos = target_pos.copy()
            return np.array([0.0, 0.0], dtype=np.float32)

        velocity = (
            target_pos[:2] - self._prev_target_motion_pos[:2]
        ) / max(delta_time, 1e-6)
        self._prev_target_motion_pos = target_pos.copy()
        return np.asarray(velocity, dtype=np.float32)

    def _update_visual_tracking_from_bbox(self):
        """Read ActorSDG bbox attrs published on the follower prim."""
        if self.follower_prim is None or not self.follower_prim.IsValid():
            return

        visible_attr = self.follower_prim.GetAttribute("follower:target_bbox_visible")
        center_attr = self.follower_prim.GetAttribute("follower:target_bbox_center_x")
        frame_attr = self.follower_prim.GetAttribute("follower:target_bbox_frame")

        frame_id = None
        if frame_attr and frame_attr.Get() is not None:
            try:
                frame_id = int(frame_attr.Get())
            except Exception:
                frame_id = None

        if frame_id is not None:
            self._last_target_bbox_frame = frame_id

        visible = bool(visible_attr.Get()) if visible_attr and visible_attr.Get() is not None else False

        if visible and center_attr and center_attr.Get() is not None:
            try:
                center_x = float(center_attr.Get())
                if math.isfinite(center_x):
                    self.human_center_x = float(np.clip(center_x, 0.0, 1.0))
                    self.last_seen_center_x = self.human_center_x
                    self.human_in_frame = True
                    return
            except Exception:
                pass

        self.human_in_frame = False
        self.human_center_x = self.last_seen_center_x

    def _is_human_stationary(self):
        if len(self.target_pos_history) < 5:
            return False
        points = np.asarray([p[:2] for p in self.target_pos_history], dtype=np.float32)
        displacement = float(np.linalg.norm(points[-1] - points[0]))
        return displacement < 0.05

    def _target_is_closing_on_follower(self, follower_pos):
        if len(self.target_pos_history) < 2:
            return False
        prev_target = np.asarray(self.target_pos_history[-2][:2], dtype=np.float32)
        curr_target = np.asarray(self.target_pos_history[-1][:2], dtype=np.float32)
        target_step = curr_target - prev_target
        to_follower = np.asarray(follower_pos[:2], dtype=np.float32) - curr_target
        dist = float(np.linalg.norm(to_follower))
        if dist < 1e-6:
            return True
        closing_step = float(np.dot(target_step, to_follower / dist))
        return closing_step > max(0.005, float(self.stuck_move_eps) * 0.25)

    def _should_enter_too_close(
        self, dist_to_human, within_goal_stop_radius, target_closing
    ):
        _ = within_goal_stop_radius
        too_close_distance = self._too_close_distance()
        if dist_to_human < too_close_distance:
            return True
        if target_closing and dist_to_human < self.safe_dis_min:
            return True
        return False

    def _too_close_distance(self):
        safe_min = float(getattr(self, "safe_dis_min", 1.2))
        return float(np.clip(
            float(getattr(self, "too_close_distance", 1.0)),
            0.0,
            safe_min,
        ))

    def _tracking_translation_scale(self, tracking_distance):
        if tracking_distance is None or not math.isfinite(float(tracking_distance)):
            return 1.0

        tracking_distance = float(tracking_distance)
        too_close_distance = self._too_close_distance()
        slow_outer = max(float(self.safe_dis_min), too_close_distance)
        if tracking_distance < slow_outer:
            slow_width = max(slow_outer - too_close_distance, 1e-6)
            ratio = (tracking_distance - too_close_distance) / slow_width
            return float(np.clip(ratio, 0.0, 1.0) ** 1.5)

        full_speed_distance = max(float(self.safe_dis_max), slow_outer + 1e-6)
        chase_ratio = (tracking_distance - slow_outer) / (
            full_speed_distance - slow_outer
        )
        max_chase_scale = 1.0 / 0.75
        return float(
            1.0 + np.clip(chase_ratio, 0.0, 1.0) * (max_chase_scale - 1.0)
        )

    @staticmethod
    def _cross2d(a, b):
        return float(a[0] * b[1] - a[1] * b[0])

    def _visible_target_yaw_error(self, vis_angle, robot_forward_2d, dir_human):
        human_cross = float(np.clip(
            self._cross2d(robot_forward_2d, dir_human),
            -1.0,
            1.0,
        ))
        if not self.human_in_frame:
            return human_cross

        edge_dist = min(self.human_center_x, 1.0 - self.human_center_x)
        bbox_weight = float(np.clip(
            0.80 + 1.5 * max(0.0, 0.3 - edge_dist),
            0.80,
            1.0,
        ))
        return bbox_weight * vis_angle + (1.0 - bbox_weight) * human_cross

    def _historical_trail_waypoint(self, follower_pos):
        """Pick a recent human position ahead as a narrow-passage breadcrumb."""
        if self.navmesh is None or len(self.target_pos_history) < 4:
            return None
        follower_pos = np.asarray(follower_pos, dtype=np.float32)
        min_distance = max(0.35, float(self.my_radius) * 2.0)
        for max_distance in (1.25, 2.0):
            for point in reversed(self.target_pos_history[:-2]):
                point = np.asarray(point, dtype=np.float32)
                distance = float(np.linalg.norm(point[:2] - follower_pos[:2]))
                if distance < min_distance or distance > max_distance:
                    continue
                projected = self._project_to_navmesh(
                    point, agent_radius=self.my_radius
                )
                if projected is None:
                    continue
                snap = float(np.linalg.norm(projected[:2] - point[:2]))
                if snap > float(self.max_navmesh_snap):
                    continue
                projected = np.asarray(projected, dtype=np.float32)
                projected[2] = follower_pos[2]
                trail_path = self._compute_path(
                    follower_pos,
                    projected,
                    agent_radius=self.my_radius,
                )
                if not trail_path or len(trail_path) < 2:
                    continue
                # Follow the validated route to the breadcrumb instead of
                # pointing straight across the inside of a corridor corner.
                waypoint = self._select_carrot_point(
                    follower_pos,
                    trail_path,
                    lookahead_dist=max(0.30, float(self.my_radius) * 2.0),
                    start_segment_idx=0,
                    stop_at_waypoint_idx=1,
                )
                if waypoint is not None:
                    waypoint = np.asarray(waypoint, dtype=np.float32)
                    waypoint[2] = follower_pos[2]
                    return waypoint
        return None

    def _center_corridor_waypoint(self, follower_pos, waypoint):
        """Bias a local waypoint toward equal left/right NavMesh clearance."""
        if self.navmesh is None or waypoint is None:
            return waypoint

        follower_pos = np.asarray(follower_pos, dtype=np.float32)
        waypoint = np.asarray(waypoint, dtype=np.float32)
        route = waypoint[:2] - follower_pos[:2]
        route_norm = float(np.linalg.norm(route))
        if route_norm < 1e-4:
            return waypoint

        forward = route / route_norm
        left = np.array([-forward[1], forward[0]], dtype=np.float32)
        lookahead = float(np.clip(route_norm, 0.30, 0.55))
        probe_center = follower_pos.copy()
        probe_center[:2] += forward * lookahead

        probe_step = 0.05
        max_probe = 0.60
        snap_limit = max(0.025, min(0.06, float(self.my_radius) * 0.25))

        def side_clearance(sign):
            clearance = 0.0
            for offset in np.arange(probe_step, max_probe + 1e-6, probe_step):
                candidate = probe_center.copy()
                candidate[:2] += left * float(sign * offset)
                projected = self._project_to_navmesh(
                    candidate, agent_radius=self.my_radius
                )
                if projected is None:
                    break
                snap = float(np.linalg.norm(projected[:2] - candidate[:2]))
                if snap > snap_limit:
                    break
                clearance = float(offset)
            return clearance

        left_clearance = side_clearance(1.0)
        right_clearance = side_clearance(-1.0)
        corridor_width = left_clearance + right_clearance
        corridor_detected = (
            left_clearance < max_probe - 1e-6
            and right_clearance < max_probe - 1e-6
            and corridor_width <= 1.0
        )
        desired_offset = 0.0
        if corridor_detected:
            desired_offset = float(np.clip(
                0.5 * (left_clearance - right_clearance),
                -0.12,
                0.12,
            ))

        self._corridor_center_offset = (
            0.75 * float(self._corridor_center_offset)
            + 0.25 * desired_offset
        )
        if abs(self._corridor_center_offset) < 0.01:
            return waypoint

        centered = probe_center.copy()
        centered[:2] += left * self._corridor_center_offset
        projected = self._project_to_navmesh(
            centered, agent_radius=self.my_radius
        )
        if projected is None:
            return waypoint
        snap = float(np.linalg.norm(projected[:2] - centered[:2]))
        if snap > snap_limit:
            return waypoint

        centered = np.asarray(projected, dtype=np.float32)
        centered[2] = follower_pos[2]
        self._log_event(
            "corridor_centering",
            (
                f"corridor center: left={left_clearance:.2f}, "
                f"right={right_clearance:.2f}, "
                f"offset={self._corridor_center_offset:+.3f}"
            ),
            cooldown=0.8,
        )
        return centered

    def _compute_habitat_nav_command(
        self, follower_pos, follower_yaw, target_pos, to_human_2d, dist_to_human
    ):
        """Habitat oracle to_navmesh_waypoint mapped to Isaac body velocities."""
        next_waypoint = self._get_next_waypoint(follower_pos, target_pos, dist_to_human)
        path_points = self.cached_path if self.cached_path else None
        if self._consecutive_path_failures >= 3:
            trail_waypoint = self._historical_trail_waypoint(follower_pos)
            if trail_waypoint is not None:
                next_waypoint = trail_waypoint
                path_points = None
                self._current_nav_waypoint = trail_waypoint.copy()
                self._log_event(
                    "historical_trail_fallback",
                    (
                        f"path failed {self._consecutive_path_failures} times; "
                        f"following observed human trail point "
                        f"({trail_waypoint[0]:.2f},{trail_waypoint[1]:.2f})"
                    ),
                    cooldown=0.5,
                    level="warn",
                )
        if next_waypoint is None:
            if self.navmesh is not None:
                # A global path can temporarily fail at a narrow corner or when
                # the moving target is off the clearance mesh. Keep pursuing
                # locally; _project_motion_to_navmesh validates every small
                # displacement and will hold before crossing an obstacle.
                next_waypoint = target_pos.copy()
                self._current_nav_waypoint = next_waypoint.copy()
                self._log_event(
                    "nav_local_pursuit_fallback",
                    "nav path unavailable; using NavMesh-validated local pursuit",
                    cooldown=0.8,
                    level="warn",
                )
                path_points = None
            else:
                next_waypoint = target_pos
                path_points = None
                self._log_event(
                    "nav_fallback_target",
                    "nav path unavailable; using target direction as Habitat fallback",
                    cooldown=0.8,
                    level="warn",
                )

        next_waypoint = self._center_corridor_waypoint(
            follower_pos, next_waypoint
        )
        avoidance_wp = self._compute_avoidance_waypoint(
            follower_pos, follower_yaw, next_waypoint
        )
        if avoidance_wp is not None:
            self._log_event(
                "avoidance_active",
                (
                    f"avoid wp from ({next_waypoint[0]:.2f},{next_waypoint[1]:.2f}) "
                    f"to ({avoidance_wp[0]:.2f},{avoidance_wp[1]:.2f})"
                ),
                cooldown=0.5,
            )
            next_waypoint = avoidance_wp
        self._current_nav_waypoint = next_waypoint.copy()

        robot_forward_2d, robot_left_2d = self._forward_left_vectors(follower_yaw)

        vec_to_nav = np.array([
            next_waypoint[0] - follower_pos[0],
            next_waypoint[1] - follower_pos[1],
        ], dtype=np.float32)
        nav_norm = float(np.linalg.norm(vec_to_nav))
        dir_nav = vec_to_nav / nav_norm if nav_norm > 1e-6 else robot_forward_2d.copy()

        human_norm = float(np.linalg.norm(to_human_2d))
        dir_human = (
            np.asarray(to_human_2d, dtype=np.float32) / human_norm
            if human_norm > 1e-6 else dir_nav.copy()
        )

        target_visual_conf = 1.0 if self.human_in_frame else 0.0
        self.visual_confidence = (
            0.75 * self.visual_confidence + 0.25 * target_visual_conf
        )

        edge_dist = min(self.human_center_x, 1.0 - self.human_center_x)
        alpha_edge = float(np.clip(0.2 + 3.0 * edge_dist, 0.2, 0.9))
        divergence_cos = float(np.clip(np.dot(dir_nav, dir_human), -1.0, 1.0))
        alpha_divergence = float(np.clip(
            0.25 + 0.65 * max(0.0, divergence_cos), 0.25, 0.9
        ))

        e_vis = 0.5 - self.human_center_x
        vis_angle = float(np.clip(e_vis * (np.pi / 2.0), -0.9, 0.9))
        self._raw_human_angle_diff = vis_angle

        if self.human_in_frame:
            dir_vis = robot_forward_2d + math.tan(vis_angle) * robot_left_2d
            vis_norm = float(np.linalg.norm(dir_vis))
            dir_vis = dir_vis / vis_norm if vis_norm > 1e-6 else dir_human.copy()
        else:
            dir_vis = dir_nav.copy()

        tracking_distance = self._get_tracking_distance(dist_to_human)
        self._nav_dominant = bool(
            path_points is not None
            and self._route_requires_nav_dominant(path_points, dist_to_human)
        )

        w_nav = alpha_divergence * alpha_edge * (1.0 - 0.55 * self.visual_confidence)
        if self._nav_dominant:
            w_nav = 1.0
            self._log_event(
                "nav_dominant",
                (
                    f"nav-dominant move: visible={self.human_in_frame}, "
                    f"divergence={divergence_cos:.3f}, "
                    f"path={self._last_nav_path_distance if self._last_nav_path_distance is not None else -1.0:.3f}, "
                    f"straight={dist_to_human:.3f}, tracking={tracking_distance:.3f}, "
                    f"nav_norm={nav_norm:.3f}, visual={self.visual_confidence:.3f}"
                ),
                cooldown=0.5,
            )
        elif path_points is None:
            w_nav = min(0.35, w_nav)
        w_nav = float(np.clip(w_nav, 0.1, 1.0))

        dir_move = w_nav * dir_nav + (1.0 - w_nav) * dir_vis
        move_norm = float(np.linalg.norm(dir_move))
        dir_move = dir_move / move_norm if move_norm > 1e-6 else dir_nav.copy()

        move_alignment = float(np.clip(np.dot(robot_forward_2d, dir_move), -1.0, 1.0))
        lateral_alignment = float(np.clip(np.dot(robot_left_2d, dir_move), -1.0, 1.0))

        nav_cross = float(np.clip(self._cross2d(robot_forward_2d, dir_nav), -1.0, 1.0))
        visual_yaw_error = self._visible_target_yaw_error(
            vis_angle,
            robot_forward_2d,
            dir_human,
        )
        if self.motion_type == "differential":
            # A differential base must point along its translation route. If
            # visual target centering is blended into body yaw at a corner,
            # the speed gate waits for alignment with the NavMesh waypoint
            # while yaw keeps steering back toward the pedestrian. That
            # conflicting pair of objectives produces an in-place spin.
            yaw_error = nav_cross
        elif self._nav_dominant:
            # Omni translation can follow the safe path while the body and
            # camera continue facing the target, including after visual loss.
            yaw_error = visual_yaw_error
        else:
            yaw_error = (
                (1.0 - self.visual_confidence) * nav_cross
                + self.visual_confidence * visual_yaw_error
            )
        derivative_t = yaw_error - self.prev_error_t
        self.prev_error_t = yaw_error

        # Habitat's yaw action sign is adapted to Isaac's +Z counterclockwise yaw.
        yaw_speed = (1.4 * yaw_error + 0.6 * derivative_t) * self.turn_velocity
        min_forward_alignment = 0.2
        if (
            self.motion_type == "omnidirectional"
            and self.human_in_frame
            and abs(lateral_alignment) > max(move_alignment, 0.0)
        ):
            min_forward_alignment = 0.05
        move_speed = 0.75 * self.forward_velocity * max(
            min_forward_alignment,
            move_alignment,
        )
        lateral_speed = 0.75 * self.lateral_velocity * lateral_alignment

        if self.human_in_frame:
            center_error = abs(self.human_center_x - 0.5)
            if center_error > 0.15:
                recenter_scale = float(np.clip(
                    1.0 - 2.0 * (center_error - 0.15), 0.35, 1.0
                ))
                move_speed *= recenter_scale
                lateral_speed *= recenter_scale
                yaw_speed *= 1.0 + min(center_error, 0.5)

        if path_points is not None and len(path_points) >= 3:
            seg_idx = int(np.clip(
                int(self.cached_waypoint_idx) - 1,
                0,
                len(path_points) - 3,
            ))
            seg1 = (
                np.asarray(path_points[seg_idx + 1], dtype=np.float32)
                - np.asarray(path_points[seg_idx], dtype=np.float32)
            )
            seg2 = (
                np.asarray(path_points[seg_idx + 2], dtype=np.float32)
                - np.asarray(path_points[seg_idx + 1], dtype=np.float32)
            )
            seg1_2d = seg1[:2]
            seg2_2d = seg2[:2]
            n1 = float(np.linalg.norm(seg1_2d))
            n2 = float(np.linalg.norm(seg2_2d))
            if n1 > 1e-6 and n2 > 1e-6:
                turn_cos = float(np.clip(np.dot(seg1_2d / n1, seg2_2d / n2), -1.0, 1.0))
                if turn_cos < 0.866:
                    curve_scale = max(0.45, 0.45 + 0.55 * max(0.0, turn_cos))
                    move_speed *= curve_scale
                    lateral_speed *= max(0.55, curve_scale)
                    yaw_speed *= 1.0 + 0.4 * (1.0 - max(turn_cos, 0.0))

        if not self.human_in_frame:
            move_speed = max(move_speed, 0.35 * self.forward_velocity)
            lateral_speed *= 0.6
            yaw_speed *= 1.2

        if self.motion_type == "differential":
            nav_heading_error = abs(math.atan2(nav_cross, move_alignment))
            if nav_heading_error > float(self.turn_and_go_thresh):
                # A differential base cannot translate sideways. Advancing
                # through a sharp heading error cuts corridor corners and can
                # make NavMesh projection alternate between both boundaries.
                # Reduce speed continuously to avoid a visible stop/go camera
                # jerk at the turn-and-go threshold.
                stop_turn_error = max(
                    float(self.turn_and_go_thresh) + 0.1,
                    1.2,
                )
                turn_scale = float(np.clip(
                    (stop_turn_error - nav_heading_error)
                    / max(
                        stop_turn_error - float(self.turn_and_go_thresh),
                        1e-6,
                    ),
                    0.0,
                    1.0,
                ))
                move_speed *= turn_scale
                lateral_speed = 0.0

        cmd = np.array([
            float(np.clip(move_speed, -0.2 * self.forward_velocity, 0.85 * self.forward_velocity)),
            float(np.clip(lateral_speed, -0.75 * self.lateral_velocity, 0.75 * self.lateral_velocity)),
            float(np.clip(yaw_speed, -self.turn_velocity, self.turn_velocity)),
        ], dtype=np.float32)
        return cmd, next_waypoint

    def _compute_habitat_retreat_command(
        self, follower_pos, target_pos, to_human_2d, dist_to_human
    ):
        """Habitat oracle to_navmesh_retreat_waypoint mapped to Isaac body velocities."""
        retreat_ratio = 1.0
        if dist_to_human >= self.danger_dis:
            retreat_ratio = (
                (self.safe_dis_min - dist_to_human)
                / max(self.safe_dis_min - self.danger_dis, 1e-6)
            )
        retreat_gain = float(np.clip(retreat_ratio, 0.2, 1.0))

        retreat_waypoint = self._get_retreat_waypoint(
            follower_pos, target_pos, to_human_2d, dist_to_human
        )
        retreat_translation_enabled = True
        if retreat_waypoint is None:
            if self.navmesh is not None:
                retreat_waypoint = self._get_direct_retreat_waypoint(
                    follower_pos,
                    target_pos,
                    to_human_2d,
                    dist_to_human,
                )
                if retreat_waypoint is None:
                    retreat_waypoint = follower_pos.copy()
                    retreat_translation_enabled = False
                    self._current_nav_waypoint = retreat_waypoint.copy()
                    self._log_event(
                        "retreat_hold_no_safe_path",
                        "retreat path unavailable; rotating toward target without unvalidated translation",
                        cooldown=0.8,
                        level="warn",
                    )
            else:
                retreat_dir = (
                    -np.asarray(to_human_2d, dtype=np.float32)
                    / max(dist_to_human, 1e-6)
                )
                retreat_waypoint = follower_pos.copy()
                retreat_waypoint[0] += retreat_dir[0] * max(
                    self.safe_dis_min + 0.1 - dist_to_human, 0.35
                )
                retreat_waypoint[1] += retreat_dir[1] * max(
                    self.safe_dis_min + 0.1 - dist_to_human, 0.35
                )

        _, follower_rpy = self._get_prim_pose(self.follower_prim)
        follower_yaw = follower_rpy[2] if follower_rpy is not None else 0.0

        avoidance_wp = self._compute_avoidance_waypoint(
            follower_pos, follower_yaw, retreat_waypoint
        )
        if avoidance_wp is not None:
            self._log_event(
                "retreat_avoidance_active",
                (
                    f"avoid retreat wp from ({retreat_waypoint[0]:.2f},"
                    f"{retreat_waypoint[1]:.2f}) to ({avoidance_wp[0]:.2f},"
                    f"{avoidance_wp[1]:.2f})"
                ),
                cooldown=0.5,
            )
            retreat_waypoint = avoidance_wp

        robot_forward_2d, robot_left_2d = self._forward_left_vectors(follower_yaw)

        vec_to_retreat = np.array([
            retreat_waypoint[0] - follower_pos[0],
            retreat_waypoint[1] - follower_pos[1],
        ], dtype=np.float32)
        retreat_norm = float(np.linalg.norm(vec_to_retreat))
        if retreat_norm > 1e-6:
            dir_retreat = vec_to_retreat / retreat_norm
        else:
            dir_retreat = -np.asarray(to_human_2d, dtype=np.float32) / max(dist_to_human, 1e-6)

        target_visual_conf = 1.0 if self.human_in_frame else 0.0
        self.visual_confidence = 0.8 * self.visual_confidence + 0.2 * target_visual_conf

        human_norm = float(np.linalg.norm(to_human_2d))
        dir_human = (
            np.asarray(to_human_2d, dtype=np.float32) / human_norm
            if human_norm > 1e-6 else -dir_retreat.copy()
        )

        e_vis = 0.5 - self.human_center_x
        vis_angle = float(np.clip(e_vis * (np.pi / 2.0), -0.9, 0.9))
        self._raw_human_angle_diff = vis_angle

        human_cross = float(np.clip(self._cross2d(robot_forward_2d, dir_human), -1.0, 1.0))
        yaw_error = (
            0.65 * vis_angle + 0.35 * human_cross
            if self.human_in_frame else human_cross
        )
        derivative_t = yaw_error - self.prev_error_t
        self.prev_error_t = yaw_error

        yaw_speed = (1.6 * yaw_error + 0.65 * derivative_t) * self.turn_velocity
        move_alignment = float(np.clip(np.dot(robot_forward_2d, dir_retreat), -1.0, 1.0))
        lateral_alignment = float(np.clip(np.dot(robot_left_2d, dir_retreat), -1.0, 1.0))

        if retreat_translation_enabled:
            cmd = np.array([
                -0.85 * self.forward_velocity * retreat_gain * max(0.0, -move_alignment),
                0.7 * self.lateral_velocity * retreat_gain * lateral_alignment,
                yaw_speed,
            ], dtype=np.float32)
        else:
            cmd = np.array([0.0, 0.0, yaw_speed], dtype=np.float32)
        cmd[0] = float(np.clip(cmd[0], -0.85 * self.forward_velocity, 0.0))
        cmd[1] = float(np.clip(cmd[1], -0.7 * self.lateral_velocity, 0.7 * self.lateral_velocity))
        cmd[2] = float(np.clip(cmd[2], -self.turn_velocity, self.turn_velocity))
        return cmd, retreat_waypoint

    def _get_direct_retreat_waypoint(
        self, follower_pos, target_pos, to_human_2d, dist_to_human
    ):
        """Return a short NavMesh-projected retreat step when full path planning fails."""
        if dist_to_human <= 1e-6:
            return None
        retreat_dir = (
            -np.asarray(to_human_2d, dtype=np.float32)
            / max(float(dist_to_human), 1e-6)
        )
        step = float(
            np.clip(
                self.safe_dis_min + 0.1 - dist_to_human,
                max(0.25, self.my_radius * 0.75),
                min(0.55, max(self.max_navmesh_snap * 2.0, self.my_radius * 1.5)),
            )
        )
        candidate = np.asarray(follower_pos, dtype=np.float32).copy()
        candidate[0] += retreat_dir[0] * step
        candidate[1] += retreat_dir[1] * step

        projected = self._project_to_navmesh(candidate)
        if projected is None:
            return None

        snap = float(np.linalg.norm(projected[:2] - candidate[:2]))
        sep_gain = float(np.linalg.norm(projected[:2] - target_pos[:2])) - float(dist_to_human)
        if snap > max(float(self.max_navmesh_snap), self.my_radius * 0.75):
            return None
        if sep_gain <= max(0.015, self.stuck_move_eps):
            return None

        self._log_event(
            "retreat_direct_fallback",
            (
                f"direct retreat fallback step={step:.3f}, snap={snap:.3f}, "
                f"sep_gain={sep_gain:.3f}"
            ),
            cooldown=0.5,
            level="warn",
        )
        return self._preserve_follower_center_z(projected, follower_pos)

    def _apply_body_command(self, body_cmd, face_target_2d, delta_time, yaw):
        body_cmd = np.asarray(body_cmd, dtype=np.float32)
        if self.motion_type == "omnidirectional":
            world_vel = self._body_to_world_velocity(body_cmd[:2], yaw)
            self._apply_velocity_omni(
                world_vel,
                face_target_2d,
                delta_time,
                yaw,
                angular_vel=float(body_cmd[2]),
            )
        else:
            self._apply_velocity(
                float(body_cmd[0]),
                float(body_cmd[2]),
                delta_time,
                yaw,
            )

    def _smooth_body_action(self, raw_action, delta_time, alpha=None):
        """Habitat oracle 风格 EMA + 加速度限制，减少帧间波动。"""
        raw = np.asarray(raw_action, dtype=np.float32)
        if alpha is None:
            alpha = np.array([0.45, 0.35, 0.5], dtype=np.float32)
        else:
            alpha = np.asarray(alpha, dtype=np.float32)

        smoothed = alpha * raw + (1.0 - alpha) * self.prev_body_action
        max_delta = np.array(
            [
                self.max_linear_accel * delta_time,
                self.max_lateral_accel * delta_time,
                self.max_angular_accel * delta_time,
            ],
            dtype=np.float32,
        )
        delta = np.clip(smoothed - self.prev_body_action, -max_delta, max_delta)
        cmd = self.prev_body_action + delta
        cmd[0] = np.clip(cmd[0], -self.forward_velocity, self.forward_velocity)
        cmd[1] = np.clip(cmd[1], -self.lateral_velocity, self.lateral_velocity)
        cmd[2] = np.clip(cmd[2], -self.turn_velocity, self.turn_velocity)
        self.prev_body_action = cmd
        return cmd

    def _select_carrot_point(
        self,
        follower_pos,
        path_points,
        lookahead_dist,
        start_segment_idx=None,
        stop_at_waypoint_idx=None,
    ):
        """沿 NavMesh polyline 选前瞻点，并保持路径进度单调。

        转角两侧在世界坐标里可能很近，但中间隔着货架/墙体。只按欧氏
        距离在整条路径上找最近投影时，前瞻点会跳到障碍物另一侧，导致
        follower 直指边缘。缓存路径跟随时从当前 segment 投影，sharp
        corner 前不跨过当前 waypoint。
        """
        if not path_points or len(path_points) < 2:
            return None

        points = [np.asarray(p, dtype=np.float32) for p in path_points]
        if start_segment_idx is None:
            segment_start = 0
            segment_end = len(points) - 1
        else:
            segment_start = int(np.clip(start_segment_idx, 0, len(points) - 2))
            segment_end = segment_start + 1

        follower_xy = follower_pos[:2]
        best_idx = segment_start
        best_dist = float("inf")
        best_proj = points[segment_start].copy()

        for idx in range(segment_start, segment_end):
            start = points[idx]
            end = points[idx + 1]
            segment = end[:2] - start[:2]
            seg_len_sq = float(np.dot(segment, segment))
            if seg_len_sq < 1e-8:
                continue
            t = float(np.clip(
                np.dot(follower_xy - start[:2], segment) / seg_len_sq,
                0.0,
                1.0,
            ))
            proj = start + (end - start) * t
            dist = float(np.linalg.norm(follower_xy - proj[:2]))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
                best_proj = proj

        remaining = max(float(lookahead_dist), 0.0)
        cursor = best_proj
        for point_idx in range(best_idx + 1, len(points)):
            point_np = points[point_idx]
            segment = point_np[:2] - cursor[:2]
            seg_len = float(np.linalg.norm(segment))
            if seg_len < 1e-6:
                cursor = point_np
                continue
            if remaining <= seg_len:
                ratio = remaining / seg_len
                return cursor + (point_np - cursor) * ratio
            if (
                stop_at_waypoint_idx is not None
                and point_idx == stop_at_waypoint_idx
                and point_idx < len(points) - 1
            ):
                return point_np.copy()
            remaining -= seg_len
            cursor = point_np
        return points[-1]

    def _is_sharp_path_turn(self, points, waypoint_idx):
        if waypoint_idx <= 0 or waypoint_idx >= len(points) - 1:
            return False
        prev_vec = points[waypoint_idx][:2] - points[waypoint_idx - 1][:2]
        next_vec = points[waypoint_idx + 1][:2] - points[waypoint_idx][:2]
        prev_len = float(np.linalg.norm(prev_vec))
        next_len = float(np.linalg.norm(next_vec))
        if prev_len < 1e-6 or next_len < 1e-6:
            return False
        turn_cos = float(np.clip(
            np.dot(prev_vec / prev_len, next_vec / next_len),
            -1.0,
            1.0,
        ))
        return turn_cos < 0.866

    def _waypoint_reach_threshold(self, points, waypoint_idx):
        """Use a tighter reach threshold only at real corners."""
        threshold = float(self.waypoint_reach_dist)
        if waypoint_idx < len(points) - 1 and self._is_sharp_path_turn(
            points,
            waypoint_idx,
        ):
            threshold = min(
                threshold,
                max(0.08, float(self.my_radius) * 0.25),
            )
        elif waypoint_idx < len(points) - 1:
            threshold = max(
                threshold,
                min(
                    0.6,
                    max(float(self.my_radius) * 1.5, float(self.planning_radius)),
                ),
            )
        return threshold

    def _remaining_path_length_from_pos(
        self, follower_pos, path_points, start_segment_idx=None
    ):
        """Approximate remaining 2D length from follower projection to path end."""
        if not path_points or len(path_points) < 2:
            return None

        points = [np.asarray(p, dtype=np.float32) for p in path_points]
        if start_segment_idx is None:
            segment_start = 0
            segment_end = len(points) - 1
        else:
            segment_start = int(np.clip(start_segment_idx, 0, len(points) - 2))
            segment_end = segment_start + 1

        follower_xy = follower_pos[:2]
        best_idx = segment_start
        best_dist = float("inf")
        best_proj = points[segment_start].copy()

        for idx in range(segment_start, segment_end):
            start = points[idx]
            end = points[idx + 1]
            segment = end[:2] - start[:2]
            seg_len_sq = float(np.dot(segment, segment))
            if seg_len_sq < 1e-8:
                continue
            t = float(np.clip(
                np.dot(follower_xy - start[:2], segment) / seg_len_sq,
                0.0,
                1.0,
            ))
            proj = start + (end - start) * t
            dist = float(np.linalg.norm(follower_xy - proj[:2]))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
                best_proj = proj

        remaining = 0.0
        cursor = best_proj
        for point_np in points[best_idx + 1:]:
            remaining += float(np.linalg.norm(point_np[:2] - cursor[:2]))
            cursor = point_np
        return remaining

    def _is_detour_path(self, path_distance, straight_distance):
        if path_distance is None or straight_distance is None:
            return False
        if not math.isfinite(path_distance) or not math.isfinite(straight_distance):
            return False
        if path_distance <= 0.0:
            return False
        straight_distance = max(float(straight_distance), 1e-6)
        extra_distance = float(path_distance) - straight_distance
        min_extra = max(0.4, float(self.my_radius) * 1.25)
        return (
            extra_distance > min_extra
            and float(path_distance) > straight_distance * 1.2
        )

    def _is_constrained_shelf_route(self, path_distance, straight_distance):
        """Detect shelf-edge routes that need NavMesh-dominant tracking."""
        if self._is_detour_path(path_distance, straight_distance):
            return True
        if path_distance is None or straight_distance is None:
            return False
        if not math.isfinite(path_distance) or not math.isfinite(straight_distance):
            return False
        straight_distance = max(float(straight_distance), 1e-6)
        extra_distance = float(path_distance) - straight_distance
        return (
            extra_distance > max(0.25, float(self.my_radius) * 0.75)
            and float(path_distance) > straight_distance * 1.12
        )

    def _path_has_sharp_turn(self, path_points):
        if not path_points or len(path_points) < 3:
            return False
        points = [np.asarray(p, dtype=np.float32) for p in path_points]
        for idx in range(1, len(points) - 1):
            if self._is_sharp_path_turn(points, idx):
                return True
        return False

    def _choose_path_stop_distance(self, path_distance, straight_distance):
        """Return the goal pause radius; the planned path itself stays untrimmed."""
        return self._goal_stop_radius()

    def _goal_stop_radius(self):
        return max(float(self.path_stop_distance), 0.0)

    def _should_pause_for_goal_radius(self, tracking_distance):
        if tracking_distance is None or not math.isfinite(float(tracking_distance)):
            return False
        stop_radius = self._goal_stop_radius()
        return stop_radius > 0.0 and float(tracking_distance) <= stop_radius

    def _get_tracking_distance(self, straight_distance):
        path_distance = self._last_nav_path_distance
        if (
            path_distance is not None
            and math.isfinite(path_distance)
            and (
                self._last_path_detour_active
                or self._is_constrained_shelf_route(path_distance, straight_distance)
            )
        ):
            self._last_tracking_distance = max(float(straight_distance), float(path_distance))
        else:
            self._last_tracking_distance = float(straight_distance)
        return self._last_tracking_distance

    def _route_requires_nav_dominant(self, path_points, straight_distance):
        """Use pure NavMesh steering when visual tracking can cut a corner."""
        if not self.human_in_frame:
            return True
        if self._last_path_detour_active:
            return True
        if self._is_constrained_shelf_route(
            self._last_nav_path_distance,
            straight_distance,
        ):
            return True
        return self._path_has_sharp_turn(path_points)

    @staticmethod
    def _habitat_lookahead_distance(distance):
        return float(np.clip(0.8 + 0.35 * float(distance), 0.8, 1.6))

    def _densify_path(self, path_points, spacing=0.1):
        """Resample a validated NavMesh polyline without smoothing across corners."""
        if not path_points or len(path_points) < 2:
            return path_points

        dense = [np.asarray(path_points[0], dtype=np.float32).copy()]
        spacing = max(float(spacing), 1e-3)
        for idx in range(1, len(path_points)):
            start = np.asarray(path_points[idx - 1], dtype=np.float32)
            end = np.asarray(path_points[idx], dtype=np.float32)
            delta = end - start
            seg_len = float(np.linalg.norm(delta[:2]))
            if seg_len < 1e-6:
                continue
            steps = max(1, int(math.ceil(seg_len / spacing)))
            for step in range(1, steps + 1):
                point = start + delta * (float(step) / float(steps))
                if np.linalg.norm(point[:2] - dense[-1][:2]) > 1e-4:
                    dense.append(point.astype(np.float32))
        return dense

    def _route_progress_segment_idx(self, path_points):
        if not path_points or len(path_points) < 2:
            return 0
        return int(np.clip(
            int(self.cached_waypoint_idx) - 1,
            0,
            len(path_points) - 2,
        ))

    def _initial_waypoint_idx_for_path(self, follower_pos, path_points):
        """Skip only duplicate start points after replanning from current pose.

        The route is recomputed every frame and then densified, so using the
        normal waypoint reach radius here can skip the next shelf-corner point
        and make the carrot selector start after the corner.
        """
        if not path_points or len(path_points) < 2:
            return 1

        idx = 1
        points = [np.asarray(p, dtype=np.float32) for p in path_points]
        initial_skip_threshold = max(
            0.03,
            min(0.15, float(self.waypoint_reach_dist) * 0.5),
        )
        while idx < len(points) - 1:
            if self._is_sharp_path_turn(points, idx):
                break
            dist_to_wp = float(np.linalg.norm(follower_pos[:2] - points[idx][:2]))
            if dist_to_wp > initial_skip_threshold:
                break
            idx += 1
        return idx

    def _should_replan_for_target_motion(self, follower_pos, target_moved):
        """Avoid resetting route progress for small target motion during detours."""
        if target_moved <= self.path_replan_dist:
            return False, None

        if not self.cached_path or len(self.cached_path) < 2:
            return True, f"target_moved={target_moved:.3f}"

        progress_segment_idx = self._route_progress_segment_idx(self.cached_path)
        remaining_route = self._remaining_path_length_from_pos(
            follower_pos,
            self.cached_path,
            start_segment_idx=progress_segment_idx,
        )

        # At shelf corners the target often advances just over the threshold
        # while the follower still has a valid route around the obstacle. Keep
        # that route instead of rebuilding it and resetting waypoint progress.
        if (
            self._last_path_detour_active
            and remaining_route is not None
            and remaining_route > max(self.waypoint_reach_dist, self.my_radius)
            and target_moved <= max(self.path_replan_dist * 2.0, self.safe_dis_max)
        ):
            self._log_event(
                "replan_deferred",
                (
                    f"defer replan target_moved={target_moved:.3f}, "
                    f"remaining={remaining_route:.3f}"
                ),
                cooldown=0.8,
            )
            return False, None

        return True, f"target_moved={target_moved:.3f}"

    def _cached_waypoint_opposes_target(
        self,
        follower_pos,
        target_pos,
        dist_to_human,
        consider_sharp=True,
    ):
        """Detect a stale straight-route waypoint left behind the follower."""
        if not self.cached_path or len(self.cached_path) < 2:
            return False

        idx = int(np.clip(
            int(self.cached_waypoint_idx),
            0,
            len(self.cached_path) - 1,
        ))
        points = [np.asarray(p, dtype=np.float32) for p in self.cached_path]
        waypoint = points[idx]
        to_waypoint = waypoint[:2] - follower_pos[:2]
        waypoint_dist = float(np.linalg.norm(to_waypoint))
        if waypoint_dist <= self._waypoint_reach_threshold(points, idx):
            return False

        route_requires_nav = (
            self._last_path_detour_active
            or self._is_constrained_shelf_route(
                self._last_nav_path_distance,
                dist_to_human,
            )
            or (consider_sharp and self._path_has_sharp_turn(points))
        )
        if route_requires_nav:
            return False

        to_target = target_pos[:2] - follower_pos[:2]
        target_dist = float(np.linalg.norm(to_target))
        if target_dist < 1e-6 or waypoint_dist < 1e-6:
            return False

        alignment = float(np.dot(
            to_waypoint / waypoint_dist,
            to_target / target_dist,
        ))
        if alignment >= -0.25:
            return False

        self._log_event(
            "stale_waypoint_opposes_target",
            (
                f"cached waypoint opposes target: idx={idx}, "
                f"wp_dist={waypoint_dist:.3f}, align={alignment:.3f}; replanning"
            ),
            cooldown=0.5,
            level="warn",
        )
        return True

    @staticmethod
    def _compute_turn(target_2d, turn_speed, robot_forward_2d):
        """比例转向：角度差大时快转，接近目标时慢转，消除 bang-bang 抖动。

        角度差 > 1.0 rad (~57°) → 全速
        角度差 < 1.0 rad → 线性衰减（最低 10% 速度）
        叉积判断左转/右转。
        """
        # 计算角度差
        n1 = np.linalg.norm(robot_forward_2d)
        n2 = np.linalg.norm(target_2d)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos_val = np.clip(
            np.dot(robot_forward_2d, target_2d) / (n1 * n2), -1.0, 1.0
        )
        angle = math.acos(cos_val)

        # 比例缩放：角度越小速度越低
        ramp_zone = 1.0  # 1 rad 内开始减速
        if angle > ramp_zone:
            scale = 1.0
        else:
            scale = max(angle / ramp_zone, 0.1)

        speed = turn_speed * scale

        # 叉积判断方向
        cross = (
            robot_forward_2d[0] * target_2d[1]
            - robot_forward_2d[1] * target_2d[0]
        )
        return speed if cross > 0 else -speed

    @staticmethod
    def _get_angle(v1, v2):
        """计算两个 2D 向量的夹角（弧度，无符号）。"""
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return math.acos(cos_val)

    @staticmethod
    def _clamp_accel(target_vel, prev_vel, max_accel, dt):
        """限制速度变化率，防止瞬间跳变。"""
        if dt <= 0:
            return target_vel
        max_delta = max_accel * dt
        delta = target_vel - prev_vel
        if delta > max_delta:
            return prev_vel + max_delta
        if delta < -max_delta:
            return prev_vel - max_delta
        return target_vel

    def _project_to_navmesh(self, pos, agent_radius=None):
        """将任意点投影到 NavMesh；失败时返回 None。"""
        if self.navmesh is None:
            return None
        radius = self.my_radius if agent_radius is None else float(agent_radius)
        try:
            projected = self.navmesh.query_closest_point(
                carb.Float3(pos[0], pos[1], pos[2]),
                agent_radius=radius,
            )
            if not projected:
                return None
            pt = projected[0]
            return np.array([pt.x, pt.y, pt.z])
        except Exception:
            return None

    def _path_segments_stay_on_navmesh(self, path_points, agent_radius=None):
        """Reject polylines whose straight segments leave the walkable NavMesh."""
        if self.navmesh is None:
            return True
        if not path_points or len(path_points) < 2:
            return False

        radius = self.my_radius if agent_radius is None else float(agent_radius)
        sample_spacing = max(0.08, min(0.25, radius * 0.5))
        snap_limit = max(
            0.04,
            min(float(self.max_navmesh_snap) * 0.35, radius * 0.5),
        )
        points = [np.asarray(point, dtype=np.float32) for point in path_points]
        for idx in range(len(points) - 1):
            start = points[idx]
            end = points[idx + 1]
            delta = end[:2] - start[:2]
            seg_len = float(np.linalg.norm(delta))
            if seg_len < 1e-6:
                continue
            steps = max(2, int(math.ceil(seg_len / sample_spacing)))
            for step in range(1, steps):
                t = float(step) / float(steps)
                sample = start + (end - start) * t
                projected = self._project_to_navmesh(
                    sample,
                    agent_radius=radius,
                )
                if projected is None:
                    self._log_event(
                        "path_segment_off_navmesh",
                        f"path segment off NavMesh: idx={idx}, t={t:.2f}, no projection",
                        cooldown=0.8,
                        level="warn",
                    )
                    return False
                snap = float(np.linalg.norm(projected[:2] - sample[:2]))
                if snap > snap_limit:
                    self._log_event(
                        "path_segment_off_navmesh",
                        (
                            f"path segment off NavMesh: idx={idx}, t={t:.2f}, "
                            f"snap={snap:.3f}, limit={snap_limit:.3f}"
                        ),
                        cooldown=0.8,
                        level="warn",
                    )
                    return False
        return True

    def _motion_center_z(self, reference_pos=None):
        if self._follower_center_z is not None:
            return float(self._follower_center_z)
        if reference_pos is not None and len(reference_pos) >= 3:
            return float(reference_pos[2])
        return 0.0

    def _preserve_follower_center_z(self, pos, reference_pos=None):
        result = np.asarray(pos, dtype=np.float32).copy()
        if result.shape[0] >= 3:
            result[2] = self._motion_center_z(reference_pos)
        return result

    def _motion_route_progress(self, current_pos, candidate_pos):
        """Positive when a projected motion advances along the active route."""
        progress_values = []
        if self._current_nav_waypoint is not None:
            waypoint = np.asarray(self._current_nav_waypoint, dtype=np.float32)
            before = float(np.linalg.norm(waypoint[:2] - current_pos[:2]))
            after = float(np.linalg.norm(waypoint[:2] - candidate_pos[:2]))
            progress_values.append(before - after)

        if self.cached_path and len(self.cached_path) >= 2:
            segment_idx = self._route_progress_segment_idx(self.cached_path)
            before_remaining = self._remaining_path_length_from_pos(
                current_pos,
                self.cached_path,
                start_segment_idx=segment_idx,
            )
            after_remaining = self._remaining_path_length_from_pos(
                candidate_pos,
                self.cached_path,
                start_segment_idx=segment_idx,
            )
            if before_remaining is not None and after_remaining is not None:
                progress_values.append(float(before_remaining) - float(after_remaining))

        if not progress_values:
            return None
        return max(progress_values)

    @staticmethod
    def _limit_projected_step(current_pos, projected_pos, max_step):
        """Move toward a projected point without accepting the full snap at once."""
        current_pos = np.asarray(current_pos, dtype=np.float32)
        projected_pos = np.asarray(projected_pos, dtype=np.float32)
        displacement = projected_pos[:2] - current_pos[:2]
        moved = float(np.linalg.norm(displacement))
        if moved <= max_step or moved < 1e-6:
            return projected_pos.copy(), moved

        result = current_pos.copy()
        result[:2] = current_pos[:2] + displacement / moved * float(max_step)
        return result, float(max_step)

    def _max_projected_motion_step(self, commanded_step_len):
        """Bound NavMesh projection movement to the step the controller commanded."""
        step_len = max(float(commanded_step_len), 0.0)
        if step_len <= 1e-6:
            return 0.0
        slack = min(float(self.stuck_move_eps) * 0.25, step_len)
        return max(step_len * 1.25, step_len + slack)

    def _reset_snap_slide_count(self):
        self._snap_slide_count = 0

    def _record_snap_slide(self, mode, step_len, current_pos, best, best_progress):
        self._snap_slide_count = int(getattr(self, "_snap_slide_count", 0)) + 1
        moved = float(np.linalg.norm(best[:2] - current_pos[:2]))
        self._log_event(
            f"snap_slide_{mode}",
            (
                f"snap slide({mode}): step={step_len:.3f}, "
                f"moved={moved:.3f}, "
                f"route_progress={best_progress if best_progress is not None else 0.0:.3f}, "
                f"count={self._snap_slide_count}"
            ),
            cooldown=0.5,
            level="warn",
        )
        if self._snap_slide_count >= int(self.snap_slide_hold_threshold):
            self._log_event(
                f"snap_slide_sustained_{mode}",
                (
                    f"snap slide sustained({mode}): "
                    f"count={self._snap_slide_count}; keeping current route"
                ),
                cooldown=0.5,
                level="warn",
            )
        return False

    def _is_immediate_projection_return(
        self, previous_origin, current_pos, candidate_pos
    ):
        """Reject A->B->A NavMesh projection cycles while actively navigating."""
        if previous_origin is None or self._last_state != "NAV_OMNI":
            return False
        previous_origin = np.asarray(previous_origin, dtype=np.float32)
        current_pos = np.asarray(current_pos, dtype=np.float32)
        candidate_pos = np.asarray(candidate_pos, dtype=np.float32)
        previous_step = current_pos[:2] - previous_origin[:2]
        candidate_step = candidate_pos[:2] - current_pos[:2]
        previous_len = float(np.linalg.norm(previous_step))
        candidate_len = float(np.linalg.norm(candidate_step))
        min_cycle_step = max(0.01, float(self.stuck_move_eps) * 0.5)
        if previous_len < min_cycle_step or candidate_len < min_cycle_step:
            return False
        return_tolerance = max(
            float(self.stuck_move_eps) * 2.0,
            float(self.my_radius) * 0.3,
        )
        returns_to_origin = (
            float(np.linalg.norm(candidate_pos[:2] - previous_origin[:2]))
            <= return_tolerance
        )
        direction_cos = float(np.dot(previous_step, candidate_step)) / max(
            previous_len * candidate_len, 1e-6
        )
        return returns_to_origin and direction_cos <= -0.8

    def _invalidate_oscillating_route(
        self, mode, previous_origin, current_pos, candidate_pos
    ):
        self.cached_path = []
        self.cached_waypoint_idx = 0
        self.last_target_pos = None
        self._current_nav_waypoint = None
        self._last_path_detour_active = False
        self._path_query_failed = True
        self._projection_cycle_count += 1
        if self._projection_cycle_count >= 2:
            # Let the regular stuck handler select a non-local recovery point
            # on its next update instead of spending another second bouncing
            # between the same two NavMesh projections.
            self._stuck_elapsed = max(
                float(self._stuck_elapsed), float(self.stuck_recovery_sec)
            )
        self._reset_snap_slide_count()
        self._log_event(
            f"projection_two_point_cycle_{mode}",
            (
                f"reject A-B-A projection cycle({mode}): "
                f"A=({previous_origin[0]:.3f},{previous_origin[1]:.3f}), "
                f"B=({current_pos[0]:.3f},{current_pos[1]:.3f}), "
                f"candidate=({candidate_pos[0]:.3f},{candidate_pos[1]:.3f}); "
                "invalidating route"
            ),
            cooldown=0.5,
            level="warn",
        )

    def _project_motion_to_navmesh(self, current_pos, desired_pos, desired_dir, mode):
        """Constrain motion to NavMesh while preserving progress near edges.

        A tiny step beside shelves can project back to the same closest point,
        leaving a valid velocity command with zero XY motion. When that happens,
        probe slightly farther along the waypoint/command direction and use the
        closest reachable point if it makes forward progress.
        """
        if self.navmesh is None:
            self._reset_snap_slide_count()
            return self._preserve_follower_center_z(desired_pos, current_pos), False

        desired_pos = np.asarray(desired_pos, dtype=np.float32)
        current_pos = np.asarray(current_pos, dtype=np.float32)
        previous_origin = self._previous_projection_origin
        self._previous_projection_origin = current_pos.copy()
        desired_pos = self._preserve_follower_center_z(desired_pos, current_pos)
        current_pos = self._preserve_follower_center_z(current_pos, current_pos)
        desired_step = desired_pos[:2] - current_pos[:2]
        step_len = float(np.linalg.norm(desired_step))
        min_motion_step = max(0.002, float(self.stuck_move_eps) * 0.2)
        if step_len < min_motion_step:
            self._reset_snap_slide_count()
            return self._preserve_follower_center_z(current_pos, current_pos), False

        snapped_pos = self._project_to_navmesh(desired_pos)
        snap_error = (
            float(np.linalg.norm(snapped_pos[:2] - desired_pos[:2]))
            if snapped_pos is not None else float("inf")
        )
        max_projected_step = self._max_projected_motion_step(step_len)
        snap_step_limit = max(
            float(self.stuck_move_eps),
            step_len * 1.5,
            min(float(self.max_navmesh_snap) * 0.25, float(self.my_radius) * 0.2),
        )
        snap_ok = (
            snapped_pos is not None
            and snap_error <= self.max_navmesh_snap
        )
        min_progress = min(
            max_projected_step,
            max(0.001, step_len * 0.25),
        )
        if snap_ok and snap_error <= snap_step_limit:
            projected_step = float(np.linalg.norm(snapped_pos[:2] - current_pos[:2]))
            route_progress = self._motion_route_progress(current_pos, snapped_pos)
            regress_limit = -max(0.01, step_len * 0.5)
            route_ok = route_progress is None or route_progress >= regress_limit
            step_ok = projected_step <= max_projected_step
            immediate_return = self._is_immediate_projection_return(
                previous_origin, current_pos, snapped_pos
            )
            if immediate_return:
                self._log_event(
                    f"projection_return_candidate_{mode}",
                    "ignoring NavMesh candidate that returns to the previous origin",
                    cooldown=0.5,
                    level="warn",
                )
            if (
                (projected_step >= min_progress or step_len < 1e-6)
                and route_ok
                and step_ok
                and not immediate_return
            ):
                result = desired_pos.copy()
                result[0] = snapped_pos[0]
                result[1] = snapped_pos[1]
                self._reset_snap_slide_count()
                return self._preserve_follower_center_z(result, current_pos), False
            if not step_ok:
                self._log_event(
                    f"snap_step_large_{mode}",
                    (
                        f"snap step large({mode}): moved={projected_step:.3f}, "
                        f"step={step_len:.3f}, max={max_projected_step:.3f}"
                    ),
                    cooldown=0.5,
                    level="warn",
                )
                if (route_ok and projected_step >= min_progress
                        and not immediate_return):
                    result, bounded_step = self._limit_projected_step(
                        current_pos,
                        snapped_pos,
                        max_projected_step,
                    )
                    self._log_event(
                        f"snap_step_bounded_{mode}",
                        (
                            f"snap step bounded({mode}): "
                            f"moved={projected_step:.3f}->{bounded_step:.3f}, "
                            f"step={step_len:.3f}"
                        ),
                        cooldown=0.5,
                        level="warn",
                    )
                    self._reset_snap_slide_count()
                    return self._preserve_follower_center_z(result, current_pos), False
        elif snap_ok:
            self._log_event(
                f"snap_large_{mode}",
                (
                    f"snap large({mode}): snap={snap_error:.3f}, "
                    f"step={step_len:.3f}, limit={snap_step_limit:.3f}"
                ),
                cooldown=0.5,
                level="warn",
            )

        if self._current_nav_waypoint is not None:
            waypoint_delta = self._current_nav_waypoint[:2] - current_pos[:2]
            waypoint_dist = float(np.linalg.norm(waypoint_delta))
            waypoint_snap_dist = max(self.my_radius * 0.4, self.waypoint_reach_dist * 0.5)
            if self.stuck_move_eps < waypoint_dist <= waypoint_snap_dist:
                result = self._current_nav_waypoint.copy()
                bounded_dist = waypoint_dist
                if waypoint_dist > max_projected_step:
                    result, bounded_dist = self._limit_projected_step(
                        current_pos,
                        self._current_nav_waypoint,
                        max_projected_step,
                    )
                immediate_return = self._is_immediate_projection_return(
                    previous_origin, current_pos, result
                )
                if immediate_return:
                    self._log_event(
                        f"skip_returning_waypoint_{mode}",
                        "skip waypoint candidate that closes an A-B-A cycle",
                        cooldown=0.5,
                        level="warn",
                    )
                else:
                    self._log_event(
                        f"snap_waypoint_{mode}",
                        (
                            f"snap waypoint({mode}): "
                            f"dist={waypoint_dist:.3f}->{bounded_dist:.3f}, "
                            f"limit={waypoint_snap_dist:.3f}"
                        ),
                        cooldown=0.5,
                        level="warn",
                    )
                    self._reset_snap_slide_count()
                    return self._preserve_follower_center_z(result, current_pos), False

        direction_candidates = []
        desired_norm = float(np.linalg.norm(desired_dir))
        if desired_norm > 1e-6:
            direction_candidates.append(np.asarray(desired_dir, dtype=np.float32) / desired_norm)
        elif step_len > 1e-6:
            direction_candidates.append(desired_step / step_len)

        if self._current_nav_waypoint is not None:
            waypoint_dir = self._current_nav_waypoint[:2] - current_pos[:2]
            waypoint_norm = float(np.linalg.norm(waypoint_dir))
            if waypoint_norm > 1e-6:
                direction_candidates.append(waypoint_dir / waypoint_norm)

        base_dirs = list(direction_candidates)
        for direction in base_dirs:
            for angle_deg in (35.0, -35.0, 70.0, -70.0, 90.0, -90.0):
                angle = math.radians(angle_deg)
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                direction_candidates.append(np.array([
                    direction[0] * cos_a - direction[1] * sin_a,
                    direction[0] * sin_a + direction[1] * cos_a,
                ], dtype=np.float32))

        if not direction_candidates:
            self._reset_snap_slide_count()
            return self._preserve_follower_center_z(current_pos, current_pos), True

        reference_dir = direction_candidates[0]
        probe_dist = min(
            max(step_len * 3.0, self.stuck_move_eps * 4.0, self.my_radius * 0.35),
            max(self.max_navmesh_snap * 1.5, self.my_radius * 0.75),
        )
        best = None
        best_progress = None
        best_score = -1e9
        for direction in direction_candidates:
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue
            direction = direction / norm
            candidate = current_pos.copy()
            candidate[0] += direction[0] * probe_dist
            candidate[1] += direction[1] * probe_dist
            projected = self._project_to_navmesh(candidate)
            if projected is None:
                continue

            displacement = projected[:2] - current_pos[:2]
            moved = float(np.linalg.norm(displacement))
            snap = float(np.linalg.norm(projected[:2] - candidate[:2]))
            if moved < max(self.stuck_move_eps, self.my_radius * 0.1):
                continue
            if snap > max(self.max_navmesh_snap * 2.0, self.my_radius * 0.75):
                continue

            motion_target = projected
            if moved > max_projected_step:
                motion_target, moved = self._limit_projected_step(
                    current_pos,
                    projected,
                    max_projected_step,
                )

            route_progress = self._motion_route_progress(current_pos, motion_target)
            regress_limit = -max(0.02, step_len)
            if route_progress is not None and route_progress < regress_limit:
                continue
            if self._is_immediate_projection_return(
                previous_origin, current_pos, motion_target
            ):
                continue

            displacement = motion_target[:2] - current_pos[:2]
            progress = float(np.dot(displacement / max(moved, 1e-6), reference_dir))
            if progress < -0.15:
                continue
            route_score = 0.0 if route_progress is None else route_progress * 4.0
            score = progress * 2.0 + route_score + moved - snap * 0.5
            if score > best_score:
                best_score = score
                best = motion_target
                best_progress = route_progress

        if best is not None:
            result = current_pos.copy()
            result[0] = best[0]
            result[1] = best[1]
            if self._record_snap_slide(mode, step_len, current_pos, best, best_progress):
                return self._preserve_follower_center_z(current_pos, current_pos), True
            return self._preserve_follower_center_z(result, current_pos), False

        if snap_ok and snap_error <= snap_step_limit:
            projected_step = float(np.linalg.norm(snapped_pos[:2] - current_pos[:2]))
            route_progress = self._motion_route_progress(current_pos, snapped_pos)
            regress_limit = -max(0.01, step_len * 0.5)
            route_ok = route_progress is None or route_progress >= regress_limit
            immediate_return = self._is_immediate_projection_return(
                previous_origin, current_pos, snapped_pos
            )
            if (route_ok and projected_step >= min_progress
                    and not immediate_return):
                if projected_step > max_projected_step:
                    result, _ = self._limit_projected_step(
                        current_pos,
                        snapped_pos,
                        max_projected_step,
                    )
                    self._reset_snap_slide_count()
                    return self._preserve_follower_center_z(result, current_pos), False
                result = desired_pos.copy()
                result[0] = snapped_pos[0]
                result[1] = snapped_pos[1]
                self._reset_snap_slide_count()
                return self._preserve_follower_center_z(result, current_pos), False

            self._log_event(
                f"snap_no_progress_{mode}",
                (
                    f"snap made no route progress({mode}): "
                    f"moved={projected_step:.4f}, step={step_len:.4f}, "
                    f"route_progress={route_progress if route_progress is not None else 0.0:.4f}"
                ),
                cooldown=0.5,
                level="warn",
            )

        if (
            previous_origin is not None
            and snapped_pos is not None
            and self._is_immediate_projection_return(
                previous_origin, current_pos, snapped_pos
            )
        ):
            self._invalidate_oscillating_route(
                mode, previous_origin, current_pos, snapped_pos
            )
        else:
            self._reset_snap_slide_count()
        return self._preserve_follower_center_z(current_pos, current_pos), True

    def _compute_path(
        self,
        start_pos,
        end_pos,
        agent_radius=None,
        max_start_snap=None,
        max_end_snap=None,
    ):
        """NavMesh 最短路径查询。"""
        if self.navmesh is None:
            self._log_event("path_navmesh_none", "compute_path skipped: navmesh is None", cooldown=1.0, level="warn")
            return None
        radius = self.my_radius if agent_radius is None else float(agent_radius)
        start_snap_limit = self.max_navmesh_snap if max_start_snap is None else float(max_start_snap)
        end_snap_limit = self.max_navmesh_snap if max_end_snap is None else float(max_end_snap)
        try:
            start_nav = self._project_to_navmesh(start_pos, agent_radius=radius)
            end_nav = self._project_to_navmesh(end_pos, agent_radius=radius)
            if start_nav is None or end_nav is None:
                self._log_event(
                    "path_project_fail",
                    (
                        f"project fail: start_ok={start_nav is not None}, "
                        f"end_ok={end_nav is not None}"
                    ),
                    cooldown=0.8,
                    level="warn",
                )
                return None

            start_snap = np.linalg.norm(start_nav[:2] - start_pos[:2])
            end_snap = np.linalg.norm(end_nav[:2] - end_pos[:2])
            if start_snap > start_snap_limit or end_snap > end_snap_limit:
                self._log_event(
                    "path_snap_too_far",
                    (
                        f"snap too far: start={start_snap:.3f}, "
                        f"end={end_snap:.3f}, "
                        f"max_start={start_snap_limit:.3f}, "
                        f"max_end={end_snap_limit:.3f}"
                    ),
                    cooldown=0.8,
                    level="warn",
                )
                return None

            start_c = carb.Float3(start_nav[0], start_nav[1], start_nav[2])
            end_c = carb.Float3(end_nav[0], end_nav[1], end_nav[2])
            path = self.navmesh.query_shortest_path(
                start_c, end_c,
                agent_radius=radius,
                agent_height=0.5,
            )
            if path is None:
                self._log_event("path_query_none", "query_shortest_path returned None", cooldown=0.8, level="warn")
                return None
            points = path.get_points()
            result = [np.array([p.x, p.y, p.z]) for p in points]
            if len(result) < 2:
                self._log_event("path_too_short", f"query path points={len(result)}", cooldown=0.8, level="warn")
                return None
            if not self._path_segments_stay_on_navmesh(
                result,
                agent_radius=radius,
            ):
                return None
            return result
        except Exception as e:
            self._log_event("path_exception", f"Path error: {e}", cooldown=0.8, level="warn")
            return None

    def _path_length_2d(self, path_points):
        if not path_points or len(path_points) < 2:
            return 0.0
        length = 0.0
        for idx in range(1, len(path_points)):
            length += float(np.linalg.norm(
                path_points[idx][:2] - path_points[idx - 1][:2]
            ))
        return length

    def _compute_follow_path(self, follower_pos, target_pos):
        """计算正常跟随路径：目标是 human 本体，停止半径只在控制层生效。"""
        straight_distance = float(np.linalg.norm(target_pos[:2] - follower_pos[:2]))
        target_path = self._compute_path(
            follower_pos,
            target_pos,
            agent_radius=self.my_radius,
            max_end_snap=self.target_navmesh_snap,
        )
        if not target_path:
            self._last_nav_path_distance = None
            self._last_route_path_distance = None
            self._last_path_detour_active = False
            self._last_effective_path_stop_distance = float(self.path_stop_distance)
            return []
        target_path_length = self._path_length_2d(target_path)
        effective_stop_distance = self._choose_path_stop_distance(
            target_path_length,
            straight_distance,
        )
        self._last_nav_path_distance = target_path_length
        self._last_path_detour_active = self._is_detour_path(
            target_path_length,
            straight_distance,
        )
        self._last_effective_path_stop_distance = effective_stop_distance
        if self._last_path_detour_active:
            self._log_event(
                "path_detour_active",
                (
                    f"path detour: path={target_path_length:.3f}, "
                    f"straight={straight_distance:.3f}, "
                    f"stop={effective_stop_distance:.3f}"
                ),
                cooldown=0.8,
            )
        if not target_path or len(target_path) < 2:
            return []

        route_goal = target_path[-1]
        if np.linalg.norm(route_goal[:2] - target_path[0][:2]) <= 1e-6:
            return self._densify_path(target_path)

        planning_radius = max(float(self.planning_radius), float(self.my_radius))
        if planning_radius <= self.my_radius + 1e-6:
            return self._densify_path(target_path)

        clearance_start_snap = max(
            float(self.max_navmesh_snap),
            min(float(planning_radius) * 0.75, 0.5),
        )
        clearance_path = self._compute_path(
            follower_pos,
            route_goal,
            agent_radius=planning_radius,
            max_start_snap=clearance_start_snap,
        )
        if clearance_path and len(clearance_path) >= 2:
            return self._densify_path(clearance_path)

        needs_clearance = (
            self._last_path_detour_active
            or self._is_constrained_shelf_route(target_path_length, straight_distance)
            or self._path_has_sharp_turn(target_path)
        )
        self._log_event(
            "clearance_path_unavailable",
            (
                f"clearance path unavailable: radius={planning_radius:.3f}, "
                f"goal=({route_goal[0]:.2f},{route_goal[1]:.2f}); "
                f"needs_clearance={needs_clearance}"
            ),
            cooldown=0.8,
            level="warn",
        )
        if needs_clearance:
            self._last_nav_path_distance = target_path_length
            self._last_path_detour_active = True
            return []
        return self._densify_path(target_path)

    def _get_next_waypoint(self, follower_pos, target_pos, dist_to_human):
        """Compute this frame's NavMesh route and return a Habitat-style carrot."""
        self._path_replanned_this_frame = False
        self._path_query_failed = False

        recovery_wp = self._active_recovery_waypoint(follower_pos)
        if recovery_wp is not None:
            return recovery_wp

        new_path = self._compute_follow_path(follower_pos, target_pos) or []
        if new_path and len(new_path) >= 2:
            self._consecutive_path_failures = 0
            self.cached_path = new_path
            self.cached_waypoint_idx = self._initial_waypoint_idx_for_path(
                follower_pos,
                new_path,
            )
            self.last_target_pos = target_pos.copy()
            self._path_replanned_this_frame = True
            self._path_query_failed = False
            self._log_event(
                "replan_accept",
                (
                    f"realtime path accepted idx={self.cached_waypoint_idx}, "
                    f"points={len(new_path)}"
                ),
                cooldown=0.5,
            )
        elif self.cached_path and len(self.cached_path) >= 2:
            self._consecutive_path_failures += 1
            self._path_query_failed = True
            self._log_event(
                "replan_keep_cache",
                "realtime path failed; keeping previous NavMesh path",
                cooldown=0.8,
                level="warn",
            )
        else:
            self._consecutive_path_failures += 1
            self._path_query_failed = True
            self.cached_path = []
            self.last_target_pos = target_pos.copy()

        if not self.cached_path or len(self.cached_path) < 2:
            return None

        # 沿路径点序列前进：到达当前路径点后推进到下一个
        idx = self.cached_waypoint_idx
        if idx < len(self.cached_path):
            wp = self.cached_path[idx]
            dist_to_wp = np.linalg.norm(follower_pos[:2] - wp[:2])
            reach_threshold = self._waypoint_reach_threshold(
                [np.asarray(p, dtype=np.float32) for p in self.cached_path],
                idx,
            )
            if dist_to_wp < reach_threshold:
                self.cached_waypoint_idx = min(
                    idx + 1, len(self.cached_path) - 1
                )
                self._log_event(
                    "waypoint_advance",
                    (
                        f"advance waypoint {idx}->{self.cached_waypoint_idx}, "
                        f"dist_to_wp={dist_to_wp:.3f}, "
                        f"reach={reach_threshold:.3f}"
                    ),
                    cooldown=0.2,
                )
            elif reach_threshold < self.waypoint_reach_dist:
                self._log_event(
                    "waypoint_hold_corner",
                    (
                        f"hold sharp waypoint {idx}: "
                        f"dist_to_wp={dist_to_wp:.3f}, "
                        f"reach={reach_threshold:.3f}"
                    ),
                    cooldown=0.8,
                )

        idx = min(self.cached_waypoint_idx, len(self.cached_path) - 1)
        progress_segment_idx = max(
            0,
            min(idx - 1, len(self.cached_path) - 2),
        )
        remaining_route = self._remaining_path_length_from_pos(
            follower_pos,
            self.cached_path,
            start_segment_idx=progress_segment_idx,
        )
        if remaining_route is not None:
            self._last_route_path_distance = remaining_route
            if remaining_route > self.waypoint_reach_dist:
                self._last_nav_path_distance = max(
                    float(dist_to_human),
                    float(remaining_route),
                )
            self._last_path_detour_active = self._is_detour_path(
                self._last_nav_path_distance,
                dist_to_human,
            )
            self._get_tracking_distance(dist_to_human)

        lookahead_source = (
            self._last_tracking_distance
            if self._last_tracking_distance is not None else dist_to_human
        )
        lookahead_dist = self._habitat_lookahead_distance(lookahead_source)
        points = [np.asarray(p, dtype=np.float32) for p in self.cached_path]
        stop_at_idx = idx if self._is_sharp_path_turn(points, idx) else None
        carrot = self._select_carrot_point(
            follower_pos,
            self.cached_path,
            lookahead_dist,
            start_segment_idx=progress_segment_idx,
            stop_at_waypoint_idx=stop_at_idx,
        )
        if carrot is not None:
            return carrot
        return self.cached_path[idx]

    def _get_retreat_waypoint(self, follower_pos, target_pos, to_human_2d, dist_to_human):
        """Find a NavMesh-constrained waypoint that increases target separation."""
        if dist_to_human <= 1e-6:
            return None
        retreat_dir = -to_human_2d / dist_to_human
        desired_distance = self.safe_dis_min + 0.1
        retreat_goal = target_pos.copy()
        retreat_goal[0] += retreat_dir[0] * desired_distance
        retreat_goal[1] += retreat_dir[1] * desired_distance

        retreat_snap_limit = max(
            float(self.target_navmesh_snap),
            float(self.max_navmesh_snap),
        )
        path = self._compute_path(
            follower_pos,
            retreat_goal,
            max_end_snap=retreat_snap_limit,
        )
        if path and len(path) >= 2:
            end_sep = float(np.linalg.norm(path[-1][:2] - target_pos[:2]))
            if end_sep <= dist_to_human + 0.05:
                self._log_event(
                    "retreat_no_separation_gain",
                    (
                        f"retreat path rejected: end_sep={end_sep:.3f}, "
                        f"dist={dist_to_human:.3f}"
                    ),
                    cooldown=0.8,
                    level="warn",
                )
            else:
                lookahead_dist = float(np.clip(
                    max(0.35, desired_distance - dist_to_human), 0.35, 1.2
                ))
                waypoint = self._select_carrot_point(follower_pos, path, lookahead_dist)
                if waypoint is None:
                    waypoint = path[1]
                self._log_event(
                    "retreat_navmesh",
                    (
                        f"navmesh retreat via waypoint ({waypoint[0]:.2f},{waypoint[1]:.2f}) "
                        f"target=({target_pos[0]:.2f},{target_pos[1]:.2f})"
                    ),
                    cooldown=0.5,
                )
                return waypoint

        fallback_distance = max(self.safe_dis_min + 0.1 - dist_to_human, 0.35)
        fallback_goal = follower_pos.copy()
        fallback_goal[0] += retreat_dir[0] * fallback_distance
        fallback_goal[1] += retreat_dir[1] * fallback_distance
        projected_goal = self._project_to_navmesh(fallback_goal)
        if projected_goal is not None:
            snap = float(np.linalg.norm(projected_goal[:2] - fallback_goal[:2]))
            projected_sep = float(np.linalg.norm(projected_goal[:2] - target_pos[:2]))
            if snap <= retreat_snap_limit and projected_sep > dist_to_human + 0.05:
                fallback_path = self._compute_path(
                    follower_pos,
                    projected_goal,
                    max_end_snap=self.max_navmesh_snap,
                )
                if fallback_path and len(fallback_path) >= 2:
                    lookahead_dist = float(np.clip(
                        fallback_distance, 0.35, 1.2
                    ))
                    waypoint = self._select_carrot_point(
                        follower_pos,
                        fallback_path,
                        lookahead_dist,
                    )
                    if waypoint is None:
                        waypoint = fallback_path[1]
                    self._log_event(
                        "retreat_projected_fallback",
                        (
                            f"projected retreat fallback snap={snap:.3f}, "
                            f"wp=({waypoint[0]:.2f},{waypoint[1]:.2f})"
                        ),
                        cooldown=0.5,
                    )
                    return waypoint

        best_path = None
        best_snap = None
        best_score = None
        for angle_deg in (30.0, -30.0, 60.0, -60.0, 90.0, -90.0):
            angle_rad = math.radians(angle_deg)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            rotated_dir = np.array([
                retreat_dir[0] * cos_a - retreat_dir[1] * sin_a,
                retreat_dir[0] * sin_a + retreat_dir[1] * cos_a,
            ], dtype=np.float32)
            for distance in (
                fallback_distance,
                float(np.clip(fallback_distance * 1.5, 0.5, 1.2)),
            ):
                candidate = follower_pos.copy()
                candidate[0] += rotated_dir[0] * distance
                candidate[1] += rotated_dir[1] * distance
                projected = self._project_to_navmesh(candidate)
                if projected is None:
                    continue
                snap = float(np.linalg.norm(projected[:2] - candidate[:2]))
                projected_sep = float(np.linalg.norm(projected[:2] - target_pos[:2]))
                if snap > retreat_snap_limit or projected_sep <= dist_to_human + 0.05:
                    continue
                fan_path = self._compute_path(
                    follower_pos,
                    projected,
                    max_end_snap=self.max_navmesh_snap,
                )
                if not fan_path or len(fan_path) < 2:
                    continue
                alignment = float(np.dot(rotated_dir, retreat_dir))
                score = (projected_sep - dist_to_human) + 0.2 * alignment - 0.25 * snap
                if best_score is None or score > best_score:
                    best_score = score
                    best_path = fan_path
                    best_snap = snap

        if best_path is not None:
            lookahead_dist = float(np.clip(fallback_distance, 0.35, 1.2))
            waypoint = self._select_carrot_point(
                follower_pos,
                best_path,
                lookahead_dist,
            )
            if waypoint is None:
                waypoint = best_path[1]
            self._log_event(
                "retreat_fan_fallback",
                (
                    f"fan retreat fallback snap={best_snap:.3f}, "
                    f"wp=({waypoint[0]:.2f},{waypoint[1]:.2f})"
                ),
                cooldown=0.5,
            )
            return waypoint

        self._log_event(
            "retreat_navmesh_unavailable",
            "navmesh retreat waypoint unavailable",
            cooldown=0.8,
            level="warn",
        )
        return None

    def _publish_position(self, follower_pos, delta_time):
        """每帧发布 cylinder 位置到 GlobalCharacterPositionManager。

        如果 omni.anim.people 全局动态避障被打开，其他角色会看到这个
        robot/follower 障碍物；默认配置关闭该全局开关，robot 侧局部避障
        不依赖其他角色也发布到这个 manager。
        """
        if not self.dynamic_avoidance_enabled or self.character_manager is None:
            return

        num_frames = 10
        pos_carb = carb.Float3(
            follower_pos[0], follower_pos[1], follower_pos[2]
        )

        if len(self.positions_over_time) < num_frames:
            self.positions_over_time.append(pos_carb)
            self.delta_time_list.append(delta_time)
        else:
            self.positions_over_time.pop(0)
            self.positions_over_time.append(pos_carb)
            self.delta_time_list.pop(0)
            self.delta_time_list.append(delta_time)

        # 估算速度（每秒）
        dt_sum = sum(self.delta_time_list)
        if dt_sum > 1e-6 and len(self.positions_over_time) >= 2:
            oldest = self.positions_over_time[0]
            newest = self.positions_over_time[-1]
            vx = (newest.x - oldest.x) / dt_sum
            vy = (newest.y - oldest.y) / dt_sum
            vz = (newest.z - oldest.z) / dt_sum
            future_pos = carb.Float3(
                newest.x + vx, newest.y + vy, newest.z + vz
            )
        else:
            future_pos = pos_carb

        try:
            self.character_manager.set_character_current_pos(
                self.follower_prim_path, pos_carb
            )
            self.character_manager.set_character_future_pos(
                self.follower_prim_path, future_pos
            )
            self.character_manager.set_character_radius(
                self.follower_prim_path, self.dynamic_avoidance_radius
            )
        except Exception as e:
            carb.log_warn(f"[FollowerBehavior v2] Publish pos error: {e}")

    def _dynamic_obstacle_combined_radius(self, obs_radius):
        robot_radius = max(
            float(getattr(self, "my_radius", 0.0)),
            float(getattr(self, "dynamic_avoidance_radius", 0.0)),
        )
        return robot_radius + max(float(obs_radius), 0.0)

    def _orca_combined_radius(self, obs_radius):
        return (
            self._dynamic_obstacle_combined_radius(obs_radius)
            + float(self.orca_clearance_margin)
        )

    def _compute_avoidance_waypoint(
        self, follower_pos, follower_yaw, original_waypoint
    ):
        """检测前方非目标角色障碍物，计算 robot 主动绕行路径点。

        参考 NavigationManager.update_path() 的旋转避障逻辑：
        - 计算运动方向的左/右旋转点
        - 选择障碍物对侧的点
        - 验证 NavMesh 可达性
        角色位置优先读取 GlobalCharacterPositionManager；当 people 动态避障
        关闭导致 manager 没有人体位置时，直接扫描 /World/Characters。
        返回绕行路径点，或 None（无需绕行）。
        """
        if not self.dynamic_avoidance_enabled:
            return None

        # 运动方向
        move_2d = np.array([
            original_waypoint[0] - follower_pos[0],
            original_waypoint[1] - follower_pos[1],
        ])
        move_len = np.linalg.norm(move_2d)
        if move_len < 1e-6:
            return None

        move_dir = move_2d / move_len

        # 运动方向的右向量
        move_right = np.array([move_dir[1], -move_dir[0]])

        best_obs = None
        best_score = -float("inf")

        scanned_chars = 0
        candidate_chars = 0
        for _char_path, obs_np, obs_radius, obs_vel in self._iter_dynamic_obstacles():
            scanned_chars += 1

            # 障碍物相对 follower 的向量
            to_obs = obs_np - follower_pos[:2]
            dist = float(np.linalg.norm(to_obs))

            # 只关心前方 avoidance_radius 内的障碍物
            if dist > self.avoidance_radius:
                continue
            # 只关心运动方向前方的障碍物（点积 > 0）
            along_dist = float(np.dot(to_obs, move_dir))
            if along_dist <= 0:
                continue

            radius_sum = self._dynamic_obstacle_combined_radius(obs_radius)
            risk = self._dynamic_obstacle_collision_risk(
                follower_pos[:2],
                move_dir,
                move_right,
                obs_np,
                obs_vel,
                radius_sum,
                along_dist,
            )
            if risk is not None:
                candidate_chars += 1
                if risk["score"] > best_score:
                    best_score = risk["score"]
                    best_obs = (obs_np, obs_radius, obs_vel, risk)

        if candidate_chars > 0:
            closest_dist = float(best_obs[3]["current_dist"]) if best_obs else -1.0
            self._log_event(
                "avoidance_candidates",
                (
                    f"avoidance candidates={candidate_chars}/{scanned_chars}, "
                    f"closest={closest_dist:.3f}, "
                    f"ttc={best_obs[3]['closest_t']:.2f}"
                ),
                cooldown=0.4,
            )

        if best_obs is None:
            return None

        obs_np, obs_radius, obs_vel, risk = best_obs

        # 绕行角度：近距离/高风险时更明显，远距离预测风险只轻微偏航。
        radius_sum = self._dynamic_obstacle_combined_radius(obs_radius)
        urgency = 1.0 - float(risk["closest_t"]) / max(
            float(self.avoidance_time_horizon), 1e-6
        )
        base_angle_deg = min(
            (obs_radius / max(self.my_radius, 0.1)) * 30.0, 45.0
        )
        angle_deg = float(
            np.clip(base_angle_deg * (0.45 + 0.55 * urgency), 12.0, 45.0)
        )
        angle_rad = math.radians(angle_deg)

        # 绕行点 = 当前位置 + 旋转后的运动向量
        # 方向选择基于预测位置/速度：障碍未来在右侧或正向右移 → robot 从左侧绕。
        rot_angle = self._select_dynamic_avoidance_rotation(
            follower_pos[:2],
            move_dir,
            move_right,
            obs_np,
            obs_vel,
            risk,
            angle_rad,
        )
        cos_a, sin_a = math.cos(rot_angle), math.sin(rot_angle)
        rotated_dir = np.array([
            move_dir[0] * cos_a - move_dir[1] * sin_a,
            move_dir[0] * sin_a + move_dir[1] * cos_a,
        ])

        # 避障点只看短程，不把远处预测风险放大成大幅横移。
        avoid_distance = float(
            np.clip(
                risk["along_dist"],
                max(0.8, radius_sum * 1.1),
                max(1.0, min(move_len, self.forward_velocity * 0.9 + radius_sum * 0.5)),
            )
        )
        avoidance_pt_2d = follower_pos[:2] + rotated_dir * avoid_distance
        avoidance_pt = np.array([
            avoidance_pt_2d[0], avoidance_pt_2d[1], follower_pos[2]
        ])

        # NavMesh 可达性验证
        if self.navmesh is not None:
            try:
                test = self.navmesh.query_closest_point(
                    carb.Float3(
                        avoidance_pt[0], avoidance_pt[1], avoidance_pt[2]
                    ),
                    agent_radius=self.my_radius,
                )
                if test is None:
                    return None
            except Exception:
                pass

        return avoidance_pt

    def _dynamic_obstacle_collision_risk(
        self,
        follower_pos_2d,
        move_dir,
        move_right,
        obs_np,
        obs_vel,
        radius_sum,
        along_dist,
    ):
        current_delta = obs_np - follower_pos_2d
        current_lateral = abs(float(np.dot(current_delta, move_right)))
        current_dist = float(np.linalg.norm(current_delta))
        robot_speed = max(0.35, float(self.forward_velocity) * 0.75)
        robot_vel = move_dir * robot_speed
        rel_pos = obs_np - follower_pos_2d
        rel_vel = obs_vel - robot_vel
        rel_speed_sq = float(np.dot(rel_vel, rel_vel))
        horizon = max(float(self.avoidance_time_horizon), 0.1)
        if rel_speed_sq > 1e-6:
            closest_t = float(
                np.clip(-np.dot(rel_pos, rel_vel) / rel_speed_sq, 0.0, horizon)
            )
        else:
            closest_t = 0.0

        robot_at_t = follower_pos_2d + robot_vel * closest_t
        obs_at_t = obs_np + obs_vel * closest_t
        future_delta = obs_at_t - robot_at_t
        future_dist = float(np.linalg.norm(future_delta))
        future_lateral = abs(float(np.dot(future_delta, move_right)))
        clearance = float(radius_sum + self.avoidance_clearance_margin)
        close_react_dist = max(self._too_close_distance(), radius_sum * 1.2)

        current_close = (
            along_dist <= close_react_dist
            and current_lateral <= radius_sum * 1.05
        )
        predicted_collision = (
            future_dist <= clearance
            and future_lateral <= clearance
        )
        if not current_close and not predicted_collision:
            return None

        score = max(0.0, clearance - future_dist)
        score += 0.35 * (1.0 - closest_t / horizon)
        if current_close:
            score += 0.5

        return {
            "score": float(score),
            "closest_t": float(closest_t),
            "current_dist": current_dist,
            "current_lateral": current_lateral,
            "future_dist": future_dist,
            "future_lateral": future_lateral,
            "along_dist": float(along_dist),
        }

    def _select_dynamic_avoidance_rotation(
        self,
        follower_pos_2d,
        move_dir,
        move_right,
        obs_np,
        obs_vel,
        risk,
        angle_rad,
    ):
        robot_speed = max(0.35, float(self.forward_velocity) * 0.75)
        eval_t = float(
            np.clip(
                max(risk["closest_t"], 0.45),
                0.0,
                max(float(self.avoidance_time_horizon), 0.1),
            )
        )
        obs_eval = obs_np + obs_vel * eval_t

        def rotated(direction, angle):
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            return np.array([
                direction[0] * cos_a - direction[1] * sin_a,
                direction[0] * sin_a + direction[1] * cos_a,
            ], dtype=np.float32)

        left_dir = rotated(move_dir, angle_rad)
        right_dir = rotated(move_dir, -angle_rad)
        left_pos = follower_pos_2d + left_dir * robot_speed * eval_t
        right_pos = follower_pos_2d + right_dir * robot_speed * eval_t
        left_clearance = float(np.linalg.norm(obs_eval - left_pos))
        right_clearance = float(np.linalg.norm(obs_eval - right_pos))
        if abs(left_clearance - right_clearance) > 0.05:
            return angle_rad if left_clearance > right_clearance else -angle_rad

        future_side = float(np.dot(obs_eval - follower_pos_2d, move_right))
        if abs(future_side) > 0.05:
            return angle_rad if future_side > 0 else -angle_rad

        velocity_side = float(np.dot(obs_vel, move_right))
        return angle_rad if velocity_side >= 0.0 else -angle_rad

    def _iter_dynamic_obstacles(self):
        """Yield non-target character obstacles for robot-side dynamic avoidance."""
        seen = set()
        for item in self._iter_manager_dynamic_obstacles():
            char_path, _obs_np, _obs_radius, _obs_vel = item
            seen.add(char_path)
            yield item

        for item in self._iter_stage_character_obstacles():
            char_path, _obs_np, _obs_radius, _obs_vel = item
            if char_path in seen:
                continue
            yield item

    def _iter_manager_dynamic_obstacles(self):
        if self.character_manager is None:
            return

        try:
            all_chars = list(self.character_manager.get_all_managed_characters())
        except Exception:
            return

        for char_path in all_chars:
            char_path = str(char_path)
            if not self._should_consider_dynamic_obstacle(char_path):
                continue
            try:
                obs_pos = self.character_manager.get_character_current_pos(
                    char_path
                )
                obs_radius = self.character_manager.get_character_radius(
                    char_path
                )
            except Exception:
                continue
            try:
                future_pos = self.character_manager.get_character_future_pos(
                    char_path
                )
            except Exception:
                future_pos = None
            obs_np = np.array([obs_pos.x, obs_pos.y], dtype=np.float32)
            if future_pos is not None:
                future_np = np.array([future_pos.x, future_pos.y], dtype=np.float32)
            else:
                future_np = None
            obs_vel = self._estimate_dynamic_obstacle_velocity(
                char_path,
                obs_np,
                future_np=future_np,
            )
            try:
                obs_radius = float(obs_radius)
            except Exception:
                continue
            yield (
                char_path,
                obs_np,
                obs_radius,
                obs_vel,
            )

    def _iter_stage_character_obstacles(self):
        stage = getattr(self, "stage", None)
        if stage is None:
            return

        try:
            characters_root = stage.GetPrimAtPath("/World/Characters")
        except Exception:
            return
        if not characters_root or not characters_root.IsValid():
            return

        def visit(prim):
            try:
                if not prim or not prim.IsValid():
                    return
                prim_path = str(prim.GetPath())
                if "/Biped_Setup" in prim_path:
                    return
                if (
                    prim.GetTypeName() == "SkelRoot"
                    and self._should_consider_dynamic_obstacle(prim_path)
                ):
                    pos, _ = self._get_prim_pose(prim)
                    if pos is not None:
                        obs_np = np.asarray(pos[:2], dtype=np.float32)
                        yield (
                            prim_path,
                            obs_np,
                            float(self.character_obstacle_radius),
                            self._estimate_dynamic_obstacle_velocity(
                                prim_path,
                                obs_np,
                            ),
                        )
                for child in prim.GetChildren():
                    yield from visit(child)
            except Exception:
                return

        yield from visit(characters_root)

    def _should_consider_dynamic_obstacle(self, char_path):
        char_path = str(char_path)
        if char_path == str(self.follower_prim_path):
            return False
        target_path = str(getattr(self, "target_skelroot_path", "") or "")
        if target_path and (
            char_path == target_path
            or char_path.startswith(target_path + "/")
            or target_path.startswith(char_path + "/")
        ):
            return False
        return True

    def _estimate_dynamic_obstacle_velocity(
        self, char_path, obs_np, future_np=None
    ):
        if not hasattr(self, "_dynamic_obstacle_history"):
            self._dynamic_obstacle_history = {}

        obs_np = np.asarray(obs_np, dtype=np.float32)
        now = float(getattr(self, "_debug_time", 0.0))
        if future_np is not None:
            velocity = np.asarray(future_np, dtype=np.float32) - obs_np
            velocity = self._clamp_obstacle_velocity(velocity)
            self._dynamic_obstacle_history[char_path] = (
                obs_np.copy(),
                now,
                velocity.copy(),
            )
            return velocity

        previous = self._dynamic_obstacle_history.get(char_path)
        if previous is None:
            velocity = np.zeros(2, dtype=np.float32)
            self._dynamic_obstacle_history[char_path] = (
                obs_np.copy(),
                now,
                velocity.copy(),
            )
            return np.zeros(2, dtype=np.float32)

        prev_pos = previous[0]
        prev_time = previous[1]
        prev_velocity = (
            np.asarray(previous[2], dtype=np.float32)
            if len(previous) >= 3 else np.zeros(2, dtype=np.float32)
        )
        dt = now - float(prev_time)
        if dt <= 1e-5 or dt > 1.0:
            velocity = prev_velocity
        else:
            velocity = self._clamp_obstacle_velocity((obs_np - prev_pos) / dt)
        self._dynamic_obstacle_history[char_path] = (
            obs_np.copy(),
            now,
            velocity.copy(),
        )
        return velocity

    @staticmethod
    def _clamp_obstacle_velocity(velocity):
        velocity = np.asarray(velocity, dtype=np.float32)
        speed = float(np.linalg.norm(velocity))
        max_speed = 2.5
        if speed > max_speed:
            velocity = velocity * (max_speed / speed)
        return velocity

    def _orca_preferred_velocity_margin(
        self,
        current_xy,
        preferred,
        obs_np,
        obs_vel,
        combined_radius,
        dist,
    ):
        current_margin = float(dist) - float(combined_radius)
        near_margin = max(0.08, float(self.orca_clearance_margin) * 1.5)
        if current_margin <= near_margin:
            return True, 0.0, current_margin

        preferred = np.asarray(preferred, dtype=np.float32)
        preferred_speed = float(np.linalg.norm(preferred))
        if preferred_speed < 1e-4:
            return False, float(self.orca_time_horizon), current_margin

        rel_pos = np.asarray(obs_np, dtype=np.float32) - current_xy
        rel_vel = np.asarray(obs_vel, dtype=np.float32) - preferred
        rel_speed_sq = float(np.dot(rel_vel, rel_vel))
        horizon = max(float(self.orca_time_horizon), 0.1)
        if rel_speed_sq <= 1e-8:
            return False, horizon, current_margin

        closest_t = float(
            np.clip(-np.dot(rel_pos, rel_vel) / rel_speed_sq, 0.0, horizon)
        )
        closest = rel_pos + rel_vel * closest_t
        future_margin = float(np.linalg.norm(closest) - combined_radius)
        closing = float(np.dot(rel_pos, rel_vel)) < 0.0
        return closing and future_margin <= 0.0, closest_t, future_margin

    def _orca_progress_direction(self, current_xy, preferred):
        if self._current_nav_waypoint is not None:
            waypoint_dir = (
                np.asarray(self._current_nav_waypoint, dtype=np.float32)[:2]
                - current_xy
            )
            waypoint_norm = float(np.linalg.norm(waypoint_dir))
            if waypoint_norm > 1e-6:
                return waypoint_dir / waypoint_norm

        if self.cached_path and len(self.cached_path) >= 2:
            idx = int(np.clip(
                int(self.cached_waypoint_idx),
                0,
                len(self.cached_path) - 1,
            ))
            waypoint = np.asarray(self.cached_path[idx], dtype=np.float32)
            route_dir = waypoint[:2] - current_xy
            route_norm = float(np.linalg.norm(route_dir))
            if route_norm > 1e-6:
                return route_dir / route_norm

        preferred_speed = float(np.linalg.norm(preferred))
        if preferred_speed > 1e-6:
            return np.asarray(preferred, dtype=np.float32) / preferred_speed
        return None

    def _orca_navmesh_step_score(self, current_pos, candidate_vel, progress_dir):
        if self.navmesh is None or not hasattr(self.navmesh, "query_closest_point"):
            return 0.0

        candidate_vel = np.asarray(candidate_vel, dtype=np.float32)
        speed = float(np.linalg.norm(candidate_vel))
        if speed < 1e-6:
            return 0.0

        current_pos = np.asarray(current_pos, dtype=np.float32)
        step_dt = float(np.clip(float(self.orca_time_horizon) * 0.2, 0.12, 0.25))
        desired = current_pos.copy()
        desired[0] += candidate_vel[0] * step_dt
        desired[1] += candidate_vel[1] * step_dt
        projected = self._project_to_navmesh(desired)
        if projected is None:
            return -2.0

        snap = float(np.linalg.norm(projected[:2] - desired[:2]))
        snap_limit = max(float(self.max_navmesh_snap), float(self.my_radius) * 0.75)
        if snap > snap_limit:
            return -2.0 - snap

        displacement = np.asarray(projected, dtype=np.float32)[:2] - current_pos[:2]
        moved = float(np.linalg.norm(displacement))
        if moved < max(0.003, float(self.stuck_move_eps) * 0.25):
            return -0.4

        score = -1.5 * snap
        route_progress = self._motion_route_progress(current_pos, projected)
        if route_progress is not None:
            score += 2.5 * max(float(route_progress), 0.0)
            score -= 4.0 * max(-float(route_progress), 0.0)

        if progress_dir is not None:
            direction_progress = float(np.dot(displacement / moved, progress_dir))
            score += 0.8 * direction_progress
        return float(score)

    def _orca_free_space_score(self, current_pos, candidate_vel, progress_dir):
        """Bias ORCA toward locally open NavMesh space, not just non-collision."""
        if self.navmesh is None or not hasattr(self.navmesh, "query_closest_point"):
            return 0.0

        candidate_vel = np.asarray(candidate_vel, dtype=np.float32)
        speed = float(np.linalg.norm(candidate_vel))
        if speed < 1e-6:
            return -0.3

        current_pos = np.asarray(current_pos, dtype=np.float32)
        candidate_dir = candidate_vel / speed
        step_dt = float(np.clip(float(self.orca_time_horizon) * 0.35, 0.2, 0.45))
        lookahead = float(np.clip(
            speed * step_dt,
            max(float(self.my_radius) * 0.75, 0.25),
            max(float(self.my_radius) * 2.2, 0.8),
        ))
        center = current_pos.copy()
        center[0] += candidate_dir[0] * lookahead
        center[1] += candidate_dir[1] * lookahead

        projected_center = self._project_to_navmesh(center)
        if projected_center is None:
            return -2.0

        snap_limit = max(float(self.max_navmesh_snap), float(self.my_radius) * 0.8)
        center_snap = float(np.linalg.norm(projected_center[:2] - center[:2]))
        if center_snap > snap_limit:
            return -2.0 - center_snap

        directions = [candidate_dir]
        if progress_dir is not None:
            progress_dir = np.asarray(progress_dir, dtype=np.float32)
            progress_norm = float(np.linalg.norm(progress_dir))
            if progress_norm > 1e-6:
                directions.append(progress_dir / progress_norm)
        for angle_deg in (45.0, -45.0, 90.0, -90.0):
            directions.append(self._rotate_2d(candidate_dir, math.radians(angle_deg)))

        radii = (
            max(float(self.my_radius) * 1.4, 0.42),
            max(float(self.my_radius) * 2.4, 0.75),
        )
        total_weight = 0.0
        openness = 0.0
        seen = set()
        for direction in directions:
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue
            direction = direction / norm
            key = (round(float(direction[0]), 2), round(float(direction[1]), 2))
            if key in seen:
                continue
            seen.add(key)

            alignment = 0.0
            if progress_dir is not None:
                alignment = max(0.0, float(np.dot(direction, progress_dir)))
            direction_weight = 0.75 + 0.35 * alignment
            for radius in radii:
                probe = np.asarray(projected_center, dtype=np.float32).copy()
                probe[0] += direction[0] * radius
                probe[1] += direction[1] * radius
                projected = self._project_to_navmesh(probe)
                total_weight += direction_weight
                if projected is None:
                    openness -= 0.35 * direction_weight
                    continue
                snap = float(np.linalg.norm(projected[:2] - probe[:2]))
                if snap <= snap_limit:
                    openness += direction_weight * (1.0 - 0.6 * snap / snap_limit)
                else:
                    openness -= direction_weight * min(1.0, snap / snap_limit)

        if total_weight <= 1e-6:
            return 0.0
        route_progress = self._motion_route_progress(current_pos, projected_center)
        progress_score = 0.0
        if route_progress is not None:
            progress_score = (
                2.0 * max(float(route_progress), 0.0)
                - 3.0 * max(-float(route_progress), 0.0)
            )
        return float(np.clip(openness / total_weight, -1.0, 1.0) + progress_score)

    def _orca_has_urgent_obstacle(self, obstacles):
        near_margin = max(0.08, float(self.orca_clearance_margin) * 1.5)
        urgent_ttc = max(0.35, float(self.orca_time_horizon) * 0.35)
        for obstacle in obstacles:
            current_margin = float(obstacle["dist"]) - float(obstacle["radius"])
            if current_margin <= near_margin:
                return True
            if (
                obstacle.get("preferred_margin", 1.0) <= 0.0
                and obstacle.get("preferred_ttc", self.orca_time_horizon) <= urgent_ttc
            ):
                return True
        return False

    def _orca_adjust_world_velocity(
        self, current_pos, preferred_world_vel, yaw, allow_lateral=True
    ):
        """One-sided ORCA/velocity-obstacle projection for robot-only avoidance."""
        preferred = np.asarray(preferred_world_vel, dtype=np.float32)
        if not self.dynamic_avoidance_enabled:
            return preferred

        obstacles = []
        current_xy = np.asarray(current_pos[:2], dtype=np.float32)
        preferred_speed = float(np.linalg.norm(preferred))
        neighbor_dist = max(
            float(self.orca_neighbor_distance),
            preferred_speed * float(self.orca_time_horizon),
        )
        for char_path, obs_np, obs_radius, obs_vel in self._iter_dynamic_obstacles():
            offset = np.asarray(obs_np, dtype=np.float32) - current_xy
            dist = float(np.linalg.norm(offset))
            combined_radius = self._orca_combined_radius(obs_radius)
            if dist > neighbor_dist + combined_radius:
                continue
            relevant, preferred_ttc, preferred_margin = (
                self._orca_preferred_velocity_margin(
                    current_xy,
                    preferred,
                    obs_np,
                    obs_vel,
                    combined_radius,
                    dist,
                )
            )
            if not relevant:
                continue
            obstacles.append(
                {
                    "path": char_path,
                    "pos": np.asarray(obs_np, dtype=np.float32),
                    "vel": np.asarray(obs_vel, dtype=np.float32),
                    "radius": combined_radius,
                    "dist": dist,
                    "preferred_ttc": preferred_ttc,
                    "preferred_margin": preferred_margin,
                }
            )

        if not obstacles:
            return preferred

        candidates = self._orca_candidate_velocities(
            current_xy,
            preferred,
            yaw,
            allow_lateral,
            obstacles,
        )
        best_vel = preferred
        best_score = -float("inf")
        best_margin = -float("inf")
        best_ttc = float(self.orca_time_horizon)
        safe_found = False
        preferred_dir = preferred / preferred_speed if preferred_speed > 1e-6 else None
        progress_dir = self._orca_progress_direction(current_xy, preferred)
        urgent_obstacle = self._orca_has_urgent_obstacle(obstacles)

        for candidate in candidates:
            candidate = self._clip_world_velocity_to_motion_limits(
                candidate,
                yaw,
                allow_lateral,
            )
            margin, ttc, closing_violation = self._orca_velocity_margin(
                current_xy,
                candidate,
                obstacles,
            )
            safe = margin >= 0.0 and not closing_violation
            candidate_speed = float(np.linalg.norm(candidate))
            diff_cost = float(np.linalg.norm(candidate - preferred))
            progress = 0.0
            route_progress = 0.0
            side_bias = 0.0
            if preferred_dir is not None:
                progress = float(np.dot(candidate, preferred_dir))
                move_right = np.array(
                    [preferred_dir[1], -preferred_dir[0]], dtype=np.float32
                )
                for obstacle in obstacles:
                    obs_side_vel = float(np.dot(obstacle["vel"], move_right))
                    if abs(obs_side_vel) < 0.05:
                        continue
                    candidate_side = float(np.dot(candidate, move_right))
                    side_bias += max(
                        0.0,
                        -candidate_side * math.copysign(1.0, obs_side_vel),
                    )
            if progress_dir is not None:
                route_progress = float(np.dot(candidate, progress_dir))
            else:
                route_progress = progress
            forward_progress = max(route_progress, progress)
            reverse_progress = max(0.0, -route_progress)
            reverse_penalty = (2.0 if urgent_obstacle else 6.0) * reverse_progress
            stop_penalty = (
                0.8
                if preferred_speed > 0.1 and candidate_speed < 0.05
                else 0.0
            )
            navmesh_score = self._orca_navmesh_step_score(
                current_pos,
                candidate,
                progress_dir,
            )
            free_space_score = self._orca_free_space_score(
                current_pos,
                candidate,
                progress_dir,
            )

            if safe:
                score = (
                    100.0
                    - 1.15 * diff_cost
                    + 2.0 * max(0.0, forward_progress)
                    - reverse_penalty
                    - stop_penalty
                    + 0.65 * candidate_speed
                    + 0.7 * side_bias
                    + 0.35 * min(margin, 1.0)
                    + navmesh_score
                    + 1.6 * free_space_score
                )
            else:
                score = (
                    margin
                    - 0.8 * diff_cost
                    - (2.0 if closing_violation else 0.0)
                    + 0.7 * max(0.0, forward_progress)
                    - reverse_penalty
                    - stop_penalty
                    + 0.2 * side_bias
                    + 0.5 * navmesh_score
                    + 1.1 * free_space_score
                )

            if score > best_score:
                best_score = score
                best_vel = candidate
                best_margin = margin
                best_ttc = ttc
                safe_found = safe_found or safe

        if safe_found:
            adjusted = best_vel
        else:
            emergency = self._orca_emergency_velocity(
                current_xy,
                yaw,
                allow_lateral,
                obstacles,
                preferred,
            )
            emergency_margin, emergency_ttc, _ = self._orca_velocity_margin(
                current_xy,
                emergency,
                obstacles,
            )
            use_best_unsafe = (
                float(np.linalg.norm(best_vel)) > 0.05
                and best_margin >= -max(0.2, float(self.orca_clearance_margin) * 2.0)
            )
            if use_best_unsafe:
                adjusted = best_vel
            else:
                adjusted = emergency
                best_margin = emergency_margin
                best_ttc = emergency_ttc

        if float(np.linalg.norm(adjusted - preferred)) > 0.05:
            self._log_event(
                "orca_adjust",
                (
                    f"orca adjust pref=({preferred[0]:.2f},{preferred[1]:.2f}) "
                    f"-> ({adjusted[0]:.2f},{adjusted[1]:.2f}), "
                    f"margin={best_margin:.3f}, ttc={best_ttc:.2f}, "
                    f"safe={safe_found}"
                ),
                cooldown=0.3,
                level="warn" if best_margin < 0.0 else "info",
            )
        return adjusted

    def _orca_candidate_velocities(
        self,
        current_xy,
        preferred,
        yaw,
        allow_lateral,
        obstacles,
    ):
        max_speed = math.sqrt(
            float(self.forward_velocity) ** 2 + float(self.lateral_velocity) ** 2
        )
        preferred = self._clip_world_velocity_to_motion_limits(
            preferred,
            yaw,
            allow_lateral,
        )
        pref_speed = float(np.linalg.norm(preferred))
        candidates = [preferred, np.zeros(2, dtype=np.float32)]
        if pref_speed > 1e-6:
            base_dir = preferred / pref_speed
        else:
            base_dir, _ = self._forward_left_vectors(yaw)

        speeds = [
            pref_speed,
            min(max_speed, max(pref_speed * 1.15, 0.55)),
            min(max_speed, max(pref_speed, 0.35)),
            min(max_speed, pref_speed * 0.75),
            min(max_speed, pref_speed * 0.5),
            min(max_speed, pref_speed * 0.25),
        ]
        angles = (
            0.0,
            15.0,
            -15.0,
            30.0,
            -30.0,
            45.0,
            -45.0,
            60.0,
            -60.0,
            90.0,
            -90.0,
            120.0,
            -120.0,
            150.0,
            -150.0,
            180.0,
        )
        for speed in speeds:
            if speed <= 1e-6:
                continue
            for angle_deg in angles:
                direction = self._rotate_2d(base_dir, math.radians(angle_deg))
                candidates.append(direction * speed)

        for obstacle in obstacles:
            away = current_xy - obstacle["pos"]
            away_norm = float(np.linalg.norm(away))
            if away_norm < 1e-6:
                continue
            away_dir = away / away_norm
            tangent_left = np.array([-away_dir[1], away_dir[0]], dtype=np.float32)
            tangent_right = -tangent_left
            for direction in (away_dir, tangent_left, tangent_right):
                for speed in (0.35, min(max_speed, max(pref_speed, 0.6))):
                    candidates.append(direction * speed)

        unique = []
        seen = set()
        for candidate in candidates:
            clipped = self._clip_world_velocity_to_motion_limits(
                candidate,
                yaw,
                allow_lateral,
            )
            key = (round(float(clipped[0]), 3), round(float(clipped[1]), 3))
            if key in seen:
                continue
            seen.add(key)
            unique.append(clipped)
        return unique

    def _orca_velocity_margin(self, current_xy, candidate_vel, obstacles):
        horizon = max(float(self.orca_time_horizon), 0.1)
        min_margin = float("inf")
        min_ttc = horizon
        closing_violation = False
        for obstacle in obstacles:
            rel_pos = obstacle["pos"] - current_xy
            rel_vel = obstacle["vel"] - candidate_vel
            rel_speed_sq = float(np.dot(rel_vel, rel_vel))
            if rel_speed_sq > 1e-8:
                ttc = float(
                    np.clip(-np.dot(rel_pos, rel_vel) / rel_speed_sq, 0.0, horizon)
                )
            else:
                ttc = 0.0
            closest = rel_pos + rel_vel * ttc
            margin = float(np.linalg.norm(closest) - obstacle["radius"])
            if margin < min_margin:
                min_margin = margin
                min_ttc = ttc

            current_dist = float(np.linalg.norm(rel_pos))
            if current_dist > 1e-6:
                dist_rate = float(np.dot(rel_pos, rel_vel) / current_dist)
                closing_margin = max(
                    0.08,
                    min(0.18, float(self.orca_clearance_margin) * 1.5),
                )
                if (
                    current_dist - obstacle["radius"] < closing_margin
                    and dist_rate < 0.0
                ):
                    closing_violation = True

        if min_margin == float("inf"):
            min_margin = 0.0
        return min_margin, min_ttc, closing_violation

    def _orca_emergency_velocity(
        self,
        current_xy,
        yaw,
        allow_lateral,
        obstacles,
        preferred,
    ):
        nearest = min(obstacles, key=lambda obs: obs["dist"])
        away = current_xy - nearest["pos"]
        away_norm = float(np.linalg.norm(away))
        if away_norm < 1e-6:
            preferred_norm = float(np.linalg.norm(preferred))
            if preferred_norm > 1e-6:
                away = -preferred / preferred_norm
            else:
                forward, _ = self._forward_left_vectors(yaw)
                away = -forward
        else:
            away = away / away_norm
        speed = min(
            max(0.55, float(self.forward_velocity) * 0.55),
            math.sqrt(float(self.forward_velocity) ** 2 + float(self.lateral_velocity) ** 2),
        )
        return self._clip_world_velocity_to_motion_limits(
            away * speed,
            yaw,
            allow_lateral,
        )

    def _clip_world_velocity_to_motion_limits(
        self, world_vel, yaw, allow_lateral=True
    ):
        body_vel = self._world_to_body_velocity(
            np.asarray(world_vel, dtype=np.float32),
            yaw,
        )
        body_vel[0] = float(
            np.clip(body_vel[0], -self.forward_velocity, self.forward_velocity)
        )
        if allow_lateral:
            body_vel[1] = float(
                np.clip(body_vel[1], -self.lateral_velocity, self.lateral_velocity)
            )
        else:
            body_vel[1] = 0.0
        return self._body_to_world_velocity(body_vel, yaw)

    @staticmethod
    def _rotate_2d(vec, angle):
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return np.array(
            [
                vec[0] * cos_a - vec[1] * sin_a,
                vec[0] * sin_a + vec[1] * cos_a,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # 姿态工具
    # ------------------------------------------------------------------

    def _get_prim_pose(self, prim):
        """获取 prim 世界坐标 (position, rpy)。"""
        try:
            xformable = UsdGeom.Xformable(prim)
            world_tf = xformable.ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            pos = world_tf.ExtractTranslation()
            position = np.array([pos[0], pos[1], pos[2]])
            rotation = world_tf.ExtractRotationMatrix()
            rpy = self._rotation_matrix_to_rpy(rotation)
            return position, rpy
        except Exception as e:
            carb.log_error(f"[FollowerBehavior v2] Pose error: {e}")
            return None, None

    @staticmethod
    def _rotation_matrix_to_rpy(rot):
        """Gf.Matrix3d (Row-Major) → RPY (ZYX 顺序)。"""
        r00, r01 = rot[0][0], rot[0][1]
        r10, r11 = rot[1][0], rot[1][1]
        r20, r21, r22 = rot[2][0], rot[2][1], rot[2][2]
        pitch = math.asin(max(-1.0, min(1.0, -r20)))
        if abs(math.cos(pitch)) > 1e-6:
            roll = math.atan2(r21, r22)
            yaw = math.atan2(r01, r00)
        else:
            roll = 0.0
            yaw = math.atan2(-r10, r11)
        return np.array([roll, pitch, yaw])

    # ------------------------------------------------------------------
    # 速度应用：差速驱动（linear 沿前向，angular 绕 Z 轴）
    # ------------------------------------------------------------------

    def _apply_velocity(self, linear_vel, angular_vel, delta_time, yaw):
        """差速驱动：linear 沿前向，angular 绕 Z 轴。带加速度限制。"""
        try:
            current_pos, _ = self._get_prim_pose(self.follower_prim)
            if current_pos is None:
                return

            # 加速度限制：平滑过渡，防止瞬间跳变
            linear_vel = self._clamp_accel(
                linear_vel, self.prev_linear_vel,
                self.max_linear_accel, delta_time,
            )
            angular_vel = self._clamp_accel(
                angular_vel, self.prev_angular_vel,
                self.max_angular_accel, delta_time,
            )
            forward = np.array([math.sin(yaw), -math.cos(yaw)], dtype=np.float32)
            world_cmd_2d = forward * float(linear_vel)
            world_cmd_2d = self._orca_adjust_world_velocity(
                current_pos,
                world_cmd_2d,
                yaw,
                allow_lateral=False,
            )
            linear_vel = float(np.dot(world_cmd_2d, forward))
            self.prev_linear_vel = linear_vel
            self.prev_angular_vel = angular_vel
            self._last_cmd_linear = linear_vel
            self._last_cmd_lateral = 0.0
            self._last_cmd_angular = angular_vel
            self.prev_body_action = np.array(
                [linear_vel, 0.0, angular_vel], dtype=np.float32
            )

            # 位移（沿前向）
            displacement = world_cmd_2d * delta_time

            new_pos = current_pos.copy()
            new_pos[0] += displacement[0]
            new_pos[1] += displacement[1]

            # 强制约束到 NavMesh，禁止越界穿墙。
            new_pos, snap_rejected = self._project_motion_to_navmesh(
                current_pos,
                new_pos,
                displacement,
                "diff",
            )
            new_pos = self._preserve_follower_center_z(new_pos, current_pos)
            if snap_rejected:
                self._log_event(
                    "snap_rejected_diff",
                    (
                        f"snap rejected(diff): "
                        f"new=({new_pos[0]:.2f},{new_pos[1]:.2f})"
                    ),
                    cooldown=0.6,
                    level="warn",
                )
                new_pos = current_pos.copy()
                new_pos = self._preserve_follower_center_z(new_pos, current_pos)
                self.prev_linear_vel = 0.0
                self._last_cmd_linear = 0.0
                self._last_cmd_lateral = 0.0
                self.prev_body_action[:2] = 0.0
                self._publish_diagnostics(snap_rejected=True)

            # 旋转
            yaw_change = angular_vel * delta_time

            # 世界坐标 → 局部坐标
            parent = self.follower_prim.GetParent()
            if parent and parent.IsValid():
                parent_xf = UsdGeom.Xformable(parent)
                parent_w2l = parent_xf.ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                ).GetInverse()
                local_pos = parent_w2l.Transform(
                    Gf.Vec3d(float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))
                )
            else:
                local_pos = Gf.Vec3d(float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))

            # 更新 xform ops
            xformable = UsdGeom.Xformable(self.follower_prim)
            for op in xformable.GetOrderedXformOps():
                op_type = op.GetOpType()
                if op_type == UsdGeom.XformOp.TypeTranslate:
                    op.Set(local_pos)
                elif op_type == UsdGeom.XformOp.TypeRotateXYZ:
                    cur = op.Get()
                    op.Set(Gf.Vec3d(
                        cur[0], cur[1],
                        cur[2] + math.degrees(yaw_change),
                    ))
                elif op_type == UsdGeom.XformOp.TypeOrient:
                    cur_quat = op.Get()
                    delta_quat = Gf.Rotation(
                        Gf.Vec3d(0, 0, 1), math.degrees(yaw_change)
                    ).GetQuat()
                    op.Set(delta_quat * cur_quat)

        except Exception as e:
            print(f"[FollowerBehavior v2] Apply velocity error: {e}")

    def _apply_velocity_omni(
        self, world_vel_2d, face_target_2d, delta_time, yaw, angular_vel=None
    ):
        """万向轮驱动：先转成 OpenTrackVLA body-frame action，再应用到世界坐标。

        Args:
            world_vel_2d: np.array([vx, vy]) 世界坐标系速度（m/s）
            face_target_2d: np.array([dx, dy]) 期望面向的方向向量
            delta_time: 时间步长
            yaw: 当前 yaw 角度
        """
        try:
            current_pos, _ = self._get_prim_pose(self.follower_prim)
            if current_pos is None:
                return

            raw_body_vel = self._world_to_body_velocity(world_vel_2d, yaw)
            raw_body_vel[0] = np.clip(
                raw_body_vel[0], -self.forward_velocity, self.forward_velocity
            )
            raw_body_vel[1] = np.clip(
                raw_body_vel[1], -self.lateral_velocity, self.lateral_velocity
            )

            # 朝向：默认面向目标；Habitat oracle 路径可直接传入视觉/nav 混合 yaw。
            if angular_vel is None:
                robot_forward_2d, _ = self._forward_left_vectors(yaw)
                angle_to_face = self._get_angle(robot_forward_2d, face_target_2d)
                if angle_to_face > self.turn_thresh:
                    angular_vel = self._compute_turn(
                        face_target_2d, self.turn_velocity, robot_forward_2d
                    )
                else:
                    angular_vel = 0.0

            cmd = self._smooth_body_action(
                [raw_body_vel[0], raw_body_vel[1], angular_vel],
                delta_time,
            )
            world_cmd_2d = self._body_to_world_velocity(cmd[:2], yaw)
            world_cmd_2d = self._orca_adjust_world_velocity(
                current_pos,
                world_cmd_2d,
                yaw,
                allow_lateral=True,
            )
            adjusted_body = self._world_to_body_velocity(world_cmd_2d, yaw)
            cmd[0] = float(adjusted_body[0])
            cmd[1] = float(adjusted_body[1])
            self.prev_body_action[:2] = cmd[:2]
            self.prev_linear_vel = float(cmd[0])
            self.prev_angular_vel = float(cmd[2])
            self._last_cmd_linear = float(cmd[0])
            self._last_cmd_lateral = float(cmd[1])
            self._last_cmd_angular = float(cmd[2])

            # 世界坐标系平移
            new_pos = current_pos.copy()
            new_pos[0] += world_cmd_2d[0] * delta_time
            new_pos[1] += world_cmd_2d[1] * delta_time

            # 强制约束到 NavMesh，禁止越界穿墙。
            new_pos, snap_rejected = self._project_motion_to_navmesh(
                current_pos,
                new_pos,
                world_cmd_2d,
                "omni",
            )
            new_pos = self._preserve_follower_center_z(new_pos, current_pos)
            if snap_rejected:
                self._log_event(
                    "snap_rejected_omni",
                    (
                        f"snap rejected(omni): "
                        f"new=({new_pos[0]:.2f},{new_pos[1]:.2f})"
                    ),
                    cooldown=0.6,
                    level="warn",
                )
                new_pos = current_pos.copy()
                new_pos = self._preserve_follower_center_z(new_pos, current_pos)
                self.prev_linear_vel = 0.0
                self._last_cmd_linear = 0.0
                self._last_cmd_lateral = 0.0
                self.prev_body_action[:2] = 0.0
                self._publish_diagnostics(snap_rejected=True)

            yaw_change = float(cmd[2]) * delta_time

            # 世界坐标 → 局部坐标
            parent = self.follower_prim.GetParent()
            if parent and parent.IsValid():
                parent_xf = UsdGeom.Xformable(parent)
                parent_w2l = parent_xf.ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                ).GetInverse()
                local_pos = parent_w2l.Transform(
                    Gf.Vec3d(float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))
                )
            else:
                local_pos = Gf.Vec3d(float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))

            # 更新 xform ops
            xformable = UsdGeom.Xformable(self.follower_prim)
            for op in xformable.GetOrderedXformOps():
                op_type = op.GetOpType()
                if op_type == UsdGeom.XformOp.TypeTranslate:
                    op.Set(local_pos)
                elif op_type == UsdGeom.XformOp.TypeRotateXYZ:
                    cur = op.Get()
                    op.Set(Gf.Vec3d(
                        cur[0], cur[1],
                        cur[2] + math.degrees(yaw_change),
                    ))
                elif op_type == UsdGeom.XformOp.TypeOrient:
                    cur_quat = op.Get()
                    delta_quat = Gf.Rotation(
                        Gf.Vec3d(0, 0, 1), math.degrees(yaw_change)
                    ).GetQuat()
                    op.Set(delta_quat * cur_quat)

        except Exception as e:
            print(f"[FollowerBehavior v2] Omni velocity error: {e}")

    # ------------------------------------------------------------------
    # 相机跟随：刚性绑定 cylinder（位置 + 朝向一体）
    # ------------------------------------------------------------------

    def _update_camera_transform(self, target_pos, delta_time):
        """相机刚性跟随 cylinder：位置 = cylinder + offset，朝向 = cylinder yaw。

        相机和底盘是一个整体，vx/vy/yaw 完全一致。
        """
        try:
            cyl_xf = UsdGeom.Xformable(self.follower_prim)
            cyl_w = cyl_xf.ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            offset = Gf.Vec3d(*self.camera_offset)
            cam_world_pos = cyl_w.Transform(offset)

            # 直接用 cylinder 的 yaw（不独立插值）
            _, cyl_rpy = self._get_prim_pose(self.follower_prim)
            if cyl_rpy is None:
                return
            cyl_yaw_deg = math.degrees(cyl_rpy[2])

            cam_xf = UsdGeom.Xformable(self.camera_prim)
            for op in cam_xf.GetOrderedXformOps():
                op_type = op.GetOpType()
                if op_type == UsdGeom.XformOp.TypeTranslate:
                    op.Set(cam_world_pos)
                elif op_type == UsdGeom.XformOp.TypeRotateXYZ:
                    op.Set(Gf.Vec3f(
                        self.camera_rotation[0],
                        self.camera_rotation[1],
                        cyl_yaw_deg + self.camera_rotation[2],
                    ))
        except Exception as e:
            carb.log_error(
                f"[FollowerBehavior v2] Camera update error: {e}"
            )
