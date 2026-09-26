#!/usr/bin/env python3
"""Run the datagen tracking collector with a geometry-free camera carrier."""

from __future__ import annotations

import sys

import numpy as np

import tracking_episode_collection_datagen as collector


class CameraCarrier:
    """Pose-compatible replacement for the collector's articulation object."""

    def __init__(self, stage, prim_path: str, position):
        self.prim_path = prim_path
        self.num_dof = 0
        self.dof_names = []
        self._position = np.asarray(position, dtype=np.float32).copy()
        self._orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        prim = stage.DefinePrim(prim_path, "Xform")
        xformable = collector.UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        self._transform_op = xformable.AddTransformOp()
        self._write_transform()

    def _write_transform(self):
        w, x, y, z = map(float, self._orientation)
        rotation = collector.Gf.Rotation(
            collector.Gf.Quatd(w, collector.Gf.Vec3d(x, y, z))
        )
        transform = collector.Gf.Matrix4d(1.0)
        transform.SetRotate(rotation)
        transform.SetTranslateOnly(
            collector.Gf.Vec3d(*map(float, self._position))
        )
        self._transform_op.Set(transform)

    def get_world_pose(self):
        return self._position.copy(), self._orientation.copy()

    def set_world_pose(self, position, orientation):
        self._position = np.asarray(position, dtype=np.float32).copy()
        self._orientation = np.asarray(orientation, dtype=np.float32).copy()
        self._write_transform()

    def set_joint_positions(self, _positions):
        return None

    def set_joint_velocities(self, _velocities):
        return None

    def get_joint_velocities(self):
        return np.zeros(0, dtype=np.float32)

    def set_linear_velocity(self, _velocity):
        return None

    def set_angular_velocity(self, _velocity):
        return None

    def apply_action(self, _action):
        return None


def add_camera_carrier(_world, initial_position=None):
    if initial_position is None:
        initial_position = (0.0, 0.0, 0.0)

    stage = collector.omni.usd.get_context().get_stage()
    stage.DefinePrim(collector.ROBOT_PRIM_PATH, "Xform")
    carrier_path = (
        f"{collector.ROBOT_PRIM_PATH}/{collector._ROBOT_CAMERA_LINK}"
    )
    position = np.array(
        [
            float(initial_position[0]),
            float(initial_position[1]),
            float(initial_position[2]) + float(collector.ARGS.robot_z_height),
        ],
        dtype=np.float32,
    )
    carrier = CameraCarrier(stage, carrier_path, position)
    collector._ROBOT_ARTICULATION_PATH = carrier_path
    print(
        f"[CameraOnly] Carrier created at {carrier_path}; "
        "no robot asset or physical articulation was loaded"
    )
    return carrier


def disable_contact_report(_stage):
    print("[CameraOnly] Physical contact reporting disabled")
    return False


def disable_contact_sensor(_world):
    collector._robot_rigid_view = None


def no_geometry_operation(*_args, **_kwargs):
    return None


def no_physical_occupancy_collision(_position, _occupancy):
    """NavMesh projection is authoritative for the geometry-free carrier."""
    return False


def configure_camera_only_mode():
    collector.ROBOT_DRIVE_MODE = "kinematic"
    collector.ROBOT_JOINT_NAMES = []
    collector.add_robot_and_articulation = add_camera_carrier
    collector.setup_robot_contact_report = disable_contact_report
    collector.setup_robot_contact_sensor = disable_contact_sensor
    collector.hide_robot_visual_geometry = no_geometry_operation
    collector.ensure_robot_above_ground = no_geometry_operation
    collector._detect_in_obstacle = no_physical_occupancy_collision
    print("[CameraOnly] Datagen collector patched before scene initialization")


if __name__ == "__main__":
    configure_camera_only_mode()
    sys.exit(collector.main())
