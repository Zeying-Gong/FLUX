# FLUX Tracking Dataset Format

## Overview

Each tracking episode generates a directory structure under the output root:

```
logs/tracking_<timestamp>/
├── metrics.csv                         # Per-episode aggregate metrics
└── episodes/
    ├── task_description_ep0000.txt      # "Track a woman wearing lime top, blue pants, white shoes"
    ├── task_description_ep0001.txt
    ├── camera_info.json                 # Camera intrinsics + extrinsics (LeRobot compatible)
    └── episode_0000/
        ├── rgb/
        │   ├── 00000.png               # uint8 RGB, 640×360
        │   ├── 00001.png
        │   └── ...
        ├── depth_mm/
        │   ├── 00000.png               # uint16 PNG, depth in millimeters, 640×360
        │   ├── 00001.png
        │   └── ...
        ├── depth_viz/
        │   └── 00000.png               # (first frame only) Jet colormap for visual inspection
        ├── target_crops/
        │   └── 00000.png               # (first frame only) RGB crop of target person
        └── frames/
            └── frame_data.npz           # Per-step data, see NPZ Format below
```

---

## Data Location Quick Reference

| What you need | Found in | Format |
|---|---|---|
| RGB image (per step) | `episode_XXXX/rgb/{step}.png` | uint8 PNG, 640×360 |
| Depth (per step, mm) | `episode_XXXX/depth_mm/{step}.png` | uint16 PNG (I;16) |
| Depth viz (first frame only) | `episode_XXXX/depth_viz/00000.png` | uint8 colormap PNG |
| Target bbox crop (first frame) | `episode_XXXX/target_crops/00000.png` | uint8 RGB PNG |
| Camera intrinsics / extrinsics | `episode_XXXX/camera_info.json` | JSON |
| **Per-step training data** | **`episode_XXXX/frames/frame_data.npz`** | **NumPy `.npz` (see below)** |
| Task description (text) | `task_description_epXXXX.txt` | UTF-8 text |
| Episode quality metrics | `metrics.csv` | CSV |

### What's inside `frame_data.npz`

| Component | NPZ Key(s) | Shape | Location |
|---|---|---|---|
| **State vector** | `observation.state` | (T, 9) | **NPZ** |
| **Action vector** | `action` | (T, 2) | **NPZ** |
| Target 2D point (u,v) | `target_uv_u`, `target_uv_v` | (T,) | **NPZ** |
| Target 2D bbox | `target_bbox_x1/y1/x2/y2` | (T,) | **NPZ** |
| Target visible flag | `target_visible` | (T,) | **NPZ** |
| Robot trajectory | `robot_trajectory` | (T, 3) | **NPZ** |
| Target trajectory | `target_trajectory` | (T, 3) | **NPZ** |
| Task description | `task_description` | (bytes) | **NPZ** + `.txt` sidecar |
| RGB image | — | — | `rgb/{step}.png` **(separate file)** |
| Depth image | — | — | `depth_mm/{step}.png` **(separate file)** |
| Target crop | — | — | `target_crops/00000.png` **(separate file)** |
| Camera params | — | — | `camera_info.json` **(separate file)** |

---

## NPZ Format (`frame_data.npz`)

34 keys, all `float64` unless noted. T = number of steps = `episode_length`.

### Core (LeRobot-compatible)

| Key | Shape | Type | Description |
|---|---|---|---|
| `observation.state` | (T, 9) | float64 | Robot + target state vector |
| `action` | (T, 2) | float64 | Velocity command `[linear, angular]` |

### observation.state columns

```
[robot_x, robot_y, robot_z, robot_yaw,
 target_rel_x, target_rel_y, target_rel_z,
 target_dist, target_bearing_deg]
```

| Index | Field | Unit | Description |
|-------|-------|------|-------------|
| 0 | robot_x | m | Robot world X |
| 1 | robot_y | m | Robot world Y |
| 2 | robot_z | m | Robot base height (Go2: 0.40, Dingo: 0.10) |
| 3 | robot_yaw | rad | Robot heading |
| 4 | target_rel_x | m | Target relative X (forward in robot frame, ROS convention) |
| 5 | target_rel_y | m | Target relative Y (left in robot frame) |
| 6 | target_rel_z | m | Target relative Z (up in robot frame) |
| 7 | target_dist | m | Euclidean distance to target |
| 8 | target_bearing_deg | deg | Bearing angle (+ = left of robot) |

### action columns

| Index | Field | Unit | Description |
|-------|-------|------|-------------|
| 0 | action_linear | m/s | Forward velocity command |
| 1 | action_angular | rad/s | Angular velocity command (+ = left turn) |

### Task Description

| Key | Shape | Type | Description |
|---|---|---|---|
| `task_description` | (N,) | uint8 | UTF-8 encoded text, e.g. `"Track a woman wearing lime top, blue pants, white shoes"` |

Decode: `task_description.tobytes().decode("utf-8")`

### Trajectory Data

| Key | Shape | Description |
|---|---|---|
| `robot_trajectory` | (T, 3) | `[robot_x, robot_y, robot_z]` per step |
| `target_trajectory` | (T, 3) | `[target_x, target_y, target_z]` per step |

### Target Image Projection

| Key | Shape | Description |
|---|---|---|
| `target_uv_u` | (T,) | Target center projected X in pixels (ROS: `u = -fx * Y_cam / X_cam + cx`) |
| `target_uv_v` | (T,) | Target center projected Y in pixels (ROS: `v = -fy * Z_cam / X_cam + cy`) |
| `target_bbox_x1` | (T,) | 2D bounding box left (clamped to image) |
| `target_bbox_y1` | (T,) | 2D bounding box top (clamped to image) |
| `target_bbox_x2` | (T,) | 2D bounding box right (clamped to image) |
| `target_bbox_y2` | (T,) | 2D bounding box bottom (clamped to image) |
| `target_visible` | (T,) | int64: 1 = visible (unoccluded), 0 = occluded |

### Raw Per-Step Fields

| Key | Description |
|---|---|
| `robot_pos_x/y/z` | Robot world position (m) |
| `robot_yaw` | Robot heading (rad) |
| `target_pos_x/y/z` | Target world position (m) |
| `target_rel_x/y/z` | Target in robot frame (m) |
| `target_dist` | Euclidean distance (m) |
| `target_bearing_deg` | Bearing (degrees) |
| `target_elevation_deg` | Elevation angle (degrees) |
| `step` | int64: step index |
| `timestamp` | Time in seconds = `step * physics_dt * decimation` |
| `mode` | str: current pursuit mode (TRACK, STANDOFF, APPROACH, EVADE, DETOUR, SEARCH, RECOVERY) |
| `action_linear` | Linear velocity command (m/s) |
| `action_angular` | Angular velocity command (rad/s) |
| `dist_to_target` | Same as target_dist |
| `heading_error_deg` | Angular error to target (degrees) |
| `contact_force` | Max contact force on robot (N) |
| `ped_min_dist` | Minimum distance to non-target pedestrians (m) |

---

## Image Data

### RGB (`rgb/{step}.png`)

- Format: uint8 PNG, 3 channels
- Resolution: 640×360 (default), varies by `--camera_type`
- Camera: Robot-mounted forward-facing (ROS convention: +X forward, +Y left, +Z up)

### Depth (`depth_mm/{step}.png`)

- Format: uint16 PNG (mode `I;16`), single channel
- Resolution: same as RGB
- Value: depth in **millimeters**
- Convention: **distance to image plane** (perpendicular to image plane, NOT Euclidean ray distance)
- Background (no valid depth): 0
- Reading in Python:
  ```python
  from PIL import Image
  import numpy as np
  depth_mm = np.asarray(Image.open("depth_mm/00000.png"), dtype=np.uint16)
  depth_m = depth_mm.astype(np.float32) / 1000.0
  ```
- Depth source: **Composite** — RTX depth (characters, pixel-exact) + PhysX raycast (static scene background).
  RTX depth is used where available (characters), PhysX fills the rest (3DGS background).

### Target Crop (`target_crops/00000.png`)

- Only saved for the first frame of each episode
- RGB crop of the tracked person's 2D bounding box
- Intended for visual target specification / reference image

---

## Camera Info (`camera_info.json`)

```json
{
  "camera": {
    "model": "pinhole",
    "axes": "ros",
    "width": 640,
    "height": 360,
    "intrinsics": {
      "fx": 476.6,
      "fy": 476.6,
      "cx": 320.0,
      "cy": 180.0,
      "k": [[476.6, 0, 320.0], [0, 476.6, 180.0], [0, 0, 1]]
    },
    "extrinsics_robot_to_camera": {
      "translation": [0.0, 0.0, 0.3],
      "camera_link": "base"
    },
    "clipping_range": [0.01, 100.0],
    "focal_length_mm": 1.4,
    "horizontal_aperture_mm": 1.88
  }
}
```

---

## Metrics CSV (`metrics.csv`)

Per-episode aggregate statistics for data quality filtering.

| Column | Description |
|---|---|
| `episode_id` | Episode index |
| `success` | 1 if tracking_rate ≥ 0.8, final_dist ≤ max, no recovery, valid end reason |
| `had_recovery` | 1 if robot was stuck and triggered recovery at least once |
| `episode_length` | Number of steps |
| `tracking_rate` | Fraction of steps where robot was in tracking window AND target visible |
| `collision` | Number of collision events |
| `initial_dist` | Robot-target distance at step 0 (m) |
| `final_dist` | Robot-target distance at final step (m) |
| `avg_dist` | Average distance over episode (m) |
| `avg_heading_error_deg` | Average angular error to target (degrees) |
| `done_reason` | End reason: `max_steps`, `char_done`, `had_recovery`, `stuck`, `collision`, `planning_failed`, `target_lost` |

---

## Data Collection Parameters

| Argument | Default | Description |
|---|---|---|
| `--robot_type` | `go2` | `go2` (Unitree Go2, kinematic), `dingo` (diff drive), `g1` (humanoid) |
| `--character_speed` | 0.5 | Character walk speed fraction [0-1] |
| `--camera_type` | `default` | `default` (640×360, 67.8° HFOV), `realsense_d435i`, `zed` |
| `--max_steps` | 500 | Max steps per episode |
| `--allow_recovery` | True | Save episodes even with recovery events |
| `--min_tracking_rate` | 0.0 | Min tracking rate to save (0.0 = save all) |
| `--min_visible_rate` | 0.0 | Min target-in-frame rate (0.0 = save all) |
| `--max_consecutive_lost` | 10 | Abort if target out of frame for this many consecutive steps |
