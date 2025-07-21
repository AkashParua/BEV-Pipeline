#!/usr/bin/env python3
"""
Bird's-Eye-View (BEV) generation pipeline — **directory-aware version**
====================================================================
This revision adapts the earlier script to the **nested directory layout** you
described:

```
<root>
 ├─ images/
 │   ├─ CAM_BACK/          ─┐  (exact folder names)
 │   │   ├─ frame.png        │  image (png/jpg)
 │   │   └─ calib.json       │  calibration for this cam
 │   ├─ CAM_BACK_LEFT/      │  ...
 │   ├─ CAM_BACK_RIGHT/     │
 │   ├─ CAM_FRONT/          │
 │   ├─ CAM_FRONT_LEFT/     │
 │   └─ CAM_FRONT_RIGHT/    │
 └─ lidar/
     ├─ scan.bin
     └─ calib.json
```

Key differences from the previous version
-----------------------------------------
* **No separate `calib_dir` flag** — calibration JSONs are loaded **alongside
  each image** (and the LiDAR JSON alongside the `.bin`).
* **Auto-detect** image filename (first `*.png`/`*.jpg`) and JSON in each
  folder.
* **`--image_root`** points to the *parent* directory containing the six camera
  folders; **`--lidar_dir`** points to the folder holding the LiDAR files.

Quick start
-----------
```bash
pip install ultralytics opencv-python open3d matplotlib scipy

python bev_pipeline.py \
  --image_root ./data/images \
  --lidar_dir  ./data/lidar \
  --out        bev.png
```

Limitations / TODO remain as before (flat-ground assumption, single frame,…).
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# maths utilities
# ---------------------------------------------------------------------------

def quat_xyzw_to_rot(q: List[float]) -> np.ndarray:
    """Convert [x, y, z, w] quaternion to 3×3 rotation matrix."""
    return R.from_quat(q).as_matrix()

def make_se3(rot: np.ndarray, trans: List[float]) -> np.ndarray:
    """Return 4×4 SE(3) from rotation + translation."""
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = np.array(trans)
    return T

# ---------------------------------------------------------------------------
# calibration loaders
# ---------------------------------------------------------------------------

def load_sensor_calib(json_path: Path):
    meta = json.loads(json_path.read_text())
    K = np.array(meta["intrinsics"]) if meta["intrinsics"] else None
    T_sensor_to_ego = make_se3(
        quat_xyzw_to_rot(meta["extrinsics"]["rotation"]),
        meta["extrinsics"]["translation"],
    )
    T_ego_to_global = make_se3(
        quat_xyzw_to_rot(meta["ego_pose"]["rotation"]),
        meta["ego_pose"]["translation"],
    )
    return K, T_sensor_to_ego, T_ego_to_global

# ---------------------------------------------------------------------------
# LiDAR
# ---------------------------------------------------------------------------

def load_lidar_bin(bin_path: Path) -> np.ndarray:
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :4]  # x,y,z,intensity

# ---------------------------------------------------------------------------
# YOLO helper
# ---------------------------------------------------------------------------

def run_yolo(model: YOLO, img_path: Path):
    res = model(str(img_path))
    boxes = res[0].boxes
    dets = []
    for cls, conf, xyxy in zip(boxes.cls.cpu().numpy().astype(int),
                               boxes.conf.cpu().numpy(),
                               boxes.xyxy.cpu().numpy().astype(int)):
        dets.append((cls, conf, tuple(xyxy)))
    return dets

# ---------------------------------------------------------------------------
# projection utils
# ---------------------------------------------------------------------------

def pixel_box_to_vehicle(K, T_cam_to_ego, bbox, cam_height: float = 1.5):
    u = (bbox[0] + bbox[2]) / 2
    v = (bbox[1] + bbox[3]) / 2
    cam_ray = np.linalg.inv(K) @ np.array([u, v, 1.0])
    cam_ray /= cam_ray[2] if cam_ray[2] != 0 else 1.0
    scale = -cam_height / cam_ray[2]
    cam_point = cam_ray * scale
    p_ego = T_cam_to_ego @ np.append(cam_point, 1)
    return float(p_ego[0]), float(p_ego[1])

# ---------------------------------------------------------------------------
# BEV grid
# ---------------------------------------------------------------------------

def make_bev(pts, xlim=(-30, 30), ylim=(0, 60), res=0.1):
    mask = (pts[:, 2] > -2.5) & (pts[:, 2] < 2.5)
    pts = pts[mask]
    mask_roi = (xlim[0] < pts[:, 0]) & (pts[:, 0] < xlim[1]) & \
               (ylim[0] < pts[:, 1]) & (pts[:, 1] < ylim[1])
    pts = pts[mask_roi]

    xs = np.arange(xlim[0], xlim[1], res)
    ys = np.arange(ylim[0], ylim[1], res)
    bev = np.zeros((len(xs), len(ys)), dtype=np.float32)
    ix = ((pts[:, 0] - xlim[0]) / res).astype(int)
    iy = ((pts[:, 1] - ylim[0]) / res).astype(int)
    for x, y, inten in zip(ix, iy, pts[:, 3]):
        if 0 <= x < len(xs) and 0 <= y < len(ys):
            bev[x, y] = max(bev[x, y], inten)
    return bev, xs, ys

# ---------------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------------

def draw_bev(bev, xs, ys, objs_xy, out: Path = None):
    plt.figure(figsize=(8, 12))
    plt.imshow(bev.T[::-1], extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap='gray')
    if objs_xy:
        ox, oy = zip(*objs_xy)
        plt.scatter(ox, oy, c='red', marker='x', s=80, linewidths=2)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('BEV: LiDAR intensity + YOLO detections')
    plt.grid(linestyle=':')
    plt.axis('equal')
    if out:
        plt.savefig(out, dpi=300, bbox_inches='tight')
    else:
        plt.show()

# ---------------------------------------------------------------------------
# helpers for directory layout
# ---------------------------------------------------------------------------
CAM_DIRS = [
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'
]

def find_single(path: Path, exts):
    for ext in exts:
        cand = list(path.glob(ext))
        if cand:
            return cand[0]
    raise FileNotFoundError(f"None of {exts} in {path}")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Generate BEV with nested dirs.')
    ap.add_argument('--image_root', type=Path, required=True,
                    help='Parent directory containing six camera folders')
    ap.add_argument('--lidar_dir', type=Path, required=True,
                    help='Directory with LiDAR .bin and calib.json')
    ap.add_argument('--model', default='yolov8n.pt', help='YOLOv8 weight')
    ap.add_argument('--out', type=Path, default=None, help='Output PNG')
    args = ap.parse_args()

    # ---------------- load LiDAR ------------------
    bin_path = find_single(args.lidar_dir, ['*.bin'])
    lidar_json = find_single(args.lidar_dir, ['*.json'])
    K_lidar, T_lidar_to_ego, _ = load_sensor_calib(lidar_json)
    pts = load_lidar_bin(bin_path)
    pts_h = np.c_[pts[:, :3], np.ones(len(pts))]
    pts_ego = (T_lidar_to_ego @ pts_h.T).T
    pts_ego = np.c_[pts_ego[:, :3], pts[:, 3]]  # keep intensity

    bev, xs, ys = make_bev(pts_ego)

    # ---------------- cameras ---------------------
    model = YOLO(args.model)
    objs_xy = []
    for cam in CAM_DIRS:
        cam_folder = args.image_root / cam
        if not cam_folder.is_dir():
            print(f"Warning: {cam_folder} missing; skipping")
            continue
        img_path = find_single(cam_folder, ['*.jpg', '*.png'])
        calib_path = find_single(cam_folder, ['*.json'])
        K, T_cam_to_ego, _ = load_sensor_calib(calib_path)
        dets = run_yolo(model, img_path)
        for cls_id, conf, bbox in dets:
            xy = pixel_box_to_vehicle(K, T_cam_to_ego, bbox)
            if xy:
                objs_xy.append(xy)

    # ---------------- render ----------------------
    draw_bev(bev, xs, ys, objs_xy, args.out)
    if args.out:
        print(f"Saved BEV visualisation to {args.out}")

if __name__ == '__main__':
    main()