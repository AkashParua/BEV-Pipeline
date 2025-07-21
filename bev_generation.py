#!/usr/bin/env python3
"""
Bird's‑Eye‑View (BEV) generation pipeline
========================================
This script fuses multi‑camera YOLOv8 detections with a LiDAR point‑cloud to
produce a BEV map.  It is deliberately written as a single, self‑contained file
so you can drop it into a project and iterate.

Key steps
---------
1. **Load calibrations** (JSON) for the six surround cameras and the LiDAR.
2. **Load data**
   * Six RGB images (PNG/JPEG) –  _CAM_BACK, … CAM_FRONT_RIGHT_.
   * LiDAR point cloud in NuScenes/KITTI‑style **pcd.bin** (float32 x y z r …).
3. **Run object detection** on each image with **YOLOv8** (Ultralytics).
4. **Project detections** into the vehicle (ego) frame using intrinsic ➜ camera
   ➜ ego transforms.  We assume a flat road (z≈0) to obtain (x, y) ground
   positions.
5. **Voxelise / slice** the LiDAR points to a top‑down grid.
6. **Render** a composite BEV image (LiDAR intensity + detection boxes).

Quick start
-----------
```bash
pip install ultralytics opencv-python open3d matplotlib scipy

python bev_pipeline.py \
  --image_dir  ./samples/images \
  --lidar_bin  ./samples/pcd.bin \
  --calib_dir  ./samples/calib_json \
  --out        bev_result.png
```

The **calib_dir** must contain seven JSON files:
```
CAM_BACK.json            CAM_FRONT.json
CAM_BACK_LEFT.json       CAM_FRONT_LEFT.json
CAM_BACK_RIGHT.json      CAM_FRONT_RIGHT.json
LIDAR_TOP.json           # (for the LiDAR)
```
Each JSON has keys  `extrinsics`, `intrinsics`, `ego_pose`, `timestamp` exactly
as in the examples you provided.

Limitations
-----------
* Ground‑plane assumption – works for on‑road scenes; for 3‑D boxes use depth.
* No temporal tracking – one frame only.
* Only YOLOv8 bounding‑box centres are re‑projected; full 3‑D cuboids are left
  as an exercise.
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

# --------------- maths utils --------------------------------------------------

def quat_xyzw_to_rot(q: List[float]) -> np.ndarray:
    """Convert [x, y, z, w] quaternion to 3×3 rotation matrix."""
    r = R.from_quat(q)  # scipy expects x,y,z,w
    return r.as_matrix()

def make_se3(rot: np.ndarray, trans: List[float]) -> np.ndarray:
    """Build a 4×4 SE(3) matrix from rotation & translation."""
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = np.array(trans)
    return T

# --------------- calibration --------------------------------------------------

def load_sensor_calib(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (K, T_cam_to_ego, T_ego_to_global)."""
    meta = json.loads(path.read_text())
    K = np.array(meta["intrinsics"]) if meta["intrinsics"] else None
    # camera/LiDAR ➜ ego
    T = make_se3(quat_xyzw_to_rot(meta["extrinsics"]["rotation"]),
                 meta["extrinsics"]["translation"])
    # ego ➜ global (not used for BEV but parsed for completeness)
    T_ego = make_se3(quat_xyzw_to_rot(meta["ego_pose"]["rotation"]),
                     meta["ego_pose"]["translation"])
    return K, T, T_ego

# --------------- lidar --------------------------------------------------------

def load_lidar_bin(bin_path: Path) -> np.ndarray:
    """Load .bin with float32 x,y,z,intensity (and possibly extra fields)."""
    pts = np.fromfile(bin_path, dtype=np.float32)
    pts = pts.reshape(-1, 4)[:, :4]  # [N,4]
    return pts  # x,y,z,i

# --------------- yolo ---------------------------------------------------------

def run_yolo(model: YOLO, img_path: Path) -> List[Tuple[int, float, Tuple[int,int,int,int]]]:
    """Return detections as list of (cls_id, conf, (x1,y1,x2,y2))."""
    res = model(str(img_path))
    preds = res[0].boxes
    dets = []
    for cls, conf, xyxy in zip(preds.cls.cpu().numpy().astype(int),
                               preds.conf.cpu().numpy(),
                               preds.xyxy.cpu().numpy()):
        dets.append((cls, conf, tuple(xyxy.astype(int))))
    return dets

# --------------- projection ---------------------------------------------------

def pixel_box_to_vehicle(K: np.ndarray, T_cam_to_ego: np.ndarray,
                          bbox: Tuple[int, int, int, int], sensor_height: float = 1.5) -> Tuple[float, float]:
    """Project 2‑D bbox centre to (x,y) in ego frame assuming flat ground.

    * K           – 3×3 intrinsic
    * T_cam_to_ego – 4×4 camera➜ego transform
    * bbox        – (x1,y1,x2,y2)
    * sensor_height – approximate camera height above ground (metres)
    """
    u = (bbox[0] + bbox[2]) / 2.0
    v = (bbox[1] + bbox[3]) / 2.0

    # ray in camera coords: K^{-1} [u,v,1]
    cam_ray = np.linalg.inv(K) @ np.array([u, v, 1.0])
    cam_ray = cam_ray / np.linalg.norm(cam_ray)

    # scale such that Z intersects ground (Z = -sensor_height)
    if cam_ray[2] == 0:
        return None
    scale = ( -sensor_height ) / cam_ray[2]
    cam_point = cam_ray * scale  # [x,y,z]

    # homogeneous
    cam_point_h = np.concatenate([cam_point, [1]])
    ego_point = T_cam_to_ego @ cam_point_h
    return float(ego_point[0]), float(ego_point[1])

# --------------- bev grid -----------------------------------------------------

def make_bev(pts: np.ndarray, xlim=(-30, 30), ylim=(0, 60), res=0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxelise point cloud into a 2‑D occupancy/intensity grid.

    Returns (bev, xs, ys) where xs & ys give world coords for grid centres.
    """
    mask = (pts[:,2] > -2.5) & (pts[:,2] < 2.5)  # rough road slice
    pts = pts[mask]

    # keep points within roi
    mask_roi = (pts[:,0] > xlim[0]) & (pts[:,0] < xlim[1]) & \
               (pts[:,1] > ylim[0]) & (pts[:,1] < ylim[1])
    pts = pts[mask_roi]

    # convert to grid indices
    xs = np.arange(xlim[0], xlim[1], res)
    ys = np.arange(ylim[0], ylim[1], res)
    bev = np.zeros((len(xs), len(ys)), dtype=np.float32)

    ix = ((pts[:,0] - xlim[0]) / res).astype(int)
    iy = ((pts[:,1] - ylim[0]) / res).astype(int)

    # accumulate max intensity per cell
    for x_idx, y_idx, inten in zip(ix, iy, pts[:,3]):
        if 0 <= x_idx < len(xs) and 0 <= y_idx < len(ys):
            bev[x_idx, y_idx] = max(bev[x_idx, y_idx], inten)
    return bev, xs, ys

# --------------- visualisation ------------------------------------------------

def draw_bev(bev: np.ndarray, xs: np.ndarray, ys: np.ndarray,
             det_xy: List[Tuple[float,float]],
             figsize=(8, 12), out_path: Path = None):
    plt.figure(figsize=figsize)
    plt.imshow(bev.T[::-1], extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap='gray')
    if det_xy:
        xs_d, ys_d = zip(*det_xy)
        plt.scatter(xs_d, ys_d, marker='x', s=80, linewidths=2, c='red')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Bird\'s‑Eye‑View: LiDAR intensity + YOLO detections')
    plt.grid(True, linestyle=':')
    plt.axis('equal')
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

# --------------- main ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate BEV from multi‑camera images and LiDAR.')
    parser.add_argument('--image_dir', type=Path, required=True, help='Directory with 6 camera images')
    parser.add_argument('--lidar_bin', type=Path, required=True, help='LiDAR .bin file')
    parser.add_argument('--calib_dir', type=Path, required=True, help='Directory containing sensor JSON calibrations')
    parser.add_argument('--model', default='yolov8n.pt', help='Ultralytics YOLOv8 model')
    parser.add_argument('--out', type=Path, default=None, help='Output PNG for BEV visualisation')
    args = parser.parse_args()

    # load calibrations
    cams = [
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'
    ]
    cam_calibs: Dict[str, Tuple[np.ndarray,np.ndarray,np.ndarray]] = {}
    for c in cams:
        cam_calibs[c] = load_sensor_calib(args.calib_dir / f"{c}.json")

    # lidar
    K_lidar, T_lidar_to_ego, _ = load_sensor_calib(args.calib_dir / 'LIDAR_TOP.json')
    pts = load_lidar_bin(args.lidar_bin)
    # transform lidar points to ego frame
    pts_h = np.hstack([pts[:, :3], np.ones((pts.shape[0],1))])  # [N,4]
    pts_ego = (T_lidar_to_ego @ pts_h.T).T  # [N,4]
    pts_ego = np.hstack([pts_ego[:, :3], pts[:,3:4]])  # keep intensity

    # make BEV grid
    bev, xs, ys = make_bev(pts_ego)

    # run YOLO on each image
    model = YOLO(args.model)
    det_xy: List[Tuple[float,float]] = []
    for c in cams:
        img_path = args.image_dir / f"{c}.jpg"
        if not img_path.exists():
            print(f"Warning: {img_path} missing, skipping")
            continue
        detections = run_yolo(model, img_path)
        K, T_cam_to_ego, _ = cam_calibs[c]
        for cls_id, conf, bbox in detections:
            # filter for cars / trucks / pedestrians etc if desired
            xy = pixel_box_to_vehicle(K, T_cam_to_ego, bbox)
            if xy:
                det_xy.append(xy)

    # draw
    draw_bev(bev, xs, ys, det_xy, out_path=args.out)
    if args.out:
        print(f"Saved BEV visualisation to {args.out}")

if __name__ == '__main__':
    main()
