#!/usr/bin/env python3
"""
Unified LiDAR ⇆ Camera Fusion Pipeline
======================================
This single script now provides **both** views:

1. **Bird's‑Eye‑View (BEV)** – LiDAR intensity grid with YOLOv8 detections
   projected onto the ground plane (unchanged from previous version).
2. **Image Overlays** – Raw LiDAR points re‑projected **into every camera
   image** (depth‑coded) *plus* the YOLO bounding boxes.

Directory layout (unchanged)
----------------------------
```
<root>
 ├─ images/
 │   ├─ CAM_FRONT/ …  # each folder: 1 image + calib.json
 └─ lidar/
     ├─ scan.bin
     └─ calib.json
```

Quick start
-----------
```bash
pip install ultralytics opencv-python matplotlib scipy open3d

python bev_pipeline.py \
  --image_root ./data/images \
  --lidar_dir  ./data/lidar \
  --out_bev    bev.png \
  --overlay_dir overlays
```
* `--out_bev`  PNG for the top‑down view (optional).
* `--overlay_dir` directory to save each camera overlay (optional; created if it
  doesn’t exist). If omitted, overlays are just *displayed* with `cv2.imshow`.

Limitations
-----------
* Flat‑ground assumption for **detection→world** step still applies.
* LiDAR projected into image is **raw points** (no mesh/semantic labels).
* Script loops **one frame**; to process sequences, wrap a loop around `main()`.
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Maths utilities
# ---------------------------------------------------------------------------

def quat_xyzw_to_rot(q: List[float]) -> np.ndarray:
    return R.from_quat(q).as_matrix()

def make_se3(rot: np.ndarray, trans: List[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = np.array(trans)
    return T

# ---------------------------------------------------------------------------
# Calibration loading
# ---------------------------------------------------------------------------

def load_sensor_calib(json_path: Path):
    meta = json.loads(json_path.read_text())
    K = np.array(meta["intrinsics"]) if meta["intrinsics"] else None
    T_sensor_to_ego = make_se3(
        quat_xyzw_to_rot(meta["extrinsics"]["rotation"]),
        meta["extrinsics"]["translation"],
    )
    T_ego_to_sensor = np.linalg.inv(T_sensor_to_ego)
    return K, T_sensor_to_ego, T_ego_to_sensor

# ---------------------------------------------------------------------------
# LiDAR helpers
# ---------------------------------------------------------------------------

def load_lidar_bin(bin_path: Path) -> np.ndarray:
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # x y z i
    return pts[:, :4]

# ---------------------------------------------------------------------------
# YOLO detection
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
# Point projection utilities
# ---------------------------------------------------------------------------

def pixel_box_to_vehicle(K, T_cam_to_ego, bbox, cam_height=1.5):
    u = (bbox[0] + bbox[2]) / 2
    v = (bbox[1] + bbox[3]) / 2
    cam_ray = np.linalg.inv(K) @ np.array([u, v, 1.0])
    cam_ray /= cam_ray[2] if cam_ray[2] != 0 else 1.0
    scale = -cam_height / cam_ray[2]
    cam_point = cam_ray * scale
    p_ego = T_cam_to_ego @ np.append(cam_point, 1)
    return float(p_ego[0]), float(p_ego[1])


def ego_points_to_image(K, T_ego_to_cam, pts_ego: np.ndarray):
    """Project ego‑frame points into camera. Returns (uv, mask, depth)."""
    # to camera frame
    pts_cam = (T_ego_to_cam @ np.c_[pts_ego[:, :3], np.ones(len(pts_ego))].T).T
    z = pts_cam[:, 2]
    mask = z > 0.1  # in front of cam
    pts_cam = pts_cam[mask]
    z = z[mask]
    uv = (K @ pts_cam[:, :3].T).T
    uv = (uv[:, :2].T / uv[:, 2]).T  # perspective divide
    return uv, mask, z

# ---------------------------------------------------------------------------
# BEV generation
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
# Visualisation helpers
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


def colourise_depth(depth, min_d=0, max_d=70):
    norm = np.clip((depth - min_d) / (max_d - min_d), 0, 1)
    cmap = plt.get_cmap('jet')
    colours = (np.array(cmap(norm))[:, :3] * 255).astype(np.uint8)
    return colours

# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------
CAM_DIRS = [
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'
]

def find_single(path: Path, patterns):
    for pat in patterns:
        lst = list(path.glob(pat))
        if lst:
            return lst[0]
    raise FileNotFoundError(f"None of {patterns} in {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("LiDAR–Camera fusion: BEV + image overlays")
    p.add_argument('--image_root', type=Path, required=True)
    p.add_argument('--lidar_dir', type=Path, required=True)
    p.add_argument('--model', default='yolov8n.pt')
    p.add_argument('--out_bev', type=Path, default=None)
    p.add_argument('--overlay_dir', type=Path, default=None,
                   help='If set, saves per‑camera overlay PNGs here')
    p.add_argument('--sample_step', type=int, default=2,
                   help='Keep every N‑th lidar point for overlay speed')
    args = p.parse_args()

    # ---------- LiDAR ----------
    # find ALL .pcd.bin files instead of one
    lidar_bins = sorted(args.lidar_dir.glob('*.pcd.bin'))
    if not lidar_bins:
        raise FileNotFoundError(f'No .pcd.bin files found in {args.lidar_dir}')

    # still assume ONE calib JSON (shared for all scans in the folder)
    lidar_json = find_single(args.lidar_dir, ['*.json'])
    K_lid, T_lid_to_ego, T_ego_to_lid = load_sensor_calib(lidar_json)

    # load & concatenate every scan
    pts_all: list[np.ndarray] = []
    for bin_path in lidar_bins:
        pts_all.append(load_lidar_bin(bin_path))          # shape (N,4)

    pts = np.vstack(pts_all)                              # (N_total,4)

    # transform once into ego frame
    pts_h = np.c_[pts[:, :3], np.ones(len(pts))]
    pts_ego = (T_lid_to_ego @ pts_h.T).T                  # (N_total,4)
    pts_ego = np.c_[pts_ego[:, :3], pts[:, 3]]            # keep intensity


    # ---------- Cameras ----------
    model = YOLO(args.model)
    objs_xy = []
    if args.overlay_dir:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)

    for cam in CAM_DIRS:
        folder = args.image_root / cam
        if not folder.is_dir():
            print(f"[WARN] {folder} missing; skipping camera")
            continue
        img_path = find_single(folder, ['*.jpg', '*.png'])
        calib_path = find_single(folder, ['*.json'])
        K, T_cam_to_ego, T_ego_to_cam = load_sensor_calib(calib_path)

        # --- YOLO detections ---
        dets = run_yolo(model, img_path)
        for _, _, bbox in dets:
            pt = pixel_box_to_vehicle(K, T_cam_to_ego, bbox)
            if pt:
                objs_xy.append(pt)

        # --- LiDAR → image ---
        uv, mask, depth = ego_points_to_image(K, T_ego_to_cam, pts_ego[::args.sample_step])
        img = cv2.imread(str(img_path))
        colours = colourise_depth(depth)
        for (u, v), col in zip(uv.astype(int), colours):
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 1, tuple(int(c) for c in col), -1)
        # draw YOLO boxes
        for cls_id, conf, bbox in dets:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
        # save or show
        if args.overlay_dir:
            out_path = args.overlay_dir / f"{cam}.png"
            cv2.imwrite(str(out_path), img)
        else:
            cv2.imshow(cam, img)
            cv2.waitKey(1)

    if not args.overlay_dir:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ---------- BEV visualisation ----------
    bev, xs, ys = make_bev(pts_ego)
    draw_bev(bev, xs, ys, objs_xy, args.out_bev)
    if args.out_bev:
        print(f"[INFO] saved BEV to {args.out_bev}")

if __name__ == '__main__':
    main()
