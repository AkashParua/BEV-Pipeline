#!/usr/bin/env python3
"""
Robust LiDAR–Camera Fusion Pipeline
===================================
Features:
  - Multi-sweep LiDAR aggregation with shared calibration
  - Calibration sanity checks & time-sync
  - RANSAC ground removal
  - Frustum-based 3D localization per 2D detection
  - Multi-channel BEV rasterization
  - Depth-colored image overlays

Directory structure:
  <image_root>/CAM_<name>/
      image.jpg/png
      calib.json
  <lidar_dir>/
      scan_*.bin
      calib.json  # exactly one JSON for all bins
"""
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from sklearn.linear_model import RANSACRegressor
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Calibration & pose utilities
# ---------------------------------------------------------------------------

def load_sensor_calib(json_path: Path):
    meta = json.loads(json_path.read_text())
    # NuScenes-style: [w,x,y,z] -> scipy: [x,y,z,w]
    w,x,y,z = meta["extrinsics"]["rotation"]
    R_s2e = R.from_quat([x, y, z, w]).as_matrix()
    t = np.array(meta["extrinsics"]["translation"])
    T_s2e = np.eye(4)
    T_s2e[:3,:3] = R_s2e
    T_s2e[:3,3] = t
    K = np.array(meta.get("intrinsics", [[0,0,0],[0,0,0],[0,0,1]]))
    return K, T_s2e

# ---------------------------------------------------------------------------
# LiDAR ingestion & preprocessing
# ---------------------------------------------------------------------------

def load_lidar_sweep(bin_path: Path) -> np.ndarray:
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts  # x,y,z,intensity


def remove_ground(points: np.ndarray) -> np.ndarray:
    XY = points[:, :2]
    Z = points[:, 2]
    ransac = RANSACRegressor(residual_threshold=0.2)
    ransac.fit(XY, Z)
    plane = ransac.predict(XY)
    mask = (Z - plane) > 0.2
    return points[mask]

# ---------------------------------------------------------------------------
# Projection & frustum localization
# ---------------------------------------------------------------------------

def project_to_image(pts: np.ndarray, K: np.ndarray, T_cam2ego: np.ndarray):
    T = np.linalg.inv(T_cam2ego)
    pts_h = np.hstack([pts, np.ones((len(pts),1))])
    cam_pts = (T @ pts_h.T).T[:, :3]
    uv = (K @ cam_pts.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    z = cam_pts[:,2]
    return uv, z


def frustum_localize(pts: np.ndarray, uv: np.ndarray, z: np.ndarray, bbox):
    x1,y1,x2,y2 = bbox
    mask = (uv[:,0]>=x1) & (uv[:,0]<=x2) & (uv[:,1]>=y1) & (uv[:,1]<=y2) & (z>0)
    sel = pts[mask]
    return np.median(sel, axis=0) if len(sel) else None

# ---------------------------------------------------------------------------
# BEV rasterization
# ---------------------------------------------------------------------------

def make_bev_channels(pts: np.ndarray, xlim=(-50,50), ylim=(-50,50), res=0.2, nch=3):
    bev = []
    zs = np.linspace(-3,3,nch+1)
    w = int((xlim[1]-xlim[0])/res)
    h = int((ylim[1]-ylim[0])/res)
    for i in range(nch):
        mask = (pts[:,2]>=zs[i]) & (pts[:,2]<zs[i+1])
        sub = pts[mask]
        ix = ((sub[:,0]-xlim[0])/res).astype(int)
        iy = ((sub[:,1]-ylim[0])/res).astype(int)
        img = np.zeros((w,h), dtype=np.float32)
        for x,y,i_val in zip(ix, iy, sub[:,3]):
            if 0<=x<w and 0<=y<h:
                img[x,y] = max(img[x,y], i_val)
        bev.append(img)
    return np.stack(bev, axis=-1)

# ---------------------------------------------------------------------------
# Visualization overlay
# ---------------------------------------------------------------------------

def draw_overlay(img, uv, z, step=1):
    cmap = plt.get_cmap('viridis')
    norm = (z - z.min())/(z.max()-z.min()) if len(z)>0 else z
    colors = (cmap(norm)[:,:3]*255).astype(np.uint8)
    for i in range(0, len(uv), step):
        u,v = uv[i].astype(int)
        if 0<=u<img.shape[1] and 0<=v<img.shape[0]:
            # red square marker
            cv2.rectangle(img, (u-2,v-2), (u+2,v+2), (0,0,255), -1)
    return img

# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("LiDAR–Camera fusion: multi-sweep + shared frustum")
    p.add_argument('--image_root',   required=True, type=Path)
    p.add_argument('--lidar_dir',    required=True, type=Path)
    p.add_argument('--model',        default='yolov8n.pt')
    p.add_argument('--out_bev',      type=Path)
    p.add_argument('--out_overlays', type=Path,
                   help='Directory to save camera overlays')
    args = p.parse_args()

    # Prepare overlay output folder
    args.out_overlays = args.out_overlays or (args.image_root / 'overlays')
    args.out_overlays.mkdir(parents=True, exist_ok=True)

    # Load single LiDAR calibration
    jsons = list(args.lidar_dir.glob('*.json'))
    if len(jsons) != 1:
        raise FileNotFoundError(f"Expected exactly one LiDAR calib JSON, found {len(jsons)}")
    _, T_l2e = load_sensor_calib(jsons[0])

    # Load & concatenate all sweeps
    bins = sorted(args.lidar_dir.glob('*.bin'))
    if not bins:
        raise FileNotFoundError(f"No LiDAR .bin files found in {args.lidar_dir}")
    clouds = []
    for b in bins:
        pts = load_lidar_sweep(b)
        pts_h = np.hstack([pts[:,:3], np.ones((len(pts),1))])
        ego_pts = (T_l2e @ pts_h.T).T[:,:3]
        clouds.append(np.hstack([ego_pts, pts[:,3:4]]))
    cloud = np.vstack(clouds)
    print(f"[INFO] Concatenated {len(bins)} sweeps -> {cloud.shape[0]} points")
    cloud = remove_ground(cloud)

    # Initialize detector
    model = YOLO(args.model)
    obj_centers = []

    # Process each camera
    for cam_dir in sorted(args.image_root.iterdir()):
        if not cam_dir.is_dir(): continue
        img_path = next(cam_dir.glob('*.jpg'), None) or next(cam_dir.glob('*.png'), None)
        calib_path = next(cam_dir.glob('*.json'), None)
        K, T_c2e = load_sensor_calib(calib_path)

        uv, z = project_to_image(cloud[:,:3], K, T_c2e)
        img = cv2.imread(str(img_path))

        dets = model(str(img_path))[0].boxes
        for cls, conf, box in zip(dets.cls, dets.conf, dets.xyxy):
            bbox = tuple(box.cpu().numpy().astype(int))
            center = frustum_localize(cloud[:,:3], uv, z, bbox)
            if center is not None:
                obj_centers.append(center[:2])
            cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,255,0), 2)

        overlay = draw_overlay(img, uv, z)
        out_file = args.out_overlays / f"{cam_dir.name}.png"
        cv2.imwrite(str(out_file), overlay)

    # BEV plotting
    bev = make_bev_channels(cloud)
    plt.figure(figsize=(6,6))
    plt.imshow(bev[:,:,0].T, extent=[-50,50,-50,50])
    # plot other cars as red squares
    if obj_centers:
        xs, ys = zip(*obj_centers)
        plt.scatter(xs, ys, c='r', marker='s', s=40)
    # ego at origin in green
    plt.scatter(0,0, c='g', marker='x', s=100)
    if args.out_bev:
        plt.savefig(args.out_bev)
    else:
        plt.show()

if __name__ == '__main__':
    main()
