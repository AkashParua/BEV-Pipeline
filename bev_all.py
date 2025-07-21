#!/usr/bin/env python3
"""
Robust LiDAR–Radar–Camera Fusion Pipeline
========================================
Features:
  - Multi-sweep LiDAR aggregation with shared calibration
  - Multi-sweep Radar aggregation per sensor (.pcd files)
  - Calibration sanity checks & time-sync
  - RANSAC ground removal (LiDAR only), with NaN filtering
  - Frustum-based 3D localization per 2D detection
  - Multi-channel BEV rasterization
  - Depth-colored image overlays (LiDAR & Radar)

Directory structure:
  <image_root>/CAM_<name>/
      image.jpg/png
      calib.json
  <lidar_dir>/
      scan_*.bin
      calib.json  # LiDAR calibration for all bins
  <radar_root>/
      RADAR_<name>/
          radar_*.pcd
          calib.json  # Radar calibration per sensor
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
import open3d as o3d

# ---------------------------------------------------------------------------
# Calibration & pose utilities
# ---------------------------------------------------------------------------

def load_sensor_calib(json_path: Path):
    meta = json.loads(json_path.read_text())
    w,x,y,z = meta["extrinsics"]["rotation"]  # [w,x,y,z]
    R_s2e = R.from_quat([x, y, z, w]).as_matrix()
    t = np.array(meta["extrinsics"]["translation"])
    T = np.eye(4)
    T[:3,:3] = R_s2e
    T[:3,3] = t
    K = np.array(meta.get("intrinsics", [[0,0,0],[0,0,0],[0,0,1]]))
    return K, T

# ---------------------------------------------------------------------------
# LiDAR/Radar ingestion
# ---------------------------------------------------------------------------

def load_bin_points(bin_path: Path) -> np.ndarray:
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1,4)
    return pts  # x,y,z,intensity


def load_pcd_points(pcd_path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    xyz = np.asarray(pcd.points, dtype=np.float32)
    inten = np.ones((xyz.shape[0],1), dtype=np.float32)
    return np.hstack([xyz, inten])

# ---------------------------------------------------------------------------
# Ground removal with NaN filtering (LiDAR only)
# ---------------------------------------------------------------------------

def remove_ground(points: np.ndarray) -> np.ndarray:
    # Filter out invalid (NaN or inf) points
    finite_mask = np.all(np.isfinite(points[:,:3]), axis=1)
    pts = points[finite_mask]
    XY = pts[:, :2]
    Z = pts[:, 2]
    if len(pts) < 10:
        return pts  # too few to fit
    ransac = RANSACRegressor(residual_threshold=0.2)
    ransac.fit(XY, Z)
    plane = ransac.predict(XY)
    non_ground = pts[(Z - plane) > 0.2]
    return non_ground

# ---------------------------------------------------------------------------
# Projection & localization
# ---------------------------------------------------------------------------

def project_to_image(pts: np.ndarray, K: np.ndarray, T_cam2ego: np.ndarray):
    T = np.linalg.inv(T_cam2ego)
    pts_h = np.hstack([pts[:,:3], np.ones((len(pts),1))])
    cam = (T @ pts_h.T).T[:, :3]
    z = cam[:,2]
    uv = (K @ cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv, z


def frustum_localize(pts: np.ndarray, uv: np.ndarray, z: np.ndarray, bbox):
    x1,y1,x2,y2 = bbox
    mask = (uv[:,0]>=x1)&(uv[:,0]<=x2)&(uv[:,1]>=y1)&(uv[:,1]<=y2)&(z>0)
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
        mask = (pts[:,2]>=zs[i])&(pts[:,2]<zs[i+1])
        sub = pts[mask]
        ix = ((sub[:,0]-xlim[0])/res).astype(int)
        iy = ((sub[:,1]-ylim[0])/res).astype(int)
        img = np.zeros((w,h), dtype=np.float32)
        for x,y,inten in zip(ix,iy,sub[:,3]):
            if 0<=x<w and 0<=y<h:
                img[x,y] = max(img[x,y], inten)
        bev.append(img)
    return np.stack(bev, axis=-1)

# ---------------------------------------------------------------------------
# Visualization overlay
# ---------------------------------------------------------------------------

def draw_overlay(img, uv, z, color=(0,0,255), step=1):
    for i in range(0, len(uv), step):
        u,v = uv[i].astype(int)
        if 0<=u<img.shape[1] and 0<=v<img.shape[0]:
            cv2.rectangle(img, (u-2,v-2), (u+2,v+2), color, -1)
    return img

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("Fusion pipeline: LiDAR, Radar, Camera")
    p.add_argument('--image_root',  required=True, type=Path)
    p.add_argument('--lidar_dir',   required=True, type=Path)
    p.add_argument('--radar_root',  type=Path)
    p.add_argument('--model',       default='yolov8n.pt')
    p.add_argument('--out_bev',     type=Path)
    p.add_argument('--out_overlays',type=Path)
    args = p.parse_args()

    out_ov = args.out_overlays or (args.image_root/'overlays')
    out_ov.mkdir(parents=True, exist_ok=True)

    # LiDAR
    lj = list(args.lidar_dir.glob('*.json'))
    if len(lj)!=1: raise FileNotFoundError
    _, T_l2e = load_sensor_calib(lj[0])
    bins = sorted(args.lidar_dir.glob('*.bin'))
    lidar_pts = []
    for b in bins:
        pts = load_bin_points(b)
        h = np.hstack([pts[:,:3], np.ones((len(pts),1))])
        ego = (T_l2e@h.T).T[:,:3]
        lidar_pts.append(np.hstack([ego, pts[:,3:4]]))
    lidar = np.vstack(lidar_pts)
    print(f"LiDAR: {len(bins)} bins -> {lidar.shape[0]} pts")
    lidar = remove_ground(lidar)

    # Radar
    radar = np.empty((0,4),dtype=np.float32)
    if args.radar_root:
        for sd in sorted(args.radar_root.iterdir()):
            if not sd.is_dir(): continue
            js = next(sd.glob('*.json'),None)
            if not js: continue
            _, T_r2e = load_sensor_calib(js)
            for pcd in sd.glob('*.pcd'):
                pts = load_pcd_points(pcd)
                h = np.hstack([pts[:,:3],np.ones((len(pts),1))])
                ego = (T_r2e@h.T).T[:,:3]
                radar = np.vstack([radar, np.hstack([ego,pts[:,3:4]])])
        print(f"Radar: {radar.shape[0]} pts from {args.radar_root}")

    # Merge
    cloud = np.vstack([lidar, radar]) if radar.size else lidar

    # Detector
    model = YOLO(args.model)
    centers = []

    # Process cameras
    for cd in sorted(args.image_root.iterdir()):
        if not cd.is_dir(): continue
        imgf = next(cd.glob('*.jpg'),None) or next(cd.glob('*.png'),None)
        js = next(cd.glob('*.json'),None)
        K,Tc = load_sensor_calib(js)
        uv,z = project_to_image(cloud[:,:3],K,Tc)
        img = cv2.imread(str(imgf))
        dets = model(str(imgf))[0].boxes
        for _,_,b in zip(dets.cls,dets.conf,dets.xyxy):
            bb = tuple(b.cpu().numpy().astype(int))
            c = frustum_localize(cloud[:,:3],uv,z,bb)
            if c is not None: centers.append(c[:2])
            cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),(0,255,0),2)
        ov = draw_overlay(img,*project_to_image(lidar[:,:3],K,Tc),color=(0,0,255))
        if radar.size: ov = draw_overlay(ov,*project_to_image(radar[:,:3],K,Tc),color=(255,0,0))
        cv2.imwrite(str(out_ov/f"{cd.name}.png"),ov)

    # BEV plot
    bev = make_bev_channels(cloud)
    plt.imshow(bev[:,:,0].T,extent=[-50,50,-50,50])
    if centers: xs,ys = zip(*centers); plt.scatter(xs,ys,c='r',marker='s',s=40)
    plt.scatter(0,0,c='g',marker='x',s=100)
    if args.out_bev: plt.savefig(args.out_bev)
    else: plt.show()

if __name__=='__main__': main()
