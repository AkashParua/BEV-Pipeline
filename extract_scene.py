import json
from pathlib import Path
from collections import defaultdict
import os
import logging
# Change this to your dataset directory
BASE_DIR = Path("./data/v1.0-mini")

# Load all JSON files
def load_json(name): return json.load(open(BASE_DIR / f"{name}.json"))

scene_json = load_json("scene")
sample_json = load_json("sample")
sample_data_json = load_json("sample_data")
ego_pose_json = load_json("ego_pose")
calibrated_sensor_json = load_json("calibrated_sensor")
sensor_json = load_json("sensor")
attribute_json = load_json("attribute")
category_json = load_json("category")

# Build lookup dictionaries
sample_by_token = {s["token"]: s for s in sample_json}
ego_by_token = {e["token"]: e for e in ego_pose_json}
sensor_by_token = {s["token"]: s for s in sensor_json}
calib_by_token = {c["token"]: c for c in calibrated_sensor_json}
sample_data_by_sample = defaultdict(list)
for sd in sample_data_json:
    sample_data_by_sample[sd["sample_token"]].append(sd)



attribute_by_token = {a["token"]: a for a in attribute_json}
category_by_token = {c["token"]: c for c in category_json}

# ---- MAIN FUNCTION ----
def extract_scene_info(scene_name: str):
    scene = next((s for s in scene_json if s["name"] == scene_name), None)
    logging.info(f"Extracting scene {scene_name} found {len(scene)} samples in scene.json")

    if not scene:
        raise ValueError(f"Scene {scene_name} not found")

    token = scene["first_sample_token"]
    all_info = []

    while token:
        sample = sample_by_token[token]
        sample_entry = {"cameras": [], "lidars": [], "radars": []}

        for sd in sample_data_by_sample[sample["token"]]:
            calib = calib_by_token[sd["calibrated_sensor_token"]]
            sensor = sensor_by_token[calib["sensor_token"]]
            ego_pose = ego_by_token[sd["ego_pose_token"]]
            modality = sensor["modality"]

            entry = {
                "channel": sensor["channel"],
                "filename": sd["filename"],
                "timestamp": sd["timestamp"],
                "intrinsics": calib.get("camera_intrinsic", []),
                "extrinsics": {
                    "translation": calib["translation"],
                    "rotation": calib["rotation"]
                },
                "ego_pose": {
                    "translation": ego_pose["translation"],
                    "rotation": ego_pose["rotation"]
                }
            }

            # Attach annotations if it's a keyframe image
            if modality == "camera" and sd["is_key_frame"]:
                sample_entry["cameras"].append(entry)

            elif modality == "lidar":
                sample_entry["lidars"].append(entry)
            
            elif modality == "radar":
                sample_entry["radars"].append(entry)

        all_info.append(sample_entry)
        token = sample.get("next")

    return all_info


def save_scene_data(scene_id: str):
    scene_data = extract_scene_info(scene_id)
    os.makedirs(f"{scene_id}", exist_ok=True)
    json.dump(scene_data, open(f"{scene_id}/{scene_id}.json", "w"), indent=4)

    logging.info(f"Saved scene {scene_id} to {scene_id}/{scene_id}.json")
    return scene_data

import json
import os
import shutil
from pathlib import Path

def aggregate_files(scene_id: str):
    """
    Organizes scene data into a structured directory format.
    Creates folders for each frame with separate subdirectories for cameras, lidars, and radars.
    """
    root_dir = Path("data")
    scene_data = save_scene_data(scene_id)
    
    # Create the main scene directory
    scene_dir = Path(scene_id)
    scene_dir.mkdir(exist_ok=True)
    
    for i, sample in enumerate(scene_data):
        frame_dir = scene_dir / f"frame_{i}"
        
        # Create frame subdirectories
        cameras_dir = frame_dir / "cameras"
        lidars_dir = frame_dir / "lidars" 
        radars_dir = frame_dir / "radars"
        
        cameras_dir.mkdir(parents=True, exist_ok=True)
        lidars_dir.mkdir(parents=True, exist_ok=True)
        radars_dir.mkdir(parents=True, exist_ok=True)

        # Process cameras
        for camera in sample["cameras"]:
            channel = camera["channel"]
            filename = camera["filename"]
            # if "sweep" in filename:
            #     continue
            
            # Create channel directory
            channel_dir = cameras_dir / channel
            channel_dir.mkdir(exist_ok=True)
            
            # Prepare calibration data
            calibration_data = {
                "extrinsics": camera["extrinsics"],
                "intrinsics": camera["intrinsics"],
                "ego_pose": camera["ego_pose"],
                "timestamp": camera["timestamp"]
            }
            
            # Copy file and save calibration data
            source_file = root_dir / filename
            dest_file = channel_dir / Path(filename).name
            
            if source_file.exists():
                shutil.copy2(source_file, dest_file)
            else:
                print(f"Warning: Source file {source_file} does not exist")
            
            # Save calibration data
            calibration_file = channel_dir / "calibration_data.json"
            with open(calibration_file, "w") as f:
                json.dump(calibration_data, f, indent=4)
        
        # Process lidars
        for lidar in sample["lidars"]:
            channel = lidar["channel"]
            filename = lidar["filename"]
            # if "sweep" in filename:
            #     continue
            
            # Create channel directory
            channel_dir = lidars_dir / channel
            channel_dir.mkdir(exist_ok=True)
            
            # Prepare calibration data
            calibration_data = {
                "extrinsics": lidar["extrinsics"],
                "intrinsics": lidar["intrinsics"],
                "ego_pose": lidar["ego_pose"],
                "timestamp": lidar["timestamp"]
            }
            
            # Copy file and save calibration data
            source_file = root_dir / filename
            dest_file = channel_dir / Path(filename).name
            
            if source_file.exists():
                shutil.copy2(source_file, dest_file)
            else:
                print(f"Warning: Source file {source_file} does not exist")
            
            # Save calibration data
            calibration_file = channel_dir / "calibration_data.json"
            with open(calibration_file, "w") as f:
                json.dump(calibration_data, f, indent=4)
        
        # Process radars
        for radar in sample["radars"]:
            channel = radar["channel"]
            filename = radar["filename"]
            # if "sweep" in filename:
            #     continue
            
            # Create channel directory
            channel_dir = radars_dir / channel
            channel_dir.mkdir(exist_ok=True)
            
            # Prepare calibration data
            calibration_data = {
                "extrinsics": radar["extrinsics"],
                "intrinsics": radar["intrinsics"],
                "ego_pose": radar["ego_pose"],
                "timestamp": radar["timestamp"]
            }
            
            # Copy file and save calibration data
            source_file = root_dir / filename
            dest_file = channel_dir / Path(filename).name
            
            if source_file.exists():
                shutil.copy2(source_file, dest_file)
            else:
                print(f"Warning: Source file {source_file} does not exist")
            
            # Save calibration data
            calibration_file = channel_dir / "calibration_data.json"
            with open(calibration_file, "w") as f:
                json.dump(calibration_data, f, indent=4)

aggregate_files("scene-1077")