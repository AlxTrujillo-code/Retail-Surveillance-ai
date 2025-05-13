import os
import subprocess
from pathlib import Path
import csv

# ✅ Absolute path to your videos
video_folder = "/home/alxtrujillo/datasets/DCSASS/DCSASS Dataset/Shoplifting"
weights = "yolov5s.pt"
project_dir = "runs/detect"
data_yaml = "data/coco128.yaml"
imgsz = "640"
conf_thres = "0.10"  # Lowered to ensure more detections
iou_thres = "0.45"

# Detection summary
summary = []

# Find .mp4 videos
video_dir = Path(video_folder)
videos = list(video_dir.glob("*.mp4"))
print(f"\n📂 Found {len(videos)} video(s) in: {video_dir.resolve()}")
for v in videos:
    print("  •", v.name)

# Run detection on each video
for video in videos:
    video_name = video.stem.replace(" ", "_")  # Remove spaces from name
    print(f"\n🎬 Processing: {video.name}")

    cmd = [
        "python3", "detect.py",
        "--weights", weights,
        "--source", str(video),
        "--data", data_yaml,
        "--imgsz", imgsz,
        "--conf-thres", conf_thres,
        "--iou-thres", iou_thres,
        "--save-csv",
        "--project", project_dir,
        "--name", video_name,
        "--exist-ok"
    ]

    print("🚀 Running command:", " ".join(cmd))
    subprocess.run(cmd)

    # Count detections in CSV
    csv_path = Path(project_dir) / video_name / "detections.csv"
    if csv_path.exists():
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            count = sum(1 for _ in reader)
            print(f"✅ Detections in {csv_path.name}: {count}")
            summary.append((video.name, count))
    else:
        print(f"❌ No detections.csv found for {video.name}")
        summary.append((video.name, 0))

# Print final summary
print("\n📊 Final Detection Summary:")
print(f"{'Video':<35} | {'Detections':>10}")
print("-" * 50)
for video_name, count in summary:
    print(f"{video_name:<35} | {count:>10}")
