import os
import subprocess
import boto3
from pathlib import Path
import pandas as pd

# ---------- CONFIGURATION ----------
bucket_name = "retail-theft-detection-data"
store_id = "store123"
date = "2025-05-09"

s3_prefix = f"raw-footage/{store_id}/{date}/"
local_video_dir = Path("input_videos")
local_video_dir.mkdir(exist_ok=True)

processed_prefix = f"processed/{store_id}/{date}/"
weights = "yolov5s.pt"
project_dir = "runs/detect"
data_yaml = "data/coco128.yaml"
imgsz = "640"
conf_thres = "0.10"
iou_thres = "0.45"

alert_trigger_count = 3  # Minimum times a label must appear to trigger alert
alert_labels = {"suitcase", "backpack", "person"}
sns_topic_arn = "arn:aws:sns:us-west-2:336162657241:retail-detection-alerts"

s3 = boto3.client("s3")
sns = boto3.client("sns")

# ---------- DOWNLOAD .mp4 FILES ----------
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix)
videos = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".mp4")]

print(f"\n📥 Found {len(videos)} video(s) in s3://{bucket_name}/{s3_prefix}")
for key in videos:
    filename = Path(key).name
    local_path = local_video_dir / filename
    print(f"⬇️ Downloading: {filename}")
    s3.download_file(bucket_name, key, str(local_path))

# ---------- RUN DETECTION ----------
for video in local_video_dir.glob("*.mp4"):
    video_name = video.stem.replace(" ", "_")
    print(f"\n🔍 Detecting: {video.name}")

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
    subprocess.run(cmd)

    # ---------- UPLOAD RESULTS ----------
    output_dir = Path(project_dir) / video_name
    for fname in ["detections.csv", video.name]:
        local_file = output_dir / fname
        if local_file.exists():
            s3_key = f"{processed_prefix}{video_name}/{fname}"
            print(f"⬆️ Uploading: {fname} → s3://{bucket_name}/{s3_key}")
            s3.upload_file(str(local_file), bucket_name, s3_key)

    # ---------- ALERT ON REPEATED DETECTIONS ----------
    detections_csv = output_dir / "detections.csv"
    if detections_csv.exists():
        df = pd.read_csv(detections_csv)
        label_counts = df["Label"].value_counts()

        for label, count in label_counts.items():
            if label in alert_labels and count >= alert_trigger_count:
                message = f"⚠️ Alert: {label} detected {count} times in {video.name}"
                print(f"📣 Sending alert: {message}")
                sns.publish(
                    TopicArn=sns_topic_arn,
                    Message=message,
                    Subject="Retail Theft Detection Alert"
                )

print("\n✅ Detection and alerting complete.")
