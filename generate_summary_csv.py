import pandas as pd
from pathlib import Path

# Location of detection results
runs_dir = Path("runs/detect")
summary_rows = []

# Loop through each exp (video) folder
for exp_dir in runs_dir.glob("*"):
    csv_file = exp_dir / "detections.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            summary_rows.append({
                "Video Name": row["Image Name"],
                "Label": row["Label"],
                "Confidence": row["Confidence"]
            })
# Extract store and date from file path if possible
store = exp_dir.name.split("_")[0]  # e.g., store123_Shoplifting001
video_name = row["Image Name"]
date = "2025-05-09"  # Or dynamically extract from your folder structure if needed

summary_rows.append({
    "Store": store,
    "Date": date,
    "Video Name": video_name,
    "Label": row["Label"],
    "Confidence": row["Confidence"]
})

# Combine and save
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("detection_summary.csv", index=False)
print(f"\n✅ Saved summary to detection_summary.csv with {len(summary_df)} rows")
