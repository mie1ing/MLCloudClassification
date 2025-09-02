import os
import glob
import shutil
import pandas as pd

date = "20250831"
# Specify the folders to be merged
folders = [
    f"unsort_images/Atmo1_{date}",
    f"unsort_images/Atmo2_{date}",
]

out_img_dir = f"unsort_images/{date}"
os.makedirs(out_img_dir, exist_ok=True)

all_csv = []
for sub_path in folders:
    if not os.path.isdir(sub_path):
        continue

    # Find csv files
    csv_files = glob.glob(os.path.join(sub_path, "*.csv"))
    if len(csv_files) != 1:
        continue
    csv_file = csv_files[0]

    df = pd.read_csv(csv_file)
    all_csv.append(df)

    # Copy images
    for img in glob.glob(os.path.join(sub_path, "*")):
        if img.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff")):
            shutil.copy(img, out_img_dir)

# Merge all CSV
if all_csv:
    merged = pd.concat(all_csv, ignore_index=True)
    merged.to_csv(f"unsort_images/{date}/{date}.csv", index=False)
