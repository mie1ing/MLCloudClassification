import os
import shutil
import pandas as pd

# Config: recursively scan all subdirectories under top_dir; only process directories that have exactly one .csv
top_dir = "all_images"           # Top-level directory
output_root = "training_data"    # Unified destination root directory
move_files = False               # True=move, False=copy
overwrite = False                # True=overwrite existing targets; False=skip

def sanitize(name: str) -> str:
    # Clean up characters in class names that are invalid for folder names
    return "".join("_" if c in '/\\:*?"<>|' else c for c in str(name)).strip()

os.makedirs(output_root, exist_ok=True)

for dirpath, _, filenames in os.walk(top_dir):
    csvs = [f for f in filenames if f.lower().endswith(".csv")]
    if len(csvs) != 1:
        continue  # Only process directories that meet the condition
    csv_path = os.path.join(dirpath, csvs[0])
    try:
        df = pd.read_csv(csv_path, header=None)  # First column=filename, second column=class
    except Exception as e:
        print(f"Failed to read: {csv_path}: {e}")
        continue
    for fname, cls, _ in df.values:
        src = os.path.join(dirpath, str(fname))
        if not os.path.exists(src):
            print(f"Missing: {src}")
            continue
        dst_dir = os.path.join(output_root, sanitize(cls))
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(str(fname)))
        if os.path.exists(dst) and not overwrite:
            continue
        if move_files:
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)