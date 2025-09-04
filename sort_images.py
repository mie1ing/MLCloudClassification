import os
import random
import shutil
import pandas as pd
from config import SEED

# Config: recursively scan all subdirectories under top_dir; only process directories that have exactly one .csv
top_dir = "unsort_images"           # Top-level directory containing unclassified images
train_root = "new_class_test"     # Destination directory for training images
val_root = "validation_data"     # Destination directory for validation images
move_files = False               # True=move, False=copy from source to training_data
overwrite = False                # True=overwrite existing targets; False=skip
split_val = False                 # True=split 10% of images for validation before training

os.makedirs(train_root, exist_ok=True)
os.makedirs(val_root, exist_ok=True)

stats_by_src = {}
total_classified = 0

for dirpath, _, filenames in os.walk(top_dir):
    csvs = [f for f in filenames if f.lower().endswith(".csv")]
    if len(csvs) != 1:
        continue  # Only process directories that meet the condition
    csv_path = os.path.join(dirpath, csvs[0])
    try:
        # First column=filename, second column=class
        df = pd.read_csv(csv_path, header=0).dropna(subset=['1'])
    except Exception as e:
        print(f"Failed to read: {csv_path}: {e}")
        continue
    for fname, cls in df.values:
        src = os.path.join(dirpath, str(fname))
        if not os.path.exists(src):
            print(f"Missing: {src}")
            continue
        dst_dir = os.path.join(train_root, cls)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(str(fname)))
        if os.path.exists(dst) and not overwrite:
            continue
        if move_files:
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)
        rel_dir = os.path.relpath(dirpath, top_dir)
        stats_by_src[rel_dir] = stats_by_src.get(rel_dir, 0) + 1
        total_classified += 1

# After initial classification, split validation set if split=true
if split_val:
    random.seed(SEED)
    val_records = []
    for cls in os.listdir(train_root):
        cls_dir = os.path.join(train_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir)
                if os.path.isfile(os.path.join(cls_dir, f))]
        if not files:
            continue
        n_val = max(1, int(len(files) * 0.1))
        val_files = random.sample(files, n_val)
        dst_dir = val_root
        os.makedirs(dst_dir, exist_ok=True)
        for f in val_files:
            src = os.path.join(cls_dir, f)
            dst = os.path.join(dst_dir, f)
            shutil.move(src, dst)
            val_records.append((f, cls))

    if val_records:
        pd.DataFrame(val_records, columns=["filename", "class"]).to_csv(
            "val_data_class.csv", index=False
        )

# Print statistics
print("\nStatistics:")
for folder, count in sorted(stats_by_src.items()):
    print(f"{folder}: {count} images classified")
print(f"Total images classified: {total_classified}")

