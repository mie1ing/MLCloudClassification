import os
import shutil
import pandas as pd

# 配置：递归扫描 top_dir 下所有子目录；仅处理“有且仅有一个 .csv”的目录
top_dir = "all_images"           # 顶层目录
output_root = "training_data"    # 统一输出根目录
move_files = False               # True=移动，False=复制
overwrite = False                # True=覆盖已存在目标；False=跳过

def sanitize(name: str) -> str:
    # 清理类别名中不适合做文件夹的字符
    return "".join("_" if c in '/\\:*?"<>|' else c for c in str(name)).strip()

os.makedirs(output_root, exist_ok=True)

for dirpath, _, filenames in os.walk(top_dir):
    csvs = [f for f in filenames if f.lower().endswith(".csv")]
    if len(csvs) != 1:
        continue  # 只处理满足条件的目录
    csv_path = os.path.join(dirpath, csvs[0])
    try:
        df = pd.read_csv(csv_path, header=None)  # 第一列=文件名，第二列=类别
    except Exception as e:
        print(f"读取失败: {csv_path}: {e}")
        continue

    for fname, cls, _ in df.values:
        src = os.path.join(dirpath, str(fname))
        if not os.path.exists(src):
            print(f"缺失: {src}")
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