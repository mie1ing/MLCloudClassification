# python
# config.py
from typing import List, Tuple

# 数据与类别
DATA_DIR: str = "training_data/"
CLASS_NAMES: List[str] = [
    "altocumulus", "altostratus", "cirrocumulus",
    "cirrostratus", "cirrus", "cumulonimbus", "cumulus",
    "nimbostratus", "stratocumulus", "stratus"
]
NUM_CLASSES: int = len(CLASS_NAMES)

# 输入尺寸
IMAGE_SIZE: Tuple[int, int] = (128, 128)  # (H, W)

# 训练超参
VAL_SPLIT: float = 0.2
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 20
LR: float = 3e-4
WEIGHT_DECAY: float = 1e-4
NUM_WORKERS: int = 4
SEED: int = 42

# 模型
MODEL_NAME: str = "mobilenet_v3_small"

# 归一化（使用 ImageNet 预训练权重的均值/方差）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# 最佳权重保存路径
BEST_WEIGHTS_PATH: str = "cloud_classifier_pt_best.pt"