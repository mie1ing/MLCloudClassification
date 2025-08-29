# python
# config.py
from typing import List, Tuple

# Data and classes
DATA_DIR: str = "training_data/"
CLASS_NAMES: List[str] = [
    "Altocumulus", "Altostratus", "Cirrocumulus",
    "Cirrostratus", "Cirrus", "Indistinguishable", "Cumulus",
    "Clear sky", "Stratocumulus", "Stratus"
]
NUM_CLASSES: int = len(CLASS_NAMES)

# Input size
IMAGE_SIZE: Tuple[int, int] = (128, 128)  # (H, W)

# Training hyperparameters
VAL_SPLIT: float = 0.2
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 20
LR: float = 3e-4
WEIGHT_DECAY: float = 1e-4
NUM_WORKERS: int = 4
SEED: int = 42

# Model
MODEL_NAME: str = "mobilenet_v3_small"

# Normalization (using ImageNet pretrained weights mean/std)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Best weights save path
BEST_WEIGHTS_PATH: str = "cloud_classifier_pt_best.pt"