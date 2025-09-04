# python
# config.py
from typing import List, Tuple

# Data and classes
DATA_DIR: str = "GCD_class/"
CLASS_NAMES: List[str] = [
    # "Altocumulus", "Stratocumulus", "Cirrocumulus",
    # "Cirrostratus", "Cirrus", "Clear sky", "Cumulus",
    # "Stratocumulus", "Stratus", "Indistinguishable",
    "Altocumulus", "Cirrus",
    "Clear sky", "Cumulus", "Stratocumulus"
]
NUM_CLASSES: int = len(CLASS_NAMES)

# Input size
IMAGE_SIZE: Tuple[int, int] = (224, 224)  # (H, W)

# Training hyperparameters
VAL_SPLIT: float = 0.2
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 20
LR: float = 3e-4
WEIGHT_DECAY: float = 1e-4
NUM_WORKERS: int = 4
SEED: int = 42

# Model
MODEL_NAME: str = "resnet18" # "mobilenet_v3_small", "efficientnet_b0" or "resnet18"

# Normalization (using ImageNet pretrained weights mean/std)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Best weights save path
BEST_WEIGHTS_PATH: str = "resnet18_GCD_class.pt"