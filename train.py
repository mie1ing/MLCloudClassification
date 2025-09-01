# python
# train.py
import os
import json
import random
from typing import Dict, Tuple
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
from PIL import Image, ImageFile
from tqdm.auto import tqdm

from config import (
    DATA_DIR, CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE,
    VAL_SPLIT, BATCH_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY,
    NUM_WORKERS, SEED, MODEL_NAME, IMAGENET_MEAN, IMAGENET_STD,
    BEST_WEIGHTS_PATH
)

# Allow decoding truncated images to avoid hard failures on mildly corrupted files
ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # Allow non-deterministic algorithms on CPU for better performance


def make_transforms(image_size: Tuple[int, int]):
    h, w = image_size
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size=(h, w), scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, val_tf


def build_target_mapping(found_classes, desired_classes) -> Dict[int, int]:
    # found_classes: class-name list from ImageFolder, sorted alphabetically
    # desired_classes: class-name order defined in config (we want model outputs to align with this order)
    if set(found_classes) != set(desired_classes):
        missing = set(desired_classes) - set(found_classes)
        extra = set(found_classes) - set(desired_classes)
        raise ValueError(
            f"Dataset classes mismatch config. Missing: {sorted(missing)}; Extra: {sorted(extra)}"
        )
    # old_idx -> new_idx
    name_to_new = {name: i for i, name in enumerate(desired_classes)}
    old_to_new = {old_idx: name_to_new[name] for old_idx, name in enumerate(found_classes)}
    return old_to_new


def get_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model


class TargetMapper:
    """
    Top-level callable class that can be safely pickled by DataLoader multiprocess workers.
    Maps ImageFolder's original class indices to the index order specified in the config.
    """
    def __init__(self, mapping: Dict[int, int]):
        # Keep a copy to avoid accidental external modifications
        self.mapping = dict(mapping)

    def __call__(self, y: int) -> int:
        return self.mapping[y]


def is_image_ok(path: str) -> bool:
    """
    For ImageFolder's is_valid_file: filter out bad images when building the sample list.
    Performs a quick integrity check without actually decoding pixel data.
    """
    try:
        with Image.open(path) as img:
            img.verify()  # Fast structural validation of the file
        return True
    except Exception:
        return False


def safe_pil_loader(path: str) -> Image.Image:
    """
    More robust PIL loader: always convert to RGB; with LOAD_TRUNCATED_IMAGES it tolerates slight truncation.
    """
    with Image.open(path) as img:
        return img.convert("RGB")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in tqdm(loader, desc="Evaluating", leave=False):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main():
    set_seed(SEED)

    device = torch.device("cpu")  # Intel macOS，使用 CPU
    torch.set_num_threads(os.cpu_count() or 4)

    # Base dataset (used to determine classes and indices)
    base = datasets.ImageFolder(root=DATA_DIR)
    old_to_new = build_target_mapping(base.classes, CLASS_NAMES)

    # Define transforms and target index mapping (avoid defining lambda inside main; use top-level picklable objects instead)
    train_tf, val_tf = make_transforms(IMAGE_SIZE)
    target_transform = TargetMapper(old_to_new)

    # Use a robust loader and pre-filter corrupted images
    full_train = datasets.ImageFolder(
        root=DATA_DIR,
        transform=train_tf,
        target_transform=target_transform,
        loader=safe_pil_loader,
        is_valid_file=is_image_ok,
    )
    full_val = datasets.ImageFolder(
        root=DATA_DIR,
        transform=val_tf,
        target_transform=target_transform,
        loader=safe_pil_loader,
        is_valid_file=is_image_ok,
    )

    # Record filtering statistics (to help troubleshoot data issues)
    total_raw = len(getattr(base, "samples", []))
    total_valid = len(full_train)
    if total_raw and total_valid < total_raw:
        print(f"Detected and filtered corrupted/unreadable images: {total_raw - total_valid} / {total_raw}")

    n_total = len(full_train)
    if n_total == 0:
        raise RuntimeError(f"No usable samples found in {DATA_DIR}. "
                           f"Please check the directory structure and image integrity.")
    n_val = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val

    gen = torch.Generator().manual_seed(SEED)
    # Split train/validation within the set of valid samples
    train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=gen)
    train_ds = Subset(full_train, train_idx.indices if hasattr(train_idx, 'indices') else train_idx)
    val_ds = Subset(full_val, val_idx.indices if hasattr(val_idx, 'indices') else val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False
    )

    model = get_model(MODEL_NAME, NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total, running_loss, running_correct = 0, 0.0, 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            running_correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = running_correct / max(total, 1)

        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), BEST_WEIGHTS_PATH)
            with open("class_names.json", "w", encoding="utf-8") as f:
                json.dump(CLASS_NAMES, f, ensure_ascii=False, indent=2)
            print(f"  -> Saved best weights to {BEST_WEIGHTS_PATH} (Val Acc={best_acc:.4f})")

    print(f"Training completed. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    time_start = time.time()
    main()
    time_end = time.time()
    elapsed = time_end - time_start
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"Total time: {minutes:02d}:{seconds:02d}")
