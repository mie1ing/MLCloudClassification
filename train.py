# python
# train.py
import os
import json
import random
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
from PIL import Image, ImageFile

from config import (
    DATA_DIR, CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE,
    VAL_SPLIT, BATCH_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY,
    NUM_WORKERS, SEED, MODEL_NAME, IMAGENET_MEAN, IMAGENET_STD,
    BEST_WEIGHTS_PATH
)

# 允许加载被截断但仍可解码的图像，避免轻微损坏直接抛错
ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # CPU 训练，保持较好性能


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
    # found_classes: 来自 ImageFolder 的按字母序类名列表
    # desired_classes: 我们在 config 中定义的类名顺序（希望训练输出与之对齐）
    if set(found_classes) != set(desired_classes):
        missing = set(desired_classes) - set(found_classes)
        extra = set(found_classes) - set(desired_classes)
        raise ValueError(
            f"数据集类别与配置不一致。缺少: {sorted(missing)}; 多余: {sorted(extra)}"
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
        raise ValueError(f"不支持的模型: {name}")
    return model


class TargetMapper:
    """
    顶层定义的可调用类，支持被 DataLoader 的多进程 worker 安全地 pickle。
    将 ImageFolder 的原始类别索引映射到配置中指定的类别顺序索引。
    """
    def __init__(self, mapping: Dict[int, int]):
        # 存一份副本，避免外部意外修改
        self.mapping = dict(mapping)

    def __call__(self, y: int) -> int:
        return self.mapping[y]


def is_image_ok(path: str) -> bool:
    """
    用于 ImageFolder 的 is_valid_file：在构建样本列表时过滤坏图。
    仅做快速完整性校验，不会真正解码像素数据。
    """
    try:
        with Image.open(path) as img:
            img.verify()  # 快速校验文件结构
        return True
    except Exception:
        return False


def safe_pil_loader(path: str) -> Image.Image:
    """
    更健壮的 PIL 加载器：统一转 RGB，配合 LOAD_TRUNCATED_IMAGES 容忍轻微截断。
    """
    with Image.open(path) as img:
        return img.convert("RGB")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
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

    # 基础数据集（用于确定类别与索引）
    base = datasets.ImageFolder(root=DATA_DIR)
    old_to_new = build_target_mapping(base.classes, CLASS_NAMES)

    # 定义 transforms 与目标索引映射（避免在 main 内定义 lambda，改为顶层可pickle对象）
    train_tf, val_tf = make_transforms(IMAGE_SIZE)
    target_transform = TargetMapper(old_to_new)

    # 使用健壮的 loader + 预过滤坏图
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

    # 统计过滤情况（便于排查数据问题）
    total_raw = len(getattr(base, "samples", []))
    total_valid = len(full_train)
    if total_raw and total_valid < total_raw:
        print(f"检测到并已过滤损坏/不可读图像: {total_raw - total_valid} / {total_raw}")

    n_total = len(full_train)
    if n_total == 0:
        raise RuntimeError(f"在 {DATA_DIR} 未找到任何可用样本，请检查目录结构与图片完整性。")
    n_val = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val

    gen = torch.Generator().manual_seed(SEED)
    # 在有效样本范围内划分训练/验证
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

        for x, y in train_loader:
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
            print(f"  -> 保存最佳权重到 {BEST_WEIGHTS_PATH} (Val Acc={best_acc:.4f})")

    print(f"训练完成。最佳验证准确率: {best_acc:.4f}")


if __name__ == "__main__":
    main()