# python
# infer.py
import os
import csv
from typing import Tuple, List
# from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision import transforms, models

from config import (
    CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, MODEL_NAME,
    IMAGENET_MEAN, IMAGENET_STD, BEST_WEIGHTS_PATH
)

# Allow decoding truncated images to avoid hard failures on mildly corrupted files
ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_image_ok(path: str) -> bool:
    """Quick integrity check without fully decoding image data."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def safe_pil_loader(path: str) -> Image.Image:
    """Robust PIL loader that always converts to RGB."""
    with Image.open(path) as img:
        return img.convert("RGB")

def make_val_transform():
    h, w = IMAGE_SIZE
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_model(name: str, num_classes: int):
    name = name.lower()
    if name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    elif name == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupport model: {name}")
    return model

@torch.no_grad()
def predict_image(path: str) -> Tuple[str, float, List[float]]:
    device = torch.device("cpu")
    tfm = make_val_transform()

    if not is_image_ok(path):
        raise ValueError(f"Corrupted or unreadable image: {path}")
    img = safe_pil_loader(path)
    x = tfm(img).unsqueeze(0).to(device)  # [1, C, H, W]

    model = get_model(MODEL_NAME, NUM_CLASSES)
    model.load_state_dict(torch.load(BEST_WEIGHTS_PATH, map_location=device))
    model.eval()

    logits = model(x)  # [1, num_classes]
    probs = F.softmax(logits, dim=-1)[0].cpu().tolist()

    top_idx = int(torch.tensor(probs).argmax().item())
    top_class = CLASS_NAMES[top_idx] if 0 <= top_idx < len(CLASS_NAMES) else str(top_idx)
    top_prob = float(probs[top_idx])
    return top_class, top_prob, probs

def predict_folder(folder: str, csv_path: str = None):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    results = []
    total = 0
    for fn in os.listdir(folder):
    # for fn in tqdm(os.listdir(folder)):
        if fn.lower().endswith(valid_exts):
            total += 1
            p = os.path.join(folder, fn)
            if not is_image_ok(p):
                print(f"Skipping corrupted or unreadable image: {fn}")
                continue
            cls, score, _ = predict_image(p)
            results.append((fn, cls, score))
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "prediction", "score"])
            writer.writerows(results)
    if total and len(results) < total:
        print(f"Filtered out {total - len(results)} invalid images out of {total}")
    return results

if __name__ == "__main__":
    # Example: single image
    # cls, score, probs = predict_image("atmo2_20250826_111547.jpg")
    # print(f"Predicted class: {cls}, Confidence: {score:.4f}")
    # print(probs)
    # Example: batch over a directory
    # for name, cls, score in predict_folder("testing_data", csv_path="testing_data/test_results.csv"):
    #     print(f"{name}\t{cls}\t{score:.4f}")
    predict_folder("testing_data", csv_path="testing_data/test_results.csv")