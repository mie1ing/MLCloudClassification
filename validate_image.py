#!/usr/bin/env python3
"""Validate images for training or inference.

This script checks whether given image files are suitable for use as
training data or inference input for the cloud classification project.
It performs the same integrity and extension checks used during model
training and inference.

Usage:
    python validate_image.py <image1> [<image2> ...]

An "OK" result indicates the image passed all checks.
"""
import os
import sys
from typing import Iterable

from PIL import Image

# Reuse the integrity check from the training script
try:
    from train import is_image_ok
except Exception:  # pragma: no cover - fallback if train isn't available
    def is_image_ok(path: str) -> bool:
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

def check_image(path: str) -> bool:
    """Return True if the image passes all validation checks."""
    if not os.path.isfile(path):
        print(f"{path}: not a file")
        return False
    if not path.lower().endswith(VALID_EXTS):
        print(f"{path}: unsupported file extension")
        return False
    if not is_image_ok(path):
        print(f"{path}: corrupted or unreadable image")
        return False
    try:
        # Ensure the image can be fully loaded and converted to RGB
        Image.open(path).convert("RGB")
    except Exception as exc:
        print(f"{path}: failed to load ({exc})")
        return False
    return True

def main(paths: Iterable[str]) -> None:
    for p in paths:
        status = "OK" if check_image(p) else "INVALID"
        print(f"{p}\t{status}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
