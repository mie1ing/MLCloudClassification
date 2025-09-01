import argparse
import importlib
import time

import config


def time_single_epoch(image_size: int, weights_file: str) -> float:
    """Run one epoch of training at the given resolution and return seconds."""
    original_size = config.IMAGE_SIZE
    original_epochs = config.NUM_EPOCHS
    original_path = config.BEST_WEIGHTS_PATH
    config.IMAGE_SIZE = (image_size, image_size)
    config.NUM_EPOCHS = 1
    config.BEST_WEIGHTS_PATH = weights_file
    try:
        import train  # local import so reload reflects patched config
        importlib.reload(train)
        start = time.time()
        train.main()
        return time.time() - start
    finally:
        config.IMAGE_SIZE = original_size
        config.NUM_EPOCHS = original_epochs
        config.BEST_WEIGHTS_PATH = original_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate training time across image resolutions by timing a single epoch."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[config.IMAGE_SIZE[0]],
        help="List of square image sizes to evaluate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.NUM_EPOCHS,
        help="Total epochs to scale from the measured single epoch.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="est_time.pt",
        help="Where to save temporary weights for each timing run.",
    )
    args = parser.parse_args()

    for i, size in enumerate(args.sizes, start=1):
        try:
            elapsed = time_single_epoch(size, args.file)
        except Exception as exc:
            print(f"Run {i} (image_size={size}) failed: {exc}")
            continue

        one_m, one_s = divmod(int(elapsed + 0.5), 60)
        total = elapsed * args.epochs
        tot_m, tot_s = divmod(int(total + 0.5), 60)
        print(
            f"Run {i}: image_size={size} -> 1 epoch {one_m:02d}:{one_s:02d}, "
            f"{args.epochs} epochs {tot_m:02d}:{tot_s:02d}"
        )


if __name__ == "__main__":
    main()

