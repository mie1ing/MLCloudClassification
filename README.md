# README
## Project Title
Atmocorder - Cloud classification
## Overview
This repository provides code and environment specifications to reproduce the development and training environment precisely. It supports:
- Portable setup via mlccenvironment.yml
- Fully pinned, exact-reproduction setup via platform-specific .lock files

## Project Structure

```
MLCloudClassification/
├── config.py               # Configuration helpers
├── infer.py                # Run inference on new images
├── sort_images.py          # Organise dataset images
├── train.py                # Training script
├── validate_image.py       # Validate individual images
├── tensorflow/             # Example TensorFlow scripts
│   ├── mltest_tensorflow.py
│   └── modelusetest_tensorflow.py
├── mlccenvironment.yml     # Conda environment specification
└── conda-osx-64.lock       # Locked environment for macOS x86_64
```

## System Requirements
- OS/Arch: macOS (x86_64) or compatible
- Python: 3.13.2
- Package manager: conda

If you are on a different OS/architecture, prefer the mlccenvironment.yml method. The .lock file is platform-specific.
## Environment Setup
Option A — Portable (recommended for different platforms)

Use the provided `mlccenvironment.yml` file to create and activate a new environment:
``` bash
# Create and activate the environment from mlccenvironment.yml
conda env create -n <env-name> -f mlccenvironment.yml
conda activate <env-name>

# Verify Python version
python --version  # Expect 3.13.2
```
Option B — Exact reproduction (recommended on the same platform as the author)
``` bash
# Create and activate the environment from a lock file
# Replace the file name with the one in this repo (e.g., conda-osx-64.lock)
conda create -n <env-name> --file <your-platform>.lock
conda activate <env-name>

# Verify Python version
python --version  # Expect 3.13.2
```
Upgrade or update is not required for exact reproducibility. If you need to regenerate a lock file on your platform after a successful install, you can do so for team-wide consistency.
## Data Setup
The training code expects images arranged in a folder called `training_data/`,
with one subdirectory per class. The default classes are defined in
[`config.py`](config.py) (`CLASS_NAMES`) and the dataset path is set by
`DATA_DIR`.

Example directory layout:

```
training_data/
├── altocumulus/
│   ├── img001.jpg
│   └── ...
├── altostratus/
│   └── ...
...
└── stratus/
    └── ...
```

### Preparing the dataset
If your raw images are stored elsewhere and labelled via CSV files, you can
use `sort_images.py` to build the structure above:

1. Place all raw image folders under `all_images/`. Each folder must contain a
   single `.csv` file whose first column is the filename and second column is
   the class label.
2. Run the script:

   ```bash
   python sort_images.py
   ```

   Images are copied into `training_data/<class>` by default. Adjust the
   variables at the top of `sort_images.py` if you need to move files or change
   source/destination paths.

If you already have data organised by class, place it directly in
`training_data/` or update `DATA_DIR` in `config.py` to point to its location.

Optional: validate images before training to catch corrupt files:

```bash
python validate_image.py path/to/image.jpg [more_images...]
```
## Reproducibility Notes
- Use a fixed random seed when running experiments.
- Set deterministic hashing for Python:
``` bash
export PYTHONHASHSEED=0
```
- Keep the same hyperparameters, data splits, and configuration across runs when comparing results.

## How to Run
Basic command:
``` bash
conda activate <env-name>
python train.py
```
With a config file (if your workflow uses one):
``` bash
conda activate <env-name>
python train.py --config configs/example.local.yaml
```

## Inference
Run predictions on new images using `infer.py`.

### Single Image
```bash
python - <<'PY'
from infer import predict_image
cls, score, probs = predict_image("path/to/image.jpg")
print(f"Predicted class: {cls}, Confidence: {score:.4f}")
print(probs)
PY
```
Example output:
```
Predicted class: cirrus, Confidence: 0.8732
[0.002, 0.001, 0.8732, ...]
```

### Batch over a Folder
```bash
python - <<'PY'
from infer import predict_folder
for name, cls, score in predict_folder("path/to/images"):
    print(f"{name}\t{cls}\t{score:.4f}")
PY
```
Example output:
```
img1.jpg    cumulus    0.9700
img2.jpg    stratus    0.8800
```
## Troubleshooting
- Verify your environment:
``` bash
python --version
python -c "import sys, platform; print(platform.platform(), platform.machine()); print(sys.version)"
conda list
```
- If you face dependency conflicts on a different OS/arch, switch to Option A (mlccenvironment.yml). Exact lock files are platform-specific and may not resolve on other systems.
- When reporting issues, please include:
    - OS and architecture (e.g., macOS Sonoma, x86_64)
    - Python version
    - The command you ran and the full error message
