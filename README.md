# README
## Project Title
Atmocorder - Cloud classification
## Overview
This repository provides code and environment specifications to reproduce the development and training environment precisely. It supports:
- Portable setup via environment.yml
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
- OS/Arch: macOS Sonoma (x86_64) or compatible
- Python: 3.13.2
- Package manager: conda

If you are on a different OS/architecture, prefer the environment.yml method. The .lock file is platform-specific.
## Environment Setup
Option A — Portable (recommended for different platforms)
``` bash
# Create and activate the environment from environment.yml
conda env create -n <env-name> -f environment.yml
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
- Prepare datasets locally and note the paths you will use at runtime.
- Avoid hard-coded absolute paths. Use configuration or CLI flags where applicable.

Example config (optional):
``` yaml
# configs/example.local.yaml
data:
  root: "<your-dataset-path>"
output:
  dir: "./outputs"
train:
  seed: 42
  num_workers: 8
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
## Troubleshooting
- Verify your environment:
``` bash
python --version
python -c "import sys, platform; print(platform.platform(), platform.machine()); print(sys.version)"
conda list
```
- If you face dependency conflicts on a different OS/arch, switch to Option A (environment.yml). Exact lock files are platform-specific and may not resolve on other systems.
- When reporting issues, please include:
    - OS and architecture (e.g., macOS Sonoma, x86_64)
    - Python version
    - The command you ran and the full error message
