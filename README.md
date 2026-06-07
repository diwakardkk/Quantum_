# UQ-HyQNet

**UQ-HyQNet: An Uncertainty-Aware Hybrid Quantum-Classical Network for Trustworthy Brain Tumor MRI Triage**

This repository contains a beginner-friendly PyTorch + PennyLane research codebase for
brain tumor MRI classification with uncertainty estimation, calibration, rejection-based
triage, and Grad-CAM explainability.

The hybrid quantum-classical model uses the PennyLane `default.qubit` simulator. It does
not require access to real quantum hardware and does not claim quantum advantage.

## Models

- `classical_cnn`: small CNN trained from scratch
- `mobilenet_mlp`: MobileNetV2 feature extractor + classical MLP classifier
- `hybrid_quantum`: MobileNetV2 feature extractor + PennyLane variational quantum circuit

## Dataset

The code auto-detects common folder layouts:

```text
data/train/<class>/*.jpg
data/val/<class>/*.jpg
data/test/<class>/*.jpg
```

or:

```text
data/<class>/*.jpg
dataset/<class>/*.jpg
Brain_Tumor_MRI_Dataset/<split>/<class>/*.jpg
```

Images are loaded lazily with PyTorch datasets. They are not loaded fully into memory.

## Install

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For NVIDIA GPU machines, install PyTorch from the official CUDA command first:

https://pytorch.org/get-started/locally/

Then install the remaining packages:

```bash
pip install pennylane scikit-learn matplotlib pandas numpy pillow opencv-python tqdm pyyaml torchcam captum
```

## Check Environment

```bash
python -m src.check_env
```

## Smoke Test

```bash
python scripts/smoke_test.py
```

## One-Command 200-Epoch GPU-Ready Experiment

This trains and evaluates all three models sequentially using:

- `epochs: 200`
- `n_qubits: 6`
- `n_quantum_layers: 3`
- `unfreeze_last_n_blocks: 5`
- `device: auto`

Run:

```bash
python scripts/run_all_models.py --config config_200epoch_gpu.yaml
```

If Grad-CAM evaluation is too slow:

```bash
python scripts/run_all_models.py --config config_200epoch_gpu.yaml --no-gradcam
```

To run only one model:

```bash
python scripts/run_all_models.py --config config_200epoch_gpu.yaml --models hybrid_quantum
```

## Outputs

Results are saved in timestamped folders:

```text
outputs_200epoch_gpu/YYYYMMDD_HHMMSS_modelname/
```

Each run contains:

- `checkpoints/best_model.pt`
- `checkpoints/last_model.pt`
- `metrics.json`
- `metrics.csv`
- `predictions.csv`
- `class_names.json`
- `config_used.yaml`
- `model_summary.txt`
- `plots/confusion_matrix.png`
- `plots/roc_curve.png`
- `plots/reliability_diagram.png`
- `plots/uncertainty_histogram.png`
- `plots/rejection_curve.png`
- `gradcam/`

## GitHub Notes

The `.gitignore` excludes datasets, virtual environments, model checkpoints, and output
folders. Push source code, configs, and documentation. Do not push large MRI datasets or
trained checkpoints unless you intentionally use Git LFS.

## Important Research Note

This code supports experiments. It does not by itself prove clinical validity or quantum
advantage. For a paper, report all baselines honestly, run multiple seeds, check data
leakage carefully, and validate on an external dataset when possible.
