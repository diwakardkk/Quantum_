# UQ-HyQNet Run Guide

UQ-HyQNet is a beginner-friendly PyTorch + PennyLane project for brain tumor MRI
classification and uncertainty-aware triage.

This project uses a simulated hybrid quantum-classical model with PennyLane
`default.qubit`. It does not require real quantum hardware and it does not claim
quantum advantage.

## What Is Included

Models:

- `classical_cnn`: small CNN trained from scratch
- `mobilenet_mlp`: MobileNetV2 feature extractor plus classical MLP classifier
- `hybrid_quantum`: MobileNetV2 feature extractor plus PennyLane variational quantum circuit

Evaluation:

- Accuracy, precision, recall, F1
- ROC-AUC when possible
- Confusion matrix
- Brier score
- Expected Calibration Error
- Temperature scaling
- Monte Carlo dropout uncertainty
- Reliability diagram
- Uncertainty histogram
- Rejection curve
- Grad-CAM images

## Step 0: Enter The Project Folder

From your current folder:

```bash
cd UQ-HyQNet
```

## Step 1: Create Virtual Environment

Windows:

```bat
py -3.11 -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

## Step 2: Install Libraries

CPU version:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install PyTorch first using the official command from:

https://pytorch.org/get-started/locally/

Then install the remaining libraries:

```bash
pip install pennylane scikit-learn matplotlib pandas numpy pillow opencv-python tqdm pyyaml torchcam captum
```

## Step 3: Check Environment

```bash
python -m src.check_env
```

Expected ending:

```text
Environment check passed
```

## Step 4: Test The Full Project With Fake Data

This does not use your real MRI dataset.

```bash
python scripts/smoke_test.py
```

Expected ending:

```text
Smoke test passed
```

## Step 5: Prepare Real Dataset

The code supports these common structures:

```text
data/
  train/glioma/
  train/meningioma/
  train/pituitary/
  train/no_tumor/
  val/...
  test/...
```

or:

```text
data/
  glioma/
  meningioma/
  pituitary/
  no_tumor/
```

or:

```text
dataset/
  glioma/
  meningioma/
  pituitary/
  no_tumor/
```

Your existing sibling folder `Brain_Tumor_MRI_Dataset/` is also auto-detected.
It has `Train/` and `Test/`, so the code will create a reproducible validation
split from the training paths without copying images.

Run:

```bash
python scripts/prepare_data.py --config config.yaml
```

## Step 6: Train Classical CNN Baseline

```bash
python scripts/run_classical_baseline.py --config config.yaml
```

## Step 7: Train MobileNet MLP Baseline

```bash
python scripts/run_mlp_baseline.py --config config.yaml
```

## Step 8: Train Proposed Hybrid Quantum Model

```bash
python scripts/run_hybrid_quantum.py --config config.yaml
```

## Optional: One-Command Long GPU-Ready Experiment

To train and evaluate all three models with the 200-epoch GPU-ready config:

```bash
python scripts/run_all_models.py --config config_200epoch_gpu.yaml
```

This config uses:

```yaml
epochs: 200
n_qubits: 6
n_quantum_layers: 3
unfreeze_last_n_blocks: 5
device: auto
```

Results are saved in:

```text
outputs_200epoch_gpu/
```

To run only the hybrid model:

```bash
python scripts/run_all_models.py --config config_200epoch_gpu.yaml --models hybrid_quantum
```

Start with:

```yaml
epochs: 5
n_qubits: 4
n_quantum_layers: 2
```

After confirming the code works, try:

```yaml
epochs: 20
```

Then later try 30 or 50 epochs. Do not use 8 qubits until the 4-qubit model
works on your machine.

## Step 9: Evaluate A Saved Checkpoint Manually

Example:

```bash
python scripts/run_evaluate.py --config config.yaml --checkpoint outputs/YYYYMMDD_HHMMSS_modelname/checkpoints/best_model.pt
```

To skip Grad-CAM:

```bash
python scripts/run_evaluate.py --config config.yaml --checkpoint outputs/YYYYMMDD_HHMMSS_modelname/checkpoints/best_model.pt --no-gradcam
```

## Where Results Are Saved

Each run creates:

```text
outputs/
  YYYYMMDD_HHMMSS_modelname/
    checkpoints/
      best_model.pt
      last_model.pt
    plots/
      confusion_matrix.png
      roc_curve.png
      reliability_diagram.png
      uncertainty_histogram.png
      rejection_curve.png
      training_curves.png
      qubit_depth_ablation.csv
    gradcam/
    logs/
    metrics.json
    metrics.csv
    predictions.csv
    class_names.json
    config_used.yaml
    model_summary.txt
```

Important files:

- `metrics.json`: full metrics, calibration, uncertainty, parameter count, timing
- `metrics.csv`: simple scalar metric table
- `predictions.csv`: prediction per image
- `confusion_matrix.png`: class-by-class mistakes
- `roc_curve.png`: ROC curve when possible
- `reliability_diagram.png`: calibration plot
- `uncertainty_histogram.png`: entropy for correct vs wrong predictions
- `rejection_curve.png`: retained accuracy at confidence thresholds
- `gradcam/`: Grad-CAM images

## Common Errors And Fixes

### `ModuleNotFoundError`

Your virtual environment is probably not active, or dependencies were not installed.

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Windows:

```bat
venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset Not Found

Run:

```bash
python scripts/prepare_data.py --config config.yaml
```

Check that your dataset has class folders containing images. The code searches:

- `data/`
- `dataset/`
- `Brain_Tumor_MRI_Dataset/`
- the parent folder's `Brain_Tumor_MRI_Dataset/`

### MobileNet Download Fails

The scripts try to use pretrained MobileNetV2 weights. If the download fails,
the code prints a warning and uses random weights. For the best baseline results,
connect to the internet and rerun.

### Hybrid Quantum Model Is Slow

This is expected. The quantum layer uses a simulator and processes small batches.
Keep:

```yaml
n_qubits: 4
n_quantum_layers: 2
batch_size: 8
```

Reduce `mc_dropout_samples` during early debugging:

```yaml
mc_dropout_samples: 5
```

### Out Of Memory

Lower:

```yaml
batch_size: 4
image_size: 160
```

The dataset loader does not load all images into memory.
