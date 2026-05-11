<div align="center">

# 🧠 PyTorch Neural Network Comparison

A compact, reproducible PyTorch experiment for comparing a baseline MLP and an improved MLP on a synthetic 2D non-linear binary classification task.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900)
![Matplotlib](https://img.shields.io/badge/Visualization-Matplotlib-orange)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

## ✨ Overview

This repository compares two neural network configurations under the same dataset, split strategy, training loop, and evaluation workflow.

The goal is to show how several common improvements affect model performance:

- input feature standardization;
- deeper and wider hidden layers;
- optimizer replacement from SGD to Adam;
- activation replacement from ReLU to LeakyReLU;
- experiment logging and visualization.

## 📌 Experiment Pipeline

```text
Synthetic 2D Dataset
        │
        ├── Baseline MLP
        │   └── Raw features + ReLU + SGD
        │
        └── Improved MLP
            └── Standardized features + LeakyReLU + Adam
        │
        ▼
Training History → Metrics → Visualization → Result Export
```

## 📊 Visual Results

### Dataset and Decision Boundary

| Dataset Preview | Baseline Decision Boundary | Improved Decision Boundary |
| --- | --- | --- |
| ![Dataset](assets/dataset_preview.png) | ![Baseline Decision Boundary](assets/decision_boundary_baseline.png) | ![Improved Decision Boundary](assets/decision_boundary_improved.png) |

### Training Curves and Confusion Matrices

| Test Loss Curve | Test Accuracy Curve |
| --- | --- |
| ![Loss Curve](assets/loss_curve.png) | ![Accuracy Curve](assets/accuracy_curve.png) |

<p align="center">
  <img src="assets/confusion_matrices.png" width="720" alt="Confusion matrix comparison">
</p>

## 🧪 Model Comparison

| Item | Baseline | Improved |
| --- | --- | --- |
| Input features | Raw 2D features | Standardized 2D features |
| Hidden layers | 1 hidden layer | 2 hidden layers |
| Hidden units | 8 | 32 + 32 |
| Activation | ReLU | LeakyReLU |
| Optimizer | SGD | Adam |
| Learning rate | 0.05 | 0.01 |
| Trainable parameters | 42 | 1,218 |
| Loss function | CrossEntropyLoss | CrossEntropyLoss |

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For CUDA acceleration, install the PyTorch build that matches your local CUDA environment from the official PyTorch installation guide, then install the remaining dependencies.

### 2. Run the experiment

```bash
python train_compare.py
```

### 3. Force CPU execution

```bash
python train_compare.py --cpu
```

### 4. Change training settings

```bash
python train_compare.py --epochs 120 --batch-size 64 --output-dir results --assets-dir assets
```

### 5. Skip figure generation

```bash
python train_compare.py --no-plots
```

## ⚙️ Command Line Arguments

| Argument | Default | Description |
| --- | ---: | --- |
| `--samples` | 1200 | Number of generated samples |
| `--test-ratio` | 0.25 | Test set ratio |
| `--epochs` | 120 | Training epochs for each model |
| `--batch-size` | 64 | Mini-batch size |
| `--baseline-lr` | 0.05 | Learning rate for the baseline model |
| `--improved-lr` | 0.01 | Learning rate for the improved model |
| `--seed` | 42 | Random seed |
| `--num-threads` | 1 | Number of CPU threads used by PyTorch |
| `--cpu` | False | Force CPU execution |
| `--output-dir` | `results` | Directory for result files |
| `--assets-dir` | `assets` | Directory for generated figures |
| `--no-plots` | False | Disable optional figure generation |

## ✅ Example Result

One run with the default configuration produced the following result:

| Model | Parameters | Train loss | Test loss | Test accuracy | Method |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 42 | 0.1224 | 0.1354 | 97.67% | Raw features + 1 hidden layer + ReLU + SGD |
| Improved | 1,218 | 0.0238 | 0.0334 | 99.33% | Standardization + 2 hidden layers + LeakyReLU + Adam |

The improved model achieves lower training loss, lower test loss, and higher test accuracy on this synthetic task.

## 📦 Output Files

Running `train_compare.py` generates result files and visualizations:

```text
pytorch-nn-compare/
├── assets/
│   ├── accuracy_curve.png
│   ├── confusion_matrices.png
│   ├── dataset_preview.png
│   ├── decision_boundary_baseline.png
│   ├── decision_boundary_improved.png
│   └── loss_curve.png
└── results/
    ├── example_output.txt
    ├── history.csv
    ├── results.txt
    └── summary.json
```

## 📁 Project Structure

```text
pytorch-nn-compare/
├── README.md
├── train_compare.py
├── requirements.txt
├── assets/
├── docs/
│   ├── DESIGN_NOTES.md
│   └── EXPERIMENT_REPORT.md
├── results/
├── .gitignore
└── LICENSE
```

## 📝 Documents

- [`docs/EXPERIMENT_REPORT.md`](docs/EXPERIMENT_REPORT.md): detailed experiment report.
- [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md): implementation and design notes.

## ⚠️ Notes

This project uses a synthetic dataset. It is designed for controlled neural network comparison and training workflow demonstration, not for measuring performance on a real-world dataset.
