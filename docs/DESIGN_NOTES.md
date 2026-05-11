# 🛠️ Design Notes

## 1. Dataset Design

The dataset is generated rather than downloaded. This keeps the experiment lightweight and reproducible.

Each sample has two input features. The label is built from a radius term and sinusoidal perturbations, which creates a non-linear decision boundary. This makes the task simple enough to visualize but still more meaningful than a linearly separable toy dataset.

## 2. Model Design

### Baseline MLP

The baseline network is intentionally small:

```text
Input(2) → Linear(2, 8) → ReLU → Linear(8, 2)
```

It has only 42 trainable parameters. This makes it useful as a low-capacity reference model.

### Improved MLP

The improved network increases capacity and changes the optimization strategy:

```text
Input(2) → Linear(2, 32) → LeakyReLU
         → Linear(32, 32) → LeakyReLU
         → Linear(32, 2)
```

It has 1,218 trainable parameters. It also uses standardized input features and the Adam optimizer.

## 3. Training Loop

The training loop is implemented directly in `train_compare.py` instead of hiding the logic inside a framework wrapper. The script explicitly performs:

1. mini-batch loading;
2. forward propagation;
3. loss calculation;
4. gradient clearing;
5. backpropagation;
6. optimizer step;
7. per-epoch evaluation.

This structure makes the experiment easy to read and modify.

## 4. Evaluation

The script exports several evaluation artifacts:

- final train/test loss and test accuracy;
- per-epoch history in CSV format;
- summary JSON for structured result inspection;
- loss and accuracy curves;
- baseline and improved decision boundaries;
- confusion matrix comparison.

## 5. Reproducibility Choices

The script fixes random seeds for Python, NumPy, and PyTorch. It also sets the default PyTorch CPU thread count to 1 through `--num-threads 1`, which makes execution more stable across different machines.

CUDA is used automatically when available unless `--cpu` is specified.
