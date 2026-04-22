# Self-Pruning Neural Network — CIFAR-10

A neural network that learns to prune itself **during** training using learnable gate parameters.

---

## What's in here

| File | Description |
|------|-------------|
| `main.ipynb` | Full implementation — run this top to bottom |
| `REPORT.md` | Write-up: why L1 gates work, results, observations |
| `gate_distribution.png` | Generated after training — gate histograms per λ |

---

## How it works

Each linear layer has a `gate_scores` parameter (same shape as weights). During forward pass:

```
gates       = sigmoid(gate_scores)      # always in (0, 1)
pruned_w    = weight * gates            # element-wise mask
output      = pruned_w @ x + bias
```

Loss = `CrossEntropy + λ × sum(gates)` — the L1 penalty drives unused gates toward zero.

---

## Experiment

Trained on CIFAR-10 with three λ values to compare accuracy vs sparsity:

| λ | Test Accuracy | Sparsity |
|---|--------------|----------|
| 1e-4 | ~52% | ~3% |
| 1e-3 | ~48% | ~18% |
| 1e-2 | ~35% | ~62% |

---

## Setup

```bash
# uses the venv already in the repo
# just open main.ipynb and run all cells
# first cell installs torch, torchvision, matplotlib automatically
```

Tested on Python 3.11.9. No GPU required (CPU works fine, ~15–20 min for full run).
