# `train_dubins_policy.py` README

## What This Script Does

`train_dubins_policy.py` trains a simple PyTorch MLP policy for ACAS-style Dubins inputs by imitation learning from the existing ONNX teacher models in this folder.

- Input features: `[rho, theta, psi, v_own, v_int]`
- Target label: advisory class `argmin(Q)` from ACASXu ONNX teacher
- Advisory classes:
  - `0`: clear-of-conflict
  - `1`: weak-left
  - `2`: weak-right
  - `3`: strong-left
  - `4`: strong-right

The script can train one policy (`--last-cmd 0..4`) or all five (`--last-cmd all`), then saves both `.pt` and exported `.onnx` models.

## Formulas Used

`train_dubins_policy.py` uses the following core math:

1. Uniform state sampling

For each sample:

- `rho ~ U(0, 60760)`
- `theta ~ U(-pi, pi)`
- `psi ~ U(-pi, pi)`
- `v_own ~ U(100, 1200)`
- `v_int ~ U(0, 1200)`

2. Input normalization (same scaling used by ACASXu inference)

For input vector `x = [rho, theta, psi, v_own, v_int]`:

- `x_norm[i] = (x[i] - mean[i]) / range[i]`

with:

- `mean = [19791.091, 0.0, 0.0, 650.0, 600.0]`
- `range = [60261.0, 2*pi, 2*pi, 1100.0, 1200.0]`

3. Teacher label generation from ONNX

The ONNX model returns Q-values:

- `Q(x_norm) = [Q_0, Q_1, Q_2, Q_3, Q_4]`

Target class is:

- `y = argmin_a Q_a(x_norm)`, where `a in {0,1,2,3,4}`

4. Student policy network

The PyTorch MLP computes logits:

- `z = f_theta(x_norm) in R^5`

Architecture:

- `Linear(5, H) -> ReLU -> Linear(H, H) -> ReLU -> Linear(H, 5)`

where `H = --hidden-size` (default `128`).

5. Training objective

Cross-entropy loss between student logits and teacher class:

- `L = CE(z, y) = -log(softmax(z)_y)`

Optimization uses Adam with learning rate `--lr` (default `1e-3`).

## Requirements

You need these Python packages in your active virtual environment:

- `torch`
- `onnx`
- `onnxruntime`
- `numpy`

## Basic Usage

All commands below assume you are already inside the `acasxu_dubins` directory.

Quick smoke test (single policy):

```bash
python3 train_dubins_policy.py --last-cmd 0 --num-train 20000 --num-val 4000 --epochs 5
```

Train all 5 policies:

```bash
python3 train_dubins_policy.py --last-cmd all --num-train 120000 --num-val 20000 --epochs 20
```

## Key Arguments

- `--last-cmd`: `0..4` or `all` (default: `all`)
- `--num-train`: number of random training samples (default: `120000`)
- `--num-val`: number of random validation samples (default: `20000`)
- `--epochs`: training epochs (default: `20`)
- `--batch-size`: PyTorch training batch size (default: `1024`)
- `--teacher-batch-size`: ONNX teacher labeling batch size (default: `2048`)
- `--hidden-size`: hidden width of the MLP (default: `128`)
- `--lr`: learning rate (default: `1e-3`)
- `--seed`: random seed (default: `0`)
- `--out-dir`: output folder (default: `trained_policies`)

## Output Files

For each trained `last_cmd`, the script writes:

- `trained_policies/dubins_policy_last_cmd_<k>.pt`
- `trained_policies/dubins_policy_last_cmd_<k>.onnx`

Example (`k=0`):

- `trained_policies/dubins_policy_last_cmd_0.pt`
- `trained_policies/dubins_policy_last_cmd_0.onnx`

## Notes on Teacher Inference

Some ACASXu ONNX files are fixed to input batch size `1`. The script handles this automatically:

- if batched ONNX inference works, it uses batched labeling,
- otherwise it falls back to sample-by-sample teacher queries.

## Troubleshooting

1. Error: `NoSuchFile ... ACASXU_run2a_*`
- Ensure you are running from inside `acasxu_dubins` and ONNX files are present.

2. Error: ONNX input dimension expects batch `1`
- Already handled by script fallback; update to latest script if needed.

3. GPU not used
- Check:
```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
- If `False`, reinstall a CUDA-enabled PyTorch wheel matching your Python version.

## Model Scope

This is a simple imitation baseline, not the original ACASXu training pipeline. It is useful for experiments and prototyping, but does not replicate official ACAS training/verification guarantees.
