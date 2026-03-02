"""
Train a simple PyTorch policy network for Dubins-state ACAS-style advisories.

This script trains by imitation from the existing ONNX ACASXu policies:
- input:  [rho, theta, psi, v_own, v_int]
- target: advisory class argmin(Q) from selected teacher ONNX net(s)

Usage example:
  python3 train_dubins_policy.py \
    --num-train 120000 \
    --num-val 20000 \
    --epochs 20 \
    --batch-size 1024 \
    --hidden-size 128 \
    --last-cmd all \
    --out-dir trained_policies
"""

import argparse
import os
import random
import time

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MEANS_FOR_SCALING = np.array([19791.091, 0.0, 0.0, 650.0, 600.0], dtype=np.float32)
RANGE_FOR_SCALING = np.array([60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0], dtype=np.float32)

RHO_MIN = 0.0
RHO_MAX = 60760.0
THETA_MIN = -np.pi
THETA_MAX = np.pi
PSI_MIN = -np.pi
PSI_MAX = np.pi
V_OWN_MIN = 100.0
V_OWN_MAX = 1200.0
V_INT_MIN = 0.0
V_INT_MAX = 1200.0

ADVISORY_NAMES = [
    "clear-of-conflict",
    "weak-left",
    "weak-right",
    "strong-left",
    "strong-right",
]


def get_onnx_path(last_cmd):
    return os.path.join(BASE_DIR, f"ACASXU_run2a_{last_cmd + 1}_1_batch_2000.onnx")


def normalize_features(features):
    out = features.astype(np.float32, copy=True)
    out -= MEANS_FOR_SCALING
    out /= RANGE_FOR_SCALING
    return out


def sample_states(num_samples, rng):
    x = np.empty((num_samples, 5), dtype=np.float32)
    x[:, 0] = rng.uniform(RHO_MIN, RHO_MAX, size=num_samples)
    x[:, 1] = rng.uniform(THETA_MIN, THETA_MAX, size=num_samples)
    x[:, 2] = rng.uniform(PSI_MIN, PSI_MAX, size=num_samples)
    x[:, 3] = rng.uniform(V_OWN_MIN, V_OWN_MAX, size=num_samples)
    x[:, 4] = rng.uniform(V_INT_MIN, V_INT_MAX, size=num_samples)
    return x


def run_teacher_argmin(session, features, batch_size):
    input_name = session.get_inputs()[0].name
    labels = np.empty((features.shape[0],), dtype=np.int64)
    batch_supported = True

    # Some ACASXu ONNX files are exported with fixed batch=1.
    # Probe once and gracefully fall back to per-sample evaluation.
    probe = normalize_features(features[: min(2, features.shape[0])])
    try:
        session.run(None, {input_name: probe.reshape((-1, 1, 1, 5))})
    except Exception:
        batch_supported = False

    i = 0
    while i < features.shape[0]:
        if batch_supported:
            j = min(features.shape[0], i + batch_size)
            batch = normalize_features(features[i:j]).reshape((-1, 1, 1, 5))
            q_vals = session.run(None, {input_name: batch})[0]
            labels[i:j] = np.argmin(q_vals, axis=1)
            i = j
        else:
            batch = normalize_features(features[i : i + 1]).reshape((1, 1, 1, 5))
            q_vals = session.run(None, {input_name: batch})[0]
            labels[i] = int(np.argmin(q_vals[0]))
            i += 1

    return labels


class MLPPolicy(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 5),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = ce(logits, yb)

            pred = torch.argmax(logits, dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
            loss_sum += float(loss.item()) * yb.shape[0]

    acc = 0.0 if total == 0 else correct / total
    mean_loss = 0.0 if total == 0 else loss_sum / total
    return mean_loss, acc


def train_one_policy(args, last_cmd, device):
    print(f"\n=== Training policy for last_cmd={last_cmd} ({ADVISORY_NAMES[last_cmd]}) ===")
    teacher = ort.InferenceSession(get_onnx_path(last_cmd))

    rng = np.random.default_rng(args.seed + last_cmd)
    x_train = sample_states(args.num_train, rng)
    x_val = sample_states(args.num_val, rng)

    t0 = time.perf_counter()
    y_train = run_teacher_argmin(teacher, x_train, args.teacher_batch_size)
    y_val = run_teacher_argmin(teacher, x_val, args.teacher_batch_size)
    print(f"Labeled teacher dataset in {time.perf_counter() - t0:.1f}s")

    x_train_n = normalize_features(x_train)
    x_val_n = normalize_features(x_val)

    train_ds = TensorDataset(
        torch.from_numpy(x_train_n),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val_n),
        torch.from_numpy(y_val.astype(np.int64)),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = MLPPolicy(hidden_size=args.hidden_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = ce(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        train_loss, train_acc = evaluate(model, train_loader, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={100.0*train_acc:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={100.0*val_acc:.2f}%"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    pt_path = os.path.join(args.out_dir, f"dubins_policy_last_cmd_{last_cmd}.pt")
    onnx_path = os.path.join(args.out_dir, f"dubins_policy_last_cmd_{last_cmd}.onnx")

    model_cpu = model.to("cpu").eval()
    torch.save(model_cpu.state_dict(), pt_path)
    dummy = torch.zeros((1, 5), dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )

    print(f"Saved PyTorch weights: {pt_path}")
    print(f"Saved ONNX model:      {onnx_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train simple Dubins policy network via ONNX imitation.")
    p.add_argument("--num-train", type=int, default=120000, help="Number of random training states.")
    p.add_argument("--num-val", type=int, default=20000, help="Number of random validation states.")
    p.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=1024, help="Minibatch size.")
    p.add_argument("--teacher-batch-size", type=int, default=2048, help="Batch size for ONNX teacher labeling.")
    p.add_argument("--hidden-size", type=int, default=128, help="Hidden width for MLP.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument(
        "--last-cmd",
        type=str,
        default="all",
        help="Which teacher(s) to train: one of 0,1,2,3,4 or 'all'.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--out-dir", type=str, default="trained_policies", help="Output directory.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.num_train <= 0 or args.num_val <= 0:
        raise ValueError("--num-train and --num-val must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.last_cmd == "all":
        cmds = [0, 1, 2, 3, 4]
    else:
        cmd = int(args.last_cmd)
        if cmd < 0 or cmd > 4:
            raise ValueError("--last-cmd must be 0..4 or 'all'")
        cmds = [cmd]

    for last_cmd in cmds:
        train_one_policy(args, last_cmd, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
