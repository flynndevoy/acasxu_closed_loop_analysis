"""
Train a Dubins advisory policy from scratch using DQN (no ONNX imitation).

State features:
  [rho, theta, psi, v_own, v_int, last_cmd]

Action space:
  0: clear-of-conflict
  1: weak-left
  2: weak-right
  3: strong-left
  4: strong-right

This script uses a simple reward with:
- separation shaping
- alert penalty
- alert-duration penalty
- strong-alert penalty
- advisory-switch penalty
- reversal penalty
- strong well-clear violation (WCV) penalty
"""

import argparse
import math
import os
import random
from collections import deque

import numpy as np
from scipy.linalg import expm
import torch
from torch import nn


RHO_MAX = 60760.0
WCV_DEFAULT = 2200.0


def make_random_input(seed, intruder_can_turn=True, num_inputs=100):
    """Generate one random encounter."""

    np.random.seed(seed)

    # state: x1, y1, theta1, x2, y2, theta2, t
    init_vec = np.zeros(7, dtype=np.float64)
    init_vec[2] = np.pi / 2

    radius = 10000 + np.random.random() * 55000
    angle = np.random.random() * 2 * np.pi
    init_vec[3] = radius * np.cos(angle)
    init_vec[4] = radius * np.sin(angle)
    init_vec[5] = np.random.random() * 2 * np.pi

    if intruder_can_turn:
        cmd_list = [int(np.random.randint(5)) for _ in range(num_inputs)]
    else:
        cmd_list = [0] * num_inputs

    init_velo = [int(np.random.randint(100, 1200)), int(np.random.randint(0, 1200))]
    return init_vec, cmd_list, init_velo


def state7_to_state5(state7, v_own, v_int):
    """Compute ACAS-style observation [rho, theta, psi, v_own, v_int]."""

    x1, y1, theta1, x2, y2, theta2, _ = state7

    rho = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    theta = math.atan2(y2 - y1, x2 - x1) - theta1
    psi = theta2 - theta1

    while theta < -math.pi:
        theta += 2 * math.pi
    while theta > math.pi:
        theta -= 2 * math.pi

    while psi < -math.pi:
        psi += 2 * math.pi
    while psi > math.pi:
        psi -= 2 * math.pi

    return np.array([rho, theta, psi, float(v_own), float(v_int)], dtype=np.float32)


def state7_to_state8(state7, v_own, v_int):
    """Convert heading state to position-velocity state."""

    x1, y1, h1, x2, y2, h2, _ = state7
    return np.array(
        [
            x1,
            y1,
            math.cos(h1) * v_own,
            math.sin(h1) * v_own,
            x2,
            y2,
            math.cos(h2) * v_int,
            math.sin(h2) * v_int,
        ],
        dtype=np.float64,
    )


def step_state(state7, v_own, v_int, time_elapse_mat, dt):
    """One dynamics step using matrix exponential integration."""

    s8 = state7_to_state8(state7, v_own, v_int)
    nxt = time_elapse_mat @ s8
    theta1 = math.atan2(nxt[3], nxt[2])
    theta2 = math.atan2(nxt[7], nxt[6])
    return np.array([nxt[0], nxt[1], theta1, nxt[4], nxt[5], theta2, state7[-1] + dt], dtype=np.float64)


def init_time_elapse_mats(dt):
    y_list = [0.0, 1.5, -1.5, 3.0, -3.0]
    mats = []
    for own_cmd in range(5):
        row = []
        dtheta1 = y_list[own_cmd] / 180.0 * np.pi
        for int_cmd in range(5):
            dtheta2 = y_list[int_cmd] / 180.0 * np.pi
            a_mat = np.array(
                [
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, -dtheta1, 0, 0, 0, 0],
                    [0, 0, dtheta1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, -dtheta2],
                    [0, 0, 0, 0, 0, 0, dtheta2, 0],
                ],
                dtype=np.float64,
            )
            row.append(expm(a_mat * dt))
        mats.append(row)
    return mats


class DubinsRLEnv:
    """Small RL environment wrapper over Dubins encounter dynamics."""

    def __init__(self, dt, horizon, intruder_can_turn, well_clear_distance, reward_cfg):
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.intruder_can_turn = bool(intruder_can_turn)
        self.well_clear_distance = float(well_clear_distance)
        self.reward_cfg = reward_cfg
        self.mats = init_time_elapse_mats(self.dt)

        self.state = None
        self.cmd_list = None
        self.v_own = None
        self.v_int = None
        self.step_index = 0
        self.last_cmd = 0
        self.prev_rho = None
        self.had_alert = False

    def reset(self, seed):
        init_vec, cmd_list, init_velo = make_random_input(
            seed=seed, intruder_can_turn=self.intruder_can_turn, num_inputs=self.horizon
        )
        self.state = init_vec.copy()
        self.cmd_list = cmd_list
        self.v_own = float(init_velo[0])
        self.v_int = float(init_velo[1])
        self.step_index = 0
        self.last_cmd = 0
        self.had_alert = False

        rho = state7_to_state5(self.state, self.v_own, self.v_int)[0]
        self.prev_rho = float(rho)
        return self._obs()

    def _obs(self):
        s5 = state7_to_state5(self.state, self.v_own, self.v_int)
        rho, theta, psi, v_own, v_int = s5
        obs = np.array(
            [
                np.clip(rho / RHO_MAX, 0.0, 2.0),
                theta / np.pi,
                psi / np.pi,
                v_own / 1200.0,
                v_int / 1200.0,
                self.last_cmd / 4.0,
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action):
        action = int(action)
        int_cmd = self.cmd_list[min(self.step_index, len(self.cmd_list) - 1)]

        self.state = step_state(self.state, self.v_own, self.v_int, self.mats[action][int_cmd], self.dt)
        rho = float(state7_to_state5(self.state, self.v_own, self.v_int)[0])

        reward = 0.0
        reward += self.reward_cfg["sep_gain"] * (rho - self.prev_rho)
        reward -= self.reward_cfg["alert_penalty"] * float(action != 0)
        reward -= self.reward_cfg["alert_duration_penalty"] * float(action != 0)
        reward -= self.reward_cfg["strong_alert_penalty"] * float(action in (3, 4))
        reward -= self.reward_cfg["switch_penalty"] * float(action != self.last_cmd)
        reward -= self.reward_cfg["reversal_penalty"] * float(
            (self.last_cmd in (1, 3) and action in (2, 4)) or
            (self.last_cmd in (2, 4) and action in (1, 3))
        )

        done = False
        if rho < self.well_clear_distance:
            reward -= self.reward_cfg["wcv_penalty"]
            done = True

        self.step_index += 1
        if self.step_index >= self.horizon:
            done = True

        self.had_alert = self.had_alert or (action != 0)
        self.prev_rho = rho
        self.last_cmd = action

        info = {"rho": rho, "had_alert": self.had_alert}
        return self._obs(), reward, done, info


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def add(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, batch_size, device):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        s = torch.as_tensor(np.asarray(s), dtype=torch.float32, device=device)
        a = torch.as_tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
        r = torch.as_tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
        ns = torch.as_tensor(np.asarray(ns), dtype=torch.float32, device=device)
        d = torch.as_tensor(d, dtype=torch.float32, device=device).unsqueeze(1)
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buf)


class QNet(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 5),
        )

    def forward(self, x):
        return self.net(x)


def linear_epsilon(step, eps_start, eps_end, eps_decay_steps):
    if step >= eps_decay_steps:
        return eps_end
    alpha = step / float(max(1, eps_decay_steps))
    return (1.0 - alpha) * eps_start + alpha * eps_end


def evaluate_policy(env, qnet, device, episodes, seed_base):
    qnet.eval()
    total_reward = 0.0
    wcv_count = 0
    alert_count = 0

    with torch.no_grad():
        for ep in range(episodes):
            s = env.reset(seed_base + ep)
            done = False
            ep_reward = 0.0
            last_info = {"rho": np.inf, "had_alert": False}

            while not done:
                inp = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                a = int(torch.argmax(qnet(inp), dim=1).item())
                s, r, done, info = env.step(a)
                ep_reward += float(r)
                last_info = info

            total_reward += ep_reward
            if last_info["rho"] < env.well_clear_distance:
                wcv_count += 1
            if last_info["had_alert"]:
                alert_count += 1

    return {
        "avg_reward": total_reward / max(1, episodes),
        "wcv_rate": wcv_count / max(1, episodes),
        "alert_rate": alert_count / max(1, episodes),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train Dubins advisory policy from scratch with DQN.")
    p.add_argument("--episodes", type=int, default=6000, help="Training episodes.")
    p.add_argument("--horizon", type=int, default=100, help="Max steps per episode.")
    p.add_argument("--dt", type=float, default=1.0, help="Step size in seconds.")
    p.add_argument("--intruder-turn", action="store_true", default=False, help="Allow intruder maneuvers.")
    p.add_argument("--well-clear-distance", type=float, default=WCV_DEFAULT, help="Well-clear violation threshold in feet.")
    p.add_argument("--nmac-distance", type=float, default=None, help="Deprecated alias for --well-clear-distance.")

    p.add_argument("--hidden-size", type=int, default=128, help="Q-network hidden width.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size for replay updates.")
    p.add_argument("--replay-size", type=int, default=200000, help="Replay buffer capacity.")
    p.add_argument("--warmup-steps", type=int, default=5000, help="Min replay samples before learning.")
    p.add_argument("--train-freq", type=int, default=1, help="Gradient step frequency in env steps.")
    p.add_argument("--target-update", type=int, default=2000, help="Target net hard update interval in env steps.")

    p.add_argument("--eps-start", type=float, default=1.0, help="Initial epsilon.")
    p.add_argument("--eps-end", type=float, default=0.05, help="Final epsilon.")
    p.add_argument("--eps-decay-steps", type=int, default=200000, help="Linear epsilon decay steps.")

    p.add_argument("--sep-gain", type=float, default=0.002, help="Reward gain for increasing separation.")
    p.add_argument("--alert-penalty", type=float, default=0.02, help="Penalty per non-COC advisory step.")
    p.add_argument("--alert-duration-penalty", type=float, default=0.0, help="Additional per-step penalty while any alert is active.")
    p.add_argument("--strong-alert-penalty", type=float, default=0.0, help="Extra per-step penalty for strong-left/strong-right advisories.")
    p.add_argument("--switch-penalty", type=float, default=0.05, help="Penalty for advisory changes.")
    p.add_argument("--reversal-penalty", type=float, default=0.0, help="Penalty for direct left<->right reversals.")
    p.add_argument("--wcv-penalty", type=float, default=20.0, help="Penalty when rho < well-clear threshold.")
    p.add_argument("--nmac-penalty", type=float, default=None, help="Deprecated alias for --wcv-penalty.")

    p.add_argument("--eval-every", type=int, default=250, help="Eval interval (episodes).")
    p.add_argument("--eval-episodes", type=int, default=100, help="Episodes per eval pass.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--out-dir", type=str, default="trained_rl", help="Output directory.")
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    well_clear_distance = float(args.well_clear_distance)
    if args.nmac_distance is not None:
        well_clear_distance = float(args.nmac_distance)

    wcv_penalty = float(args.wcv_penalty)
    if args.nmac_penalty is not None:
        wcv_penalty = float(args.nmac_penalty)

    reward_cfg = {
        "sep_gain": float(args.sep_gain),
        "alert_penalty": float(args.alert_penalty),
        "alert_duration_penalty": float(args.alert_duration_penalty),
        "strong_alert_penalty": float(args.strong_alert_penalty),
        "switch_penalty": float(args.switch_penalty),
        "reversal_penalty": float(args.reversal_penalty),
        "wcv_penalty": wcv_penalty,
    }

    env = DubinsRLEnv(
        dt=args.dt,
        horizon=args.horizon,
        intruder_can_turn=args.intruder_turn,
        well_clear_distance=well_clear_distance,
        reward_cfg=reward_cfg,
    )
    eval_env = DubinsRLEnv(
        dt=args.dt,
        horizon=args.horizon,
        intruder_can_turn=args.intruder_turn,
        well_clear_distance=well_clear_distance,
        reward_cfg=reward_cfg,
    )

    qnet = QNet(args.hidden_size).to(device)
    tgt = QNet(args.hidden_size).to(device)
    tgt.load_state_dict(qnet.state_dict())
    tgt.eval()

    opt = torch.optim.Adam(qnet.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()
    replay = ReplayBuffer(args.replay_size)

    total_steps = 0
    best_eval_reward = -np.inf
    os.makedirs(args.out_dir, exist_ok=True)

    for ep in range(1, args.episodes + 1):
        s = env.reset(seed=args.seed + ep)
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done:
            eps = linear_epsilon(total_steps, args.eps_start, args.eps_end, args.eps_decay_steps)
            if random.random() < eps:
                a = random.randint(0, 4)
            else:
                with torch.no_grad():
                    inp = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    a = int(torch.argmax(qnet(inp), dim=1).item())

            ns, r, done, _info = env.step(a)
            replay.add(s, a, r, ns, float(done))

            s = ns
            ep_reward += float(r)
            ep_len += 1
            total_steps += 1

            if (
                len(replay) >= args.warmup_steps
                and total_steps % args.train_freq == 0
                and len(replay) >= args.batch_size
            ):
                bs, ba, br, bns, bd = replay.sample(args.batch_size, device)

                q = qnet(bs).gather(1, ba)
                with torch.no_grad():
                    nxt_q = tgt(bns).max(dim=1, keepdim=True).values
                    target = br + args.gamma * (1.0 - bd) * nxt_q

                loss = loss_fn(q, target)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(qnet.parameters(), max_norm=10.0)
                opt.step()

            if total_steps % args.target_update == 0:
                tgt.load_state_dict(qnet.state_dict())

        if ep % 50 == 0:
            print(f"ep={ep:5d} steps={total_steps:8d} eps={eps:.3f} ep_reward={ep_reward:.3f} ep_len={ep_len}")

        if ep % args.eval_every == 0:
            metrics = evaluate_policy(
                env=eval_env,
                qnet=qnet,
                device=device,
                episodes=args.eval_episodes,
                seed_base=args.seed + 100000 + ep,
            )
            print(
                f"[eval] ep={ep:5d} avg_reward={metrics['avg_reward']:.3f} "
                f"wcv_rate={100.0*metrics['wcv_rate']:.2f}% "
                f"alert_rate={100.0*metrics['alert_rate']:.2f}%"
            )

            ckpt_latest = os.path.join(args.out_dir, "dqn_latest.pt")
            torch.save(
                {
                    "episode": ep,
                    "total_steps": total_steps,
                    "model_state": qnet.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "args": vars(args),
                    "metrics": metrics,
                },
                ckpt_latest,
            )

            if metrics["avg_reward"] > best_eval_reward:
                best_eval_reward = metrics["avg_reward"]
                ckpt_best = os.path.join(args.out_dir, "dqn_best.pt")
                torch.save(
                    {
                        "episode": ep,
                        "total_steps": total_steps,
                        "model_state": qnet.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "args": vars(args),
                        "metrics": metrics,
                    },
                    ckpt_best,
                )
                print(f"Saved new best checkpoint: {ckpt_best}")

    print("Training complete.")
    print(f"Best eval avg_reward: {best_eval_reward:.3f}")


if __name__ == "__main__":
    main()
