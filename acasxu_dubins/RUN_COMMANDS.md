# Run Commands

All commands below assume your current working directory is:

```bash
~/projects/acasxu_closed_loop_analysis/acasxu_dubins
```

## 1. Run Original ACASXu Dubins Simulator

```bash
python3 acasxu_dubins.py \
  --seed min \
  --num-sims 10000 \
  --turning-ratio 0.9 \
  --max-sampling-attempts 500000 \
  --min-dwell-time 0.0 \
  --q-hysteresis-margin 0.02 \
  --continuity-bias 0.02 \
  --direct-reversal-margin 0.05 \
  --camera-fps 1.0 \
  --camera-dropout-prob 0.0 \
  --camera-pos-noise-std 0.0 \
  --camera-ema-alpha 0.35 \
  --camera-max-track-age 2.0 \
  --camera-max-speed 1400.0 \
  --camera-max-speed-jump 400.0 \
  --camera-bad-frames-trip 3 \
  --camera-good-frames-clear 5 \
  --camera-rng-seed 0 \
  --nmac-distance 500.0 \
  --false-alert-distance 4000.0 \
  --nuisance-max-alert-time 2.0
```

## 2. Run RL-Based Simulator Using Your Trained ONNX Policy

```bash
python3 acasxu_dubins_rl.py \
  --rl-model-path trained_rl/dqn_best.onnx \
  --well-clear-distance 2200 \
  --seed min \
  --num-sims 10000 \
  --turning-ratio 0.9 \
  --max-sampling-attempts 500000 \
  --min-dwell-time 0.0 \
  --q-hysteresis-margin 0.05 \
  --continuity-bias 0.05 \
  --direct-reversal-margin 0.10
```

Single-seed quick check:

```bash
python3 acasxu_dubins_rl.py --rl-model-path trained_rl/dqn_best.onnx --well-clear-distance 2200 --seed 1
```

## 3. Train RL Policy From Scratch (Well-Clear Objective)

```bash
python3 tools/train_dubins_rl.py \
  --intruder-turn \
  --well-clear-distance 2200 \
  --wcv-penalty 100.0 \
  --sep-gain 0.002 \
  --alert-penalty 0.08 \
  --alert-duration-penalty 0.05 \
  --strong-alert-penalty 0.08 \
  --switch-penalty 0.08 \
  --reversal-penalty 0.25 \
  --episodes 8000 \
  --horizon 100 \
  --eval-every 250 \
  --eval-episodes 200 \
  --eps-end 0.02
```

Quick smoke test:

```bash
python3 tools/train_dubins_rl.py --episodes 300 --horizon 60 --eval-every 100 --eval-episodes 30
```

## 4. Export RL Checkpoint to ONNX

```bash
python3 -c "import torch; from tools.train_dubins_rl import QNet; ckpt=torch.load('trained_rl/dqn_best.pt', map_location='cpu'); h=ckpt['args']['hidden_size']; m=QNet(h); m.load_state_dict(ckpt['model_state']); m.eval(); x=torch.zeros(1,6,dtype=torch.float32); torch.onnx.export(m,x,'trained_rl/dqn_best.onnx',input_names=['input'],output_names=['q_values'],dynamic_axes={'input':{0:'batch'},'q_values':{0:'batch'}},opset_version=13); print('saved trained_rl/dqn_best.onnx')"
```

## 5. Evaluate Saved RL Checkpoint

```bash
python3 -c "import torch; ckpt=torch.load('trained_rl/dqn_best.pt', map_location='cpu'); print('episode:', ckpt['episode']); print('steps:', ckpt['total_steps']); print('metrics:', ckpt['metrics'])"
```

```bash
python3 -c "import torch; from tools.train_dubins_rl import QNet,DubinsRLEnv,evaluate_policy; dev='cuda' if torch.cuda.is_available() else 'cpu'; ckpt=torch.load('trained_rl/dqn_best.pt', map_location=dev); args=ckpt['args']; net=QNet(args['hidden_size']).to(dev); net.load_state_dict(ckpt['model_state']); env=DubinsRLEnv(dt=args['dt'], horizon=args['horizon'], intruder_can_turn=args['intruder_turn'], well_clear_distance=args.get('well_clear_distance', 2200), reward_cfg={'sep_gain':args['sep_gain'],'alert_penalty':args['alert_penalty'],'alert_duration_penalty':args.get('alert_duration_penalty', 0.0),'strong_alert_penalty':args.get('strong_alert_penalty', 0.0),'switch_penalty':args['switch_penalty'],'reversal_penalty':args.get('reversal_penalty', 0.0),'wcv_penalty':args.get('wcv_penalty', args.get('nmac_penalty', 20.0))}); print(evaluate_policy(env, net, dev, episodes=1000, seed_base=123456))"
```

## 6. Train Imitation Policy From ONNX Teachers

```bash
python3 tools/train_dubins_policy.py --last-cmd all --num-train 120000 --num-val 20000 --epochs 20
```

Quick smoke test:

```bash
python3 tools/train_dubins_policy.py --last-cmd 0 --num-train 20000 --num-val 4000 --epochs 5
```

## 7. Build Lookup Table

```bash
python3 tools/build_lookup_table.py build --output acasxu_lookup_table.npz
```

Query one state:

```bash
python3 tools/build_lookup_table.py query \
  --table-path acasxu_lookup_table.npz \
  --last-cmd 0 \
  --rho 30000 \
  --theta 0.1 \
  --psi -0.2 \
  --v-own 800 \
  --v-int 500
```
