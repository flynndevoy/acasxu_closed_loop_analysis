# `build_lookup_table.py` explained

This file explains what `build_lookup_table.py` does and how to read its outputs.

## Goal

The script converts the ACAS Xu ONNX policies into a **discrete lookup table** so you can inspect decisions without running online neural-network inference each time.

It supports two modes:

- `build`: precompute the table and save it to `.npz`
- `query`: look up one state from the saved table

## Inputs to ACAS Xu policy

Each policy decision depends on:

- `last_cmd` (previous advisory, 0..4)
- `rho` (distance between aircraft, ft)
- `theta` (bearing of intruder relative to ownship heading, rad)
- `psi` (intruder heading relative to ownship heading, rad)
- `v_own` (ownship speed, ft/s)
- `v_int` (intruder speed, ft/s)

The script uses the same valid ranges/scaling constants as the simulator.

## High-level build pipeline

1. Load 5 ONNX models:
   - `ACASXU_run2a_1_1_batch_2000.onnx` ... `ACASXU_run2a_5_1_batch_2000.onnx`
2. Create bins (edges + centers) for each continuous dimension:
   - `rho`, `theta`, `psi`, `v_own`, `v_int`
3. For every `(last_cmd, rho_bin, theta_bin, psi_bin)` slice:
   - Evaluate all `(v_own_bin, v_int_bin)` center pairs
   - Run ONNX, get Q-values
   - Store `argmin(Q)` into `action_table`
   - Optionally store full `Q` vector in `q_table`
4. Save compressed NPZ file.

## Normalization before ONNX inference

Before inference, each 5D input is normalized:

`x_norm[i] = (x[i] - means[i]) / ranges[i]`

using:

- means: `[19791.091, 0.0, 0.0, 650.0, 600.0]`
- ranges: `[60261.0, 2π, 2π, 1100.0, 1200.0]`

This matches how `acasxu_dubins.py` evaluates the network.

## What gets saved

The output `.npz` contains:

- `action_table`: `uint8` table of advisories (always present)
- `q_table`: full Q-values (only if `--store-q-values` is used)
- bin edge arrays:
  - `rho_edges`
  - `theta_edges`
  - `psi_edges`
  - `v_own_edges`
  - `v_int_edges`
- `advisory_names`

The advisory mapping is:

- `0`: clear-of-conflict
- `1`: weak-left
- `2`: weak-right
- `3`: strong-left
- `4`: strong-right

## How query works

`query` does not run ONNX. It:

1. Loads the `.npz` file.
2. Quantizes each continuous value into a bin index with `searchsorted`.
3. Indexes `action_table[last_cmd, i_rho, i_theta, i_psi, i_v_own, i_v_int]`.
4. Prints advisory and bin ranges.
5. If `q_table` exists, prints the Q-vector too.

## Why output can feel “chunky”

The table is piecewise-constant. Small input changes that stay in the same bin return the same advisory; crossing a bin boundary can change advisory abruptly. Higher bin counts improve fidelity but increase build time and file size.

## Common commands

Build:

```bash
python3 build_lookup_table.py build --output acasxu_lookup_table.npz
```

Build with full Q-values:

```bash
python3 build_lookup_table.py build --output acasxu_lookup_table.npz --store-q-values
```

Query one state:

```bash
python3 build_lookup_table.py query \
  --table-path acasxu_lookup_table.npz \
  --last-cmd 0 \
  --rho 30000 \
  --theta 0.1 \
  --psi -0.2 \
  --v-own 800 \
  --v-int 500
```

## Practical tuning guidance

- Start with default bins to validate workflow.
- Increase bins gradually in the dimensions you care about most (`rho`, `theta`, `psi` usually matter most for geometry).
- Use `--store-q-values` when you want to analyze margins/hysteresis offline.
