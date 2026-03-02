# ACASXu Dubins: Original Code and Modifications

## Brief Summary of the Original Code

The original `acasxu_dubins.py` implementation is a closed-loop ACASXu simulation with Dubins-style turn dynamics for ownship and intruder. It is intentionally simplified for fast policy stress testing, not high-fidelity flight or avionics realism. It:

- loads the 5 ONNX ACASXu networks and queries advisories online,
- propagates aircraft state with fixed-step dynamics,
- generates randomized initial encounters from a seed,
- simulates one encounter and tracks minimum separation,
- optionally plots the resulting trajectory.

### Why this baseline is highly simplified

- It is a 2D horizontal-plane model only (no vertical dynamics, climb/descent, or altitude logic).
- Aircraft kinematics are reduced to constant-speed Dubins-style motion with a small discrete turn-command set.
- Advisory selection is a direct `argmin` over network Q-values, without richer pilot/automation interaction layers.
- The simulation uses fixed discrete time steps (`dt`) and synchronous decision updates.
- Encounter generation is seed-based random sampling, not structured traffic scenarios or operational procedures.
- Intruder behavior is simple (straight or sampled turning commands), without intent-aware or reactive behavior models.
- The baseline assumes idealized execution (no actuator lag, no pilot delay/compliance model, minimal control constraints).
- Sensor/track uncertainty is not part of the original baseline pipeline.
- Outcome metrics are simplified proxies (for example, minimum distance and threshold-based conflict proxies), not full safety-case metrics.

## Modifications You Added

Based on this branch's commit history and current working-tree changes, your updates include:

1. CLI and simulation usability improvements
- Added command-line controls for fixed seed selection and enabling intruder turning.
- Added `--save-mp4` support for exporting animations.
- Increased default simulation volume in the parallel workflow and improved progress/display output.

2. Scenario generation and realism changes
- Randomized ownship and intruder speeds over broader valid ranges.
- Added support for intruder maneuver command sequences (instead of always straight flight when enabled).

3. Advisory logic stabilization and control
- Avoided reissuing unchanged advisories.
- Added advisory dwell-time gating (`--min-dwell-time`).
- Added Q-value hysteresis margin (`--q-hysteresis-margin`).
- Added continuity bias against oscillatory opposite-direction switching (`--continuity-bias`).
- Added direct-reversal controls:
  - optional hard blocking (`--restrict-direct-reversal`),
  - extra margin requirement (`--direct-reversal-margin`),
  - counters for proposed vs blocked direct reversals.

4. Fault-injection and degraded-observation behavior
- Added system fault time control (`--fault-time`) to force COC after a configured time.
- Added camera/tracker-driven observation mode (`--camera-mode`) with:
  - configurable camera rate, dropout, and position noise,
  - EMA-based track estimation,
  - measurement-health debouncing (staleness, speed, speed-jump checks),
  - measurement fault tracking and reporting.

5. Expanded metrics and evaluation outputs
- Added richer per-simulation metrics:
  - first alert timing/range,
  - alert-active duration,
  - advisory change and reversal counts,
  - direct-reversal proposal/block counts,
  - measurement fault steps.
- Added aggregate performance summarization across batch runs.
- Added classification thresholds and controls for NMAC/false-alert/nuisance-alert analysis:
  - `--nmac-distance`,
  - `--false-alert-distance`,
  - `--nuisance-max-alert-time`.

6. Start-state sampling workflow updates
- Added `"min"` seed mode to search over many seeds and report worst-case minimum distance.
- Added stratified seed sampling with target turning-advisory ratio:
  - `--num-sims`,
  - `--turning-ratio`,
  - `--max-sampling-attempts`.

7. Lookup-table tooling
- Added `build_lookup_table.py` to precompute/query a discretized advisory table from ONNX outputs.
- Added/used `acasxu_lookup_table.npz` for offline advisory queries.
- Added supporting documentation (`BUILD_LOOKUP_TABLE_EXPLAINED.md`).

## Notes

- This summary is derived from `git log` entries for `acasxu_dubins.py` plus current uncommitted edits visible in the working tree.
- If you want, this file can be split into:
  - "Committed Changes" vs
  - "Current Uncommitted Changes"
  with exact commit hashes under each item.
