"""
Build and query a discretized lookup table for ACAS Xu advisories.

This lets you inspect advisory decisions without running neural network inference online.
"""

import argparse
import os
import time

import numpy as np
import onnxruntime as ort


MEANS_FOR_SCALING = np.array(
    [19791.091, 0.0, 0.0, 650.0, 600.0],
    dtype=np.float32,
)
RANGE_FOR_SCALING = np.array(
    [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0],
    dtype=np.float32,
)

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


def get_onnx_path(last_command):
    """Get network filename path for a given previous advisory."""

    dirname = os.path.dirname(__file__)
    name = f"ACASXU_run2a_{last_command + 1}_1_batch_2000.onnx"
    return os.path.join(dirname, name)


def load_sessions():
    """Load all 5 ACAS Xu ONNX sessions."""

    sessions = []

    for last_cmd in range(5):
        path = get_onnx_path(last_cmd)
        sessions.append(ort.InferenceSession(path))

    return sessions


def make_edges(vmin, vmax, bins):
    """Create bin edges and centers for one dimension."""

    edges = np.linspace(vmin, vmax, bins + 1, dtype=np.float32)
    centers = (edges[:-1] + edges[1:]) * 0.5

    return edges, centers


def normalize_features(features):
    """Normalize features with ACAS Xu input scaling."""

    out = features.astype(np.float32, copy=True)
    out -= MEANS_FOR_SCALING
    out /= RANGE_FOR_SCALING

    return out


def session_supports_batch(session):
    """Check whether ONNX model accepts batch dimension > 1."""

    input_name = session.get_inputs()[0].name
    probe = np.zeros((2, 1, 1, 5), dtype=np.float32)

    try:
        session.run(None, {input_name: probe})
        return True
    except Exception:
        return False


def run_q_values(session, features, batch_ok):
    """Run session and return q-values for shape (N, 5) feature array."""

    input_name = session.get_inputs()[0].name
    scaled = normalize_features(features)

    if batch_ok:
        net_input = scaled.reshape((scaled.shape[0], 1, 1, 5))
        outputs = session.run(None, {input_name: net_input})[0]
        return outputs.astype(np.float32, copy=False)

    out = np.empty((scaled.shape[0], 5), dtype=np.float32)

    for i in range(scaled.shape[0]):
        net_input = scaled[i].reshape((1, 1, 1, 5))
        out[i] = session.run(None, {input_name: net_input})[0][0]

    return out


def quantize_to_index(val, edges):
    """Quantize a value to a bin index for given edges."""

    idx = int(np.searchsorted(edges, val, side="right") - 1)
    idx = max(0, idx)
    idx = min(idx, len(edges) - 2)
    return idx


def build_table(args):
    """Build lookup table from ONNX networks and save it as NPZ."""

    sessions = load_sessions()
    batch_ok = [session_supports_batch(s) for s in sessions]
    print(f"Batch inference support per net: {batch_ok}")

    rho_edges, rho_centers = make_edges(RHO_MIN, RHO_MAX, args.rho_bins)
    theta_edges, theta_centers = make_edges(THETA_MIN, THETA_MAX, args.theta_bins)
    psi_edges, psi_centers = make_edges(PSI_MIN, PSI_MAX, args.psi_bins)
    v_own_edges, v_own_centers = make_edges(V_OWN_MIN, V_OWN_MAX, args.v_own_bins)
    v_int_edges, v_int_centers = make_edges(V_INT_MIN, V_INT_MAX, args.v_int_bins)

    shape = (
        5,
        len(rho_centers),
        len(theta_centers),
        len(psi_centers),
        len(v_own_centers),
        len(v_int_centers),
    )
    action_table = np.empty(shape, dtype=np.uint8)
    q_table = None

    if args.store_q_values:
        q_table = np.empty(shape + (5,), dtype=np.float32)

    total_outer = 5 * len(rho_centers) * len(theta_centers) * len(psi_centers)
    done_outer = 0
    start = time.perf_counter()
    last_report = start

    for last_cmd, session in enumerate(sessions):
        print(f"Building slices for last_cmd={last_cmd} ({ADVISORY_NAMES[last_cmd]})")

        for i_rho, rho in enumerate(rho_centers):
            for i_theta, theta in enumerate(theta_centers):
                for i_psi, psi in enumerate(psi_centers):
                    n_points = len(v_own_centers) * len(v_int_centers)
                    features = np.empty((n_points, 5), dtype=np.float32)

                    idx = 0
                    for v_own in v_own_centers:
                        for v_int in v_int_centers:
                            features[idx, 0] = rho
                            features[idx, 1] = theta
                            features[idx, 2] = psi
                            features[idx, 3] = v_own
                            features[idx, 4] = v_int
                            idx += 1

                    q_vals = run_q_values(session, features, batch_ok[last_cmd])
                    actions = np.argmin(q_vals, axis=1).astype(np.uint8)
                    actions = actions.reshape((len(v_own_centers), len(v_int_centers)))

                    action_table[last_cmd, i_rho, i_theta, i_psi, :, :] = actions

                    if q_table is not None:
                        q_vals = q_vals.reshape((len(v_own_centers), len(v_int_centers), 5))
                        q_table[last_cmd, i_rho, i_theta, i_psi, :, :, :] = q_vals

                    done_outer += 1
                    now = time.perf_counter()

                    if now - last_report >= args.progress_sec:
                        elapsed = now - start
                        pct = 100.0 * done_outer / total_outer
                        print(
                            f"  progress: {done_outer}/{total_outer} outer cells "
                            f"({pct:.1f}%), elapsed={elapsed:.1f}s"
                        )
                        last_report = now

    save_kwargs = {
        "action_table": action_table,
        "rho_edges": rho_edges,
        "theta_edges": theta_edges,
        "psi_edges": psi_edges,
        "v_own_edges": v_own_edges,
        "v_int_edges": v_int_edges,
        "advisory_names": np.array(ADVISORY_NAMES),
    }
    if q_table is not None:
        save_kwargs["q_table"] = q_table

    np.savez_compressed(args.output, **save_kwargs)

    duration = time.perf_counter() - start
    size_mb = os.path.getsize(args.output) / (1024.0 * 1024.0)
    print(f"Saved lookup table to {args.output}")
    print(f"Build time: {duration:.1f}s, file size: {size_mb:.1f} MB")
    print(f"Action table shape: {action_table.shape}")
    if q_table is not None:
        print(f"Q table shape: {q_table.shape}")


def query_table(args):
    """Query one lookup cell and print advisory details."""

    data = np.load(args.table_path, allow_pickle=False)

    action_table = data["action_table"]
    has_q_table = "q_table" in data.files
    q_table = data["q_table"] if has_q_table else None
    rho_edges = data["rho_edges"]
    theta_edges = data["theta_edges"]
    psi_edges = data["psi_edges"]
    v_own_edges = data["v_own_edges"]
    v_int_edges = data["v_int_edges"]
    advisory_names = data["advisory_names"].tolist()

    if args.last_cmd < 0 or args.last_cmd > 4:
        raise ValueError("--last-cmd must be in [0, 4]")

    i_rho = quantize_to_index(args.rho, rho_edges)
    i_theta = quantize_to_index(args.theta, theta_edges)
    i_psi = quantize_to_index(args.psi, psi_edges)
    i_v_own = quantize_to_index(args.v_own, v_own_edges)
    i_v_int = quantize_to_index(args.v_int, v_int_edges)

    action = int(action_table[args.last_cmd, i_rho, i_theta, i_psi, i_v_own, i_v_int])

    print("Lookup query result")
    print(f"  last_cmd: {args.last_cmd} ({advisory_names[args.last_cmd]})")
    print(f"  rho={args.rho:.3f} -> bin {i_rho} [{rho_edges[i_rho]:.3f}, {rho_edges[i_rho+1]:.3f}]")
    print(
        f"  theta={args.theta:.6f} -> bin {i_theta} "
        f"[{theta_edges[i_theta]:.6f}, {theta_edges[i_theta+1]:.6f}]"
    )
    print(f"  psi={args.psi:.6f} -> bin {i_psi} [{psi_edges[i_psi]:.6f}, {psi_edges[i_psi+1]:.6f}]")
    print(
        f"  v_own={args.v_own:.3f} -> bin {i_v_own} "
        f"[{v_own_edges[i_v_own]:.3f}, {v_own_edges[i_v_own+1]:.3f}]"
    )
    print(
        f"  v_int={args.v_int:.3f} -> bin {i_v_int} "
        f"[{v_int_edges[i_v_int]:.3f}, {v_int_edges[i_v_int+1]:.3f}]"
    )
    print(f"  advisory: {action} ({advisory_names[action]})")

    if q_table is not None:
        q_vals = q_table[args.last_cmd, i_rho, i_theta, i_psi, i_v_own, i_v_int]
        print(f"  q-values: {np.array2string(q_vals, precision=6)}")


def main():
    """Parse arguments and run build/query command."""

    parser = argparse.ArgumentParser(
        description="Build or query ACAS Xu lookup table from ONNX networks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build and save a lookup table.")
    build_parser.add_argument(
        "--output",
        type=str,
        default="acasxu_lookup_table.npz",
        help="Output NPZ path.",
    )
    build_parser.add_argument("--rho-bins", type=int, default=11, help="Number of rho bins.")
    build_parser.add_argument("--theta-bins", type=int, default=13, help="Number of theta bins.")
    build_parser.add_argument("--psi-bins", type=int, default=13, help="Number of psi bins.")
    build_parser.add_argument("--v-own-bins", type=int, default=7, help="Number of ownship speed bins.")
    build_parser.add_argument("--v-int-bins", type=int, default=7, help="Number of intruder speed bins.")
    build_parser.add_argument(
        "--store-q-values",
        action="store_true",
        default=False,
        help="Store full Q-values (larger file). By default only advisories are stored.",
    )
    build_parser.add_argument(
        "--progress-sec",
        type=float,
        default=2.0,
        help="Progress print interval in seconds.",
    )

    query_parser = subparsers.add_parser("query", help="Query advisory for one state from a table.")
    query_parser.add_argument(
        "--table-path",
        type=str,
        default="acasxu_lookup_table.npz",
        help="Path to lookup table NPZ file.",
    )
    query_parser.add_argument("--last-cmd", type=int, required=True, help="Previous advisory in [0,4].")
    query_parser.add_argument("--rho", type=float, required=True, help="Distance rho in feet.")
    query_parser.add_argument("--theta", type=float, required=True, help="Relative angle theta in radians.")
    query_parser.add_argument("--psi", type=float, required=True, help="Heading angle psi in radians.")
    query_parser.add_argument("--v-own", type=float, required=True, help="Ownship speed in feet/sec.")
    query_parser.add_argument("--v-int", type=float, required=True, help="Intruder speed in feet/sec.")

    args = parser.parse_args()

    if args.command == "build":
        for name in ["rho_bins", "theta_bins", "psi_bins", "v_own_bins", "v_int_bins"]:
            if getattr(args, name) <= 0:
                parser.error(f"--{name.replace('_', '-')} must be positive")
        if args.progress_sec <= 0:
            parser.error("--progress-sec must be positive")
        build_table(args)
    else:
        query_table(args)


if __name__ == "__main__":
    main()
