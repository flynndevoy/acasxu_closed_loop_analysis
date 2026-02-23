Contains two files that do random simulations on the closed loop acasxu system with dubins car dynamics: `acasxu_dubins.py` and `parallel_acasxu_dubins.py`

On my system, I can run 10000 simulations in about 12 seconds single-threaded (1.2 ms per sim), and about 2 seconds multi-threaded (0.2 ms per sim). This also uses numba to speed things up using jit decortors.

## Lookup table tool

You can precompute a discretized lookup table from the ONNX networks:

```bash
python3 build_lookup_table.py build --output acasxu_lookup_table.npz
```

Then query a specific state without running online neural network inference:

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
