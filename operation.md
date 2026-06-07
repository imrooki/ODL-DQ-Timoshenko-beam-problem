# Operation Guide

Set up the environment once (Python 3.13; see [`requirements.txt`](requirements.txt)):

```bash
pip install -r requirements.txt
```

All paths below are relative to this folder (the one that holds `README.md` and `main.py`).

---

## 1. Single-case solve — `main.py`

Edit `params.py` to configure the case (`method`, `optim_name`, `mode`, `bc_type`,
`distr_type`, load `q`, foundation `k1`/`k2`, ...). Then, from this folder:

```bash
python main.py
```

Outputs are written to `results/main/<bc_type>/<distribution>/<parameter-combo>/`:

- `data/`   — displacement CSV (`linear_u/w/phi`, `nonlinear_u/w/phi`)
- `models/` — best-loss checkpoints (linear & nonlinear)
- `plots/`  — deflection comparison and loss curves
- `logs/`   — training logs

---

## 2. ODIL-vs-PINN comparison — `one_start.py`

```bash
cd experiments/Comparison_of_computational_time
python one_start.py
```

Runs the full pipeline in order: **PINN sweep → ODIL sweep → compare**. Each step runs as a
subprocess; its output is streamed to the terminal and saved to a per-step log.

Outputs (under `experiments/Comparison_of_computational_time/results/`):

- `comparison_table.csv`, `comparison_table.md` — the headline comparison table
- `plots/`  — comparison figures
- `logs/`   — one timestamped log per step

Optional flags:

| Flag | Effect |
|------|--------|
| `--skip-pinn` | skip the PINN sweep (reuse existing `pinn/` results) |
| `--skip-odil` | skip the ODIL sweep (reuse existing `odil/` results) |
| `--skip-compare` | skip the aggregation step |
| `--only pinn` \| `odil` \| `compare` | run only that one step |

Example — re-aggregate the table/plots without recomputing:

```bash
python one_start.py --only compare
```
