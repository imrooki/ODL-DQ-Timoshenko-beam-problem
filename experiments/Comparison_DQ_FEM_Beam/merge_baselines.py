


import csv
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DQ_CSV = os.path.join(RESULTS, "dq_baseline.csv")
FEM_CSV = os.path.join(RESULTS, "fem_baseline.csv")
ODIL_JSON = os.path.join(RESULTS, "odil", "sweep_index.json")
OUT_DIR = os.path.join(RESULTS, "comparison")
OUT_CSV = os.path.join(OUT_DIR, "q2_comparison.csv")


CONV_TOL = 1.0e-6

COLUMNS = ["method", "scenario", "init", "w_linear", "w_nonlinear",
           "nl_residual", "iters", "converged", "elapsed_s", "source"]


ODIL_OPT_SHORT = {
    "levenberg-marquardt": "odil-lm",
    "gauss-newton": "odil-gn",
    "lbfgs": "odil-lbfgs",
}


def _num(x):
    
    if x is None or x == "":
        return ""
    return repr(float(x))


def _row(method, scenario, init, w_lin, w_nl, res, iters, conv, t, source):
    return {
        "method": method,
        "scenario": scenario,
        "init": init,
        "w_linear": _num(w_lin),
        "w_nonlinear": _num(w_nl),
        "nl_residual": _num(res),
        "iters": "" if iters is None or iters == "" else int(iters),
        "converged": "" if conv is None else bool(conv),
        "elapsed_s": _num(t),
        "source": source,
    }


def collect_dq(rows):
    with open(DQ_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(_row(
                method="dq-" + r["method"],          
                scenario=r["scenario"],
                init=r["init_guess"],
                w_lin=None,                           
                w_nl=r["w_mid"],
                res=None,                             
                iters=r["iters"],
                conv=(str(r["converged"]).strip().lower() == "true"),
                t=r["elapsed_s"],
                source="dq_baseline.csv",
            ))


def collect_fem(rows):
    with open(FEM_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            res = float(r["nl_final_res"])
            rows.append(_row(
                method="fem-newton",
                scenario=r["scenario"],
                init=r.get("init_strategy", r.get("init_guess", "")),
                w_lin=r["w_linear"],
                w_nl=r["w_nonlinear"],
                res=res,
                iters=None,                           
                conv=(res < CONV_TOL),                
                t=r["elapsed_s"],
                source="fem_baseline.csv",
            ))


def collect_odil(rows):
    with open(ODIL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    for run in data.get("runs", []):
        method = ODIL_OPT_SHORT.get(run["optimizer"], "odil-" + run["optimizer"])
        scenario = run["scenario"]
        
        w_lin_mag = run.get("linear", {}).get("w_max")    
        w_lin = None if w_lin_mag is None else -float(w_lin_mag)
        by_strat = run.get("nonlinear_by_strategy", {})
        for strat in ("linear", "random"):
            nl = by_strat.get(strat)
            if nl is None:
                continue
            res = nl.get("R_PDE_norm")
            w_nl_mag = nl.get("w_max")
            rows.append(_row(
                method=method,
                scenario=scenario,
                init=strat,
                w_lin=w_lin,
                w_nl=None if w_nl_mag is None else -float(w_nl_mag),
                res=res,                              
                iters=nl.get("iterations"),
                conv=(res is not None and float(res) < CONV_TOL),
                t=nl.get("elapsed_s"),
                source="odil/sweep_index.json",
            ))


def main():
    missing = [p for p in (DQ_CSV, FEM_CSV, ODIL_JSON) if not os.path.isfile(p)]
    if missing:
        print("[ERROR] 缺少输入文件, 无法合并:")
        for p in missing:
            print("   -", p)
        print("  请先运行 one_start.py (或 run_dq_newton.py / run_fem_newton.py / solver/odil/run_odil.py)。")
        return 1

    rows = []
    collect_dq(rows)
    collect_fem(rows)
    collect_odil(rows)

    
    method_order = {m: i for i, m in enumerate(
        ["dq-newton", "dq-picard", "fem-newton", "odil-lm", "odil-gn", "odil-lbfgs"])}
    init_order = {m: i for i, m in enumerate(["zero", "linear", "gauss", "random"])}
    rows.sort(key=lambda r: (
        r["scenario"],
        method_order.get(r["method"], 99),
        init_order.get(r["init"], 99),
    ))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    n_conv = sum(1 for r in rows if r["converged"] is True)
    print("=" * 72)
    print("Q2 三方法统一对比表已生成")
    print("=" * 72)
    print(f"  输出: {OUT_CSV}")
    print(f"  总行数: {len(rows)}  (DQ 16 + FEM 8 + ODIL 12)")
    print(f"  收敛行: {n_conv} / {len(rows)}  (converged==True)")
    print("  提示: nl_residual 各方法定义不同 (FEM 弱式 / ODIL 强式 / DQ 未记录),")
    print("        仅用于收敛判别; 跨方法比较请看 w_linear / w_nonlinear。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
