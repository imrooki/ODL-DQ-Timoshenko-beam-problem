


import argparse
import os
import subprocess
import sys
import time


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))


STEPS = [
    ("dq",   "run_dq_newton.py",
     "Direct-DQ Newton + Picard（初值×方法扫描）-> results/dq_baseline.csv (+residual_history)"),
    ("fem",  "run_fem_newton.py",
     "弱式 FEM 标准 Newton (n_elem=1500) -> results/fem_baseline.csv (+residual_history)"),
    ("odil", os.path.join("solver", "odil", "run_odil.py"),
     "ODIL (proposed) LM/GN/L-BFGS × 2 场景 × {linear,random}初值 -> results/odil/<场景>/<优化器>/"),
    ("merge", "merge_baselines.py",
     "DQ+FEM+ODIL 三方法统一对比表（纯聚合, 不重跑求解器）-> results/comparison/q2_comparison.csv"),
]

BANNER = ("Q2 conventional baseline 总任务 —— Timoshenko/FSDT 梁弯曲 (C-C, ±弹性基础)\n"
          "DQ-Newton/Picard + 弱式 FEM + ODIL(proposed) + 三方法统一对比表")

HEAVY = {"fem", "odil"}   


def _run_step(name, rel_path, env):
    
    script = os.path.join(HERE, rel_path)
    if not os.path.isfile(script):
        print(f"[MISSING] {name}: 找不到 {script}")
        return {"name": name, "status": "MISSING", "seconds": 0.0}
    print("\n" + "=" * 78)
    print(f"  >> {name}    ({rel_path})")
    print("=" * 78, flush=True)
    t0 = time.perf_counter()
    try:
        rc = subprocess.run([sys.executable, script], cwd=HERE, env=env).returncode
    except KeyboardInterrupt:
        raise
    except Exception as exc:          
        dt = time.perf_counter() - t0
        print(f"[ERROR] {name}: {exc}")
        return {"name": name, "status": "ERROR", "seconds": dt}
    dt = time.perf_counter() - t0
    status = "OK" if rc == 0 else f"FAIL(rc={rc})"
    print(f"\n[{status}] {name}  ({dt:.1f}s)")
    return {"name": name, "status": status, "seconds": dt}


def main():
    names = [s[0] for s in STEPS]
    ap = argparse.ArgumentParser(description="Q2 conventional baseline 总任务一键执行")
    ap.add_argument("--only", default="",
                    help="只运行这些步骤(逗号分隔)，可选: " + ", ".join(names))
    ap.add_argument("--skip", default="", help="跳过这些步骤(逗号分隔)")
    ap.add_argument("--list", action="store_true", help="只列出步骤，不执行")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="某步失败即停止(默认继续跑完其余步骤)")
    args = ap.parse_args()

    if args.list:
        print("可用步骤(按执行顺序):")
        for n, p, desc in STEPS:
            print(f"  {n:8s} {p:24s} {desc}")
        return 0

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    skip = {x.strip() for x in args.skip.split(",") if x.strip()}
    bad = (only | skip) - set(names)
    if bad:
        print(f"未知步骤名: {', '.join(sorted(bad))}；可选: {', '.join(names)}")
        return 2
    plan = [s for s in STEPS if (not only or s[0] in only) and s[0] not in skip]
    if not plan:
        print("没有要执行的步骤(检查 --only/--skip)。")
        return 1

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"     

    print("=" * 78)
    print("  " + BANNER.replace("\n", "\n  "))
    print("  解释器: " + sys.executable)
    print("  计划: " + " -> ".join(s[0] for s in plan))
    if HEAVY & {s[0] for s in plan}:
        print("  注: fem / odil(L-BFGS) 可能较慢，可 Ctrl+C 中断或用 --skip。")
    print("=" * 78, flush=True)

    t_all = time.perf_counter()
    results = []
    for s in plan:
        r = _run_step(s[0], s[1], env)
        results.append(r)
        if args.stop_on_error and not str(r["status"]).startswith("OK"):
            print(f"\n[stop-on-error] 在 {r['name']} 处停止，跳过其余步骤。")
            break
    total = time.perf_counter() - t_all

    print("\n" + "=" * 78)
    print("  one_start 汇总")
    print("=" * 78)
    print(f"  {'step':10s}{'status':16s}{'time':>9s}")
    for r in results:
        print(f"  {r['name']:10s}{str(r['status']):16s}{r['seconds']:8.1f}s")
    n_ok = sum(1 for r in results if str(r["status"]).startswith("OK"))
    print("  " + "-" * 34)
    print(f"  合计 {total:.1f}s ，{n_ok}/{len(results)} 成功")
    print("  产物: results/*.csv (+*_residual_history.csv) 、 results/odil/ 、 results/comparison/q2_comparison.csv 。")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
