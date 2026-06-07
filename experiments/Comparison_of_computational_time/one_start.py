"""
one-key launcher.

Runs the full PINN-vs-ODIL benchmark pipeline in sequence:

    Step 1.  pinn/run_pinn.py     -- 2 scenarios x 2 optimizers = 4 PINN runs
    Step 2.  odil/run_odil.py     -- 2 scenarios x 2 optimizers = 4 ODIL runs
    Step 3.  compare.py           -- aggregate, build comparison table & plots

Each step is launched as a subprocess inheriting the current Python
interpreter (so conda env `claude_test` is preserved). All stdout/stderr
is streamed to both the terminal in real time and to a per-step log
file under `results/logs/`. The launcher tracks per-step wall time and
prints a final summary.

Even if PINN fails, ODIL is still attempted (and vice versa); compare
runs at the end whenever at least one side has produced any output.
The exit code is non-zero iff any required step failed.

Usage:
    conda activate claude_test
    cd experiments/Comparison_of_computational_time
    python one_start.py

Optional flags:
    --skip-pinn     do not run PINN sweep (use existing pinn/ results)
    --skip-odil     do not run ODIL sweep (use existing odil/ results)
    --skip-compare  do not run compare.py
    --only <name>   shortcut for "skip everything except <name>",
                    where <name> in {pinn, odil, compare}
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


THIS_DIR = Path(__file__).resolve().parent
PINN_DIR = THIS_DIR / "pinn"
ODIL_DIR = THIS_DIR / "odil"
RESULTS_DIR = THIS_DIR / "results"
LOG_DIR = RESULTS_DIR / "logs"


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def _hr(char: str = "=", n: int = 78) -> str:
    return char * n


def banner(title: str, sub: str = "") -> None:
    print()
    print(_hr("="))
    print(f"  {title}")
    if sub:
        print(f"  {sub}")
    print(_hr("="))


def section(title: str) -> None:
    print()
    print(_hr("-"))
    print(f"  {title}")
    print(_hr("-"))


# ---------------------------------------------------------------------------
# Subprocess streaming with tee
# ---------------------------------------------------------------------------
def _stream_subprocess(cmd: List[str], cwd: Path, log_path: Path) -> int:
    """Run `cmd` in `cwd`; tee output to terminal and to `log_path`.

    Returns the subprocess return code.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # On Windows, force MKL/OpenMP to coexist (PINN code already does this,
    # but the launcher subprocess starts before Python imports run).
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # Ensure Python prints unbuffered so the tee captures lines as they
    # come, even when the child has not flushed.
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        # Header that lands in the log file but not the terminal
        logf.write(
            f"[one_start] cmd={cmd}\n"
            f"[one_start] cwd={cwd}\n"
            f"[one_start] timestamp={datetime.now().isoformat(timespec='seconds')}\n"
            f"[one_start] {'-' * 60}\n"
        )
        logf.flush()

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,            # line-buffered
                env=env,
            )
        except FileNotFoundError as exc:
            print(f"[one_start] FAILED to launch: {exc}")
            logf.write(f"[one_start] FAILED to launch: {exc}\n")
            return 127

        try:
            assert proc.stdout is not None  # for type checkers
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                logf.write(line)
            proc.wait()
        except KeyboardInterrupt:
            print("\n[one_start] KeyboardInterrupt - terminating subprocess.")
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
            raise

    return int(proc.returncode or 0)


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------
def _run_step(name: str, cmd: List[str], cwd: Path) -> Dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{name}_{timestamp}.log"

    section(f"STEP: {name}")
    print(f"  Command : {' '.join(str(c) for c in cmd)}")
    print(f"  CWD     : {cwd}")
    print(f"  Log file: {log_path}")
    print()

    t0 = time.time()
    rc = _stream_subprocess(cmd, cwd, log_path)
    elapsed = time.time() - t0

    status = "OK" if rc == 0 else f"FAILED (rc={rc})"
    print()
    print(f"  Step {name!r} finished: {status}   "
          f"elapsed = {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print(f"  Log saved to: {log_path}")

    return {
        "name": name,
        "rc": rc,
        "ok": (rc == 0),
        "elapsed_s": elapsed,
        "log": str(log_path),
    }


def step_pinn() -> Dict:
    cmd = [sys.executable, "run_pinn.py"]
    return _run_step("pinn", cmd, PINN_DIR)


def step_odil() -> Dict:
    cmd = [sys.executable, "run_odil.py"]
    return _run_step("odil", cmd, ODIL_DIR)


def step_compare() -> Dict:
    cmd = [sys.executable, "compare.py"]
    return _run_step("compare", cmd, THIS_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="one-key launcher (PINN sweep + ODIL sweep + compare).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--skip-pinn", action="store_true",
                   help="Skip PINN sweep (use existing results).")
    p.add_argument("--skip-odil", action="store_true",
                   help="Skip ODIL sweep (use existing results).")
    p.add_argument("--skip-compare", action="store_true",
                   help="Skip compare.py.")
    p.add_argument("--only", choices=["pinn", "odil", "compare"], default=None,
                   help="Run only one step (shortcut).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # --only short-circuits the skip flags
    if args.only is not None:
        args.skip_pinn = (args.only != "pinn")
        args.skip_odil = (args.only != "odil")
        args.skip_compare = (args.only != "compare")

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    banner(
        "one-key launcher",
        sub=f"PINN sweep + ODIL sweep + compare   "
            f"(timestamp: {datetime.now():%Y-%m-%d %H:%M:%S})",
    )
    print(f"  Project root : {THIS_DIR}")
    print(f"  Python       : {sys.executable}")
    print(f"  Logs root    : {LOG_DIR}")
    print(f"  Steps        : "
          f"PINN={'SKIP' if args.skip_pinn else 'RUN'}, "
          f"ODIL={'SKIP' if args.skip_odil else 'RUN'}, "
          f"compare={'SKIP' if args.skip_compare else 'RUN'}")

    pipeline_t0 = time.time()
    results: List[Dict] = []

    # ---------------------- Step 1: PINN ----------------------
    if args.skip_pinn:
        section("STEP: pinn (SKIPPED)")
        results.append({"name": "pinn", "ok": True, "skipped": True,
                        "elapsed_s": 0.0, "log": None, "rc": 0})
    else:
        results.append(step_pinn())

    # ---------------------- Step 2: ODIL ----------------------
    # Run ODIL even if PINN failed (independent computation; helpful for
    # partial debugging)
    if args.skip_odil:
        section("STEP: odil (SKIPPED)")
        results.append({"name": "odil", "ok": True, "skipped": True,
                        "elapsed_s": 0.0, "log": None, "rc": 0})
    else:
        results.append(step_odil())

    # ---------------------- Step 3: compare ----------------------
    if args.skip_compare:
        section("STEP: compare (SKIPPED)")
        results.append({"name": "compare", "ok": True, "skipped": True,
                        "elapsed_s": 0.0, "log": None, "rc": 0})
    else:
        results.append(step_compare())

    pipeline_elapsed = time.time() - pipeline_t0

    # ---------------------- Final summary ----------------------
    banner("Pipeline summary", sub=f"total wall time: "
                                    f"{pipeline_elapsed:.1f} s "
                                    f"({pipeline_elapsed / 60:.1f} min)")
    print(f"  {'Step':<10}  {'Status':<10}  {'Elapsed':<14}  Log")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 14}  {'-' * 50}")
    for r in results:
        if r.get("skipped"):
            status = "SKIPPED"
            elapsed = "-"
        else:
            status = "OK" if r.get("ok") else f"FAIL(rc={r.get('rc')})"
            elapsed = f"{r['elapsed_s']:.1f} s ({r['elapsed_s']/60:.1f} min)"
        log_str = r.get("log") or "-"
        print(f"  {r['name']:<10}  {status:<10}  {elapsed:<14}  {log_str}")

    # Hint to user about where the headline outputs landed
    print()
    print("Headline outputs (when present):")
    print(f"  - {RESULTS_DIR / 'comparison_table.csv'}")
    print(f"  - {RESULTS_DIR / 'comparison_table.md'}")
    print(f"  - {RESULTS_DIR / 'plots'}")

    failed = [r for r in results if not r.get("skipped") and not r.get("ok")]
    if failed:
        print()
        print(f"[!] {len(failed)} step(s) failed: "
              f"{', '.join(r['name'] for r in failed)}. See logs above.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
