from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    case_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(case_dir.parent))

    from plot_depth_analysis_processed_impl import run_case

    run_case(case_dir)


if __name__ == "__main__":
    main()
