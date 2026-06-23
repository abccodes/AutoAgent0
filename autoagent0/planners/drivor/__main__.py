"""DrivoR planner subprocess entry point.

    python -m autoagent0.planners.drivor --output <dir>
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from autoagent0.planners.base import run_subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description="DrivoR planner subprocess")
    parser.add_argument("--output", required=True, help="HUGSIM output directory containing FIFO pipes")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    # Import after arg parse so DRIVOR_REPO_ROOT-dependent imports happen lazily.
    from autoagent0.planners.drivor.planner import DrivorPlanner

    service = DrivorPlanner(output_dir)
    logger = logging.getLogger("drivor_planner")
    try:
        service.setup()
    except Exception:
        logger.exception("Failed to set up DrivoR planner")
        return 1
    return run_subprocess(service, output_dir, logger=logger)


if __name__ == "__main__":
    sys.exit(main())
