"""RAP planner subprocess entry point.

Launched by ``rap/launch.sh`` (which sets up the RAP environment) as::

    python -m autoagent0.planners.rap --output <dir>
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from autoagent0.planners.base import run_subprocess
from autoagent0.planners.rap.planner import RAPPlanner


def main() -> int:
    parser = argparse.ArgumentParser(description="RAP planner subprocess")
    parser.add_argument("--output", required=True, help="HUGSIM output directory containing FIFO pipes")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    service = RAPPlanner(output_dir)
    logger = logging.getLogger("rap_planner")
    try:
        service.setup()
    except Exception:
        logger.exception("Failed to set up RAP planner")
        return 1
    return run_subprocess(service, output_dir, logger=logger)


if __name__ == "__main__":
    sys.exit(main())
