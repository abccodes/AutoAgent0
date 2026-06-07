#!/usr/bin/env python3
import os
import runpy


if __name__ == "__main__":
    target = os.path.join(os.path.dirname(__file__), "baselines", "common", "resolve_output_path.py")
    runpy.run_path(target, run_name="__main__")
