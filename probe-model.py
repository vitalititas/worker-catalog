#!/usr/bin/env python3
"""Compatibility entry point for the old catalog probe.

The former implementation leaked GPU endpoints whenever create, poll, parsing,
or delete failed.  All probes now use catalog_worker.py's mandatory teardown.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    argv = sys.argv[1:]
    if "--keep" in argv:
        raise SystemExit("--keep was removed: disposable GPU probes must always tear down")
    if "--image" not in argv:
        argv.extend(("--image", "runpod/worker-v1-vllm:v2.25.1"))
    return subprocess.run([str(ROOT / "catalog_worker.py"), "test", *argv], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
