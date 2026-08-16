#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/catalog_worker.py" deploy "$@"
