#!/usr/bin/env python3
"""Bake exactly the public Hugging Face revision declared in bake-spec.json."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text())
    if not spec.get("bake"):
        return
    model = spec.get("model", "")
    revision = spec.get("revision", "")
    if not isinstance(model, str) or model.count("/") != 1:
        raise ValueError("bake-spec model must be a public org/model ID")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("bake-spec revision must be an immutable 40-character SHA")
    snapshot_download(model, revision=revision, local_dir="/models/model")
    Path("/baked-model-path").write_text("/models/model")


if __name__ == "__main__":
    main()
