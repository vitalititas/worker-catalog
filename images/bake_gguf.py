#!/usr/bin/env python3
"""Bake one immutable public GGUF file into the llama.cpp worker image."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text())
    if not spec.get("bake"):
        return
    model, revision, filename = spec.get("model", ""), spec.get("revision", ""), spec.get("gguf_file", "")
    if not isinstance(model, str) or model.count("/") != 1:
        raise ValueError("GGUF bake model must be a public org/model ID")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("GGUF bake revision must be an immutable 40-character SHA")
    if not isinstance(filename, str) or "/" in filename or not filename.endswith(".gguf"):
        raise ValueError("GGUF bake filename must be a top-level .gguf filename")
    Path("/models").mkdir(exist_ok=True)
    path = hf_hub_download(model, filename=filename, revision=revision, local_dir="/models")
    if Path(path).stat().st_size < 1 << 30:
        raise ValueError("GGUF bake file is unexpectedly smaller than 1GiB")
    Path("/baked-gguf-path").write_text(path)


if __name__ == "__main__":
    main()
