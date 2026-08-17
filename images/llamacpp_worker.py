"""Baked-GGUF llama.cpp RunPod worker; it never downloads weights at runtime."""
from __future__ import annotations

import os
import subprocess
import time
import sys
from pathlib import Path

import requests


PORT = 8080
MODEL_PATH = Path("/baked-gguf-path")
ALIAS = os.environ.get("MODEL_ALIAS", "model")
CTX = os.environ.get("CTX", "32768")


def boot() -> subprocess.Popen[bytes]:
    if not MODEL_PATH.exists():
        raise RuntimeError("baked GGUF marker missing; runtime downloads are prohibited")
    model = Path(MODEL_PATH.read_text().strip())
    if not model.is_file() or model.stat().st_size < 1 << 30:
        raise RuntimeError("baked GGUF is missing or too small")
    command = [
        "/app/llama-server", "-m", str(model), "--alias", ALIAS, "-c", CTX,
        "-ngl", "99", "--host", "127.0.0.1", "--port", str(PORT), "--flash-attn", "--jinja",
        *os.environ.get("LLAMA_EXTRA_ARGS", "").split(),
    ]
    print(f"llama-worker: launching model={model} ctx={CTX} command={command!r}", file=sys.stderr, flush=True)
    proc = subprocess.Popen(command)
    deadline = time.monotonic() + int(os.environ.get("BOOT_TIMEOUT", "1200"))
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            print(f"llama-worker: llama-server exited rc={proc.returncode}", file=sys.stderr, flush=True)
            raise RuntimeError(f"llama-server exited during startup ({proc.returncode})")
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                print("llama-worker: llama-server healthy", file=sys.stderr, flush=True)
                return proc
        except requests.RequestException:
            pass
        time.sleep(2)
    proc.terminate()
    raise RuntimeError("llama-server health check timed out")


PROCESS = boot()


def handler(job: dict) -> dict:
    payload = dict(job.get("input") or {})
    path = payload.pop("path", "/v1/chat/completions")
    payload.setdefault("model", ALIAS)
    try:
        response = requests.post(f"http://127.0.0.1:{PORT}{path}", json=payload, timeout=600)
        return response.json()
    except ValueError:
        return {"error": {"message": f"llama-server HTTP {response.status_code}: {response.text[:2000]}"}}


import runpod
runpod.serverless.start({"handler": handler})
