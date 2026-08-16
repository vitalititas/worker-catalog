"""Minimal RunPod wrapper around a pinned vLLM server.

The image owns the vLLM version.  Runtime configuration stays in the endpoint
template; a baked image writes /baked-model-path and never downloads weights
after a worker starts.
"""
from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, AsyncGenerator

import aiohttp

HOST = os.environ.get("VLLM_HOST", "127.0.0.1")
PORT = os.environ.get("VLLM_PORT", "8000")
BASE = f"http://{HOST}:{PORT}"
PROCESS: subprocess.Popen[str] | None = None


def _truth(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def vllm_command() -> list[str]:
    baked = Path("/baked-model-path")
    model = baked.read_text().strip() if baked.exists() else os.environ.get("MODEL_NAME", "")
    if not model:
        raise RuntimeError("MODEL_NAME is required when the image has no baked model")
    argv = ["vllm", "serve", model, "--host", HOST, "--port", PORT]
    values = {
        "MAX_MODEL_LEN": "--max-model-len",
        "GPU_MEMORY_UTILIZATION": "--gpu-memory-utilization",
        "KV_CACHE_DTYPE": "--kv-cache-dtype",
        "TOOL_CALL_PARSER": "--tool-call-parser",
        "QUANTIZATION": "--quantization",
        "OPENAI_SERVED_MODEL_NAME_OVERRIDE": "--served-model-name",
        "TENSOR_PARALLEL_SIZE": "--tensor-parallel-size",
    }
    for env, flag in values.items():
        if value := os.environ.get(env):
            argv.extend((flag, value))
    for env, flag in {
        "ENABLE_AUTO_TOOL_CHOICE": "--enable-auto-tool-choice",
        "ENABLE_CHUNKED_PREFILL": "--enable-chunked-prefill",
        "ENABLE_PREFIX_CACHING": "--enable-prefix-caching",
        "TRUST_REMOTE_CODE": "--trust-remote-code",
    }.items():
        if _truth(env):
            argv.append(flag)
    argv.extend(shlex.split(os.environ.get("VLLM_EXTRA_ARGS", "")))
    return argv


async def wait_ready() -> None:
    deadline = time.monotonic() + int(os.environ.get("VLLM_STARTUP_TIMEOUT", "1200"))
    while time.monotonic() < deadline:
        if PROCESS and PROCESS.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup ({PROCESS.returncode})")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{BASE}/health", timeout=5) as response:
                    if response.status == 200:
                        return
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(2)
    raise RuntimeError("vLLM health check timed out")


def normalize(job_input: dict[str, Any]) -> tuple[str, str, dict[str, Any] | None]:
    if "openai_input" in job_input:
        return job_input.get("openai_route", "/v1/chat/completions"), "POST", job_input["openai_input"]
    if "route" in job_input:
        body = job_input.get("body")
        return job_input["route"], job_input.get("method", "POST" if body else "GET").upper(), body
    if "messages" in job_input:
        return "/v1/chat/completions", "POST", job_input
    raise ValueError("input requires openai_input, route, or messages")


async def handler(job: dict[str, Any]) -> AsyncGenerator[Any, None]:
    try:
        route, method, body = normalize(job.get("input") or {})
        timeout = aiohttp.ClientTimeout(total=float(os.environ.get("REQUEST_TIMEOUT", "3600")))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(method, f"{BASE}{route}", json=body) as response:
                if response.status >= 400:
                    yield {"error": f"vLLM HTTP {response.status}: {await response.text()}"}
                    return
                yield await response.json(content_type=None)
    except Exception as exc:
        yield {"error": str(exc)}


def stop(_signum: int, _frame: Any) -> None:
    if PROCESS and PROCESS.poll() is None:
        PROCESS.terminate()
    raise SystemExit(0)


def main() -> None:
    global PROCESS
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    PROCESS = subprocess.Popen(vllm_command())
    asyncio.run(wait_ready())
    import runpod
    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": lambda _n: int(os.environ.get("MAX_CONCURRENCY", "1")),
        "return_aggregate_stream": True,
    })


if __name__ == "__main__":
    main()
