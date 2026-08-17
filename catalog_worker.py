#!/usr/bin/env python3
"""Sovereign Worker Catalog control plane.

`test` is intentionally destructive to the temporary resources it creates:
every exit path reconciles by its unique test name, sets workersMax=0, proves
the setting, deletes endpoints/templates, and waits for account spend to hit 0.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "catalog.jsonl"
CONNECTORS = Path.home() / ".legion" / "connectors.env"
REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
RUN = "https://api.runpod.ai/v2"
UA = "curl/8.5.0"
VLLM_VERSION = "v0.27.1"
DEFAULT_IMAGE = "ghcr.io/vitalititas/worker-catalog"

# These are deployment safety defaults, not performance claims. A model with
# raw BF16 weights must fit weights *and* KV cache; never inherit the 24GB
# worker default just because its image built successfully.
MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen2.5-Coder-7B-Instruct": {"gpu": "NVIDIA RTX A5000", "gpu_vram_gb": 24, "containerDiskInGb": 30, "max_model_len": 32768, "parser": "hermes"},
    "SWE-Model-Testing/SWE-UT-8B-Qwen3-Coder-Distill": {"gpu": "NVIDIA RTX A5000", "gpu_vram_gb": 24, "containerDiskInGb": 30, "max_model_len": 32768, "parser": "hermes"},
    "jica98/qwen3.5-4B-super-coder": {"gpu": "NVIDIA RTX A4000", "gpu_vram_gb": 16, "containerDiskInGb": 30, "max_model_len": 32768, "parser": "hermes"},
    "Qwen/Qwen2.5-Coder-14B-Instruct": {"gpu": "NVIDIA RTX A6000", "gpu_vram_gb": 48, "containerDiskInGb": 60, "max_model_len": 32768, "parser": "hermes"},
    "Qwen/Qwen2.5-Coder-32B-Instruct": {"gpu": "NVIDIA A100 80GB PCIe", "gpu_vram_gb": 80, "containerDiskInGb": 100, "max_model_len": 32768, "parser": "hermes"},
    "SvenBrnn/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-gptq-w4a16": {"gpu": "NVIDIA RTX A6000", "gpu_vram_gb": 48, "containerDiskInGb": 60, "max_model_len": 262144, "parser": "gemma4", "kv_cache_dtype": "fp8"},
    "mradermacher/Muse-Glimmer-30B-heretic-plus-GGUF": {"gpu": "NVIDIA RTX A6000", "gpu_vram_gb": 48, "containerDiskInGb": 60, "max_model_len": 32768, "parser": "llama.cpp"},
    "mradermacher/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated-GGUF": {"gpu": "NVIDIA RTX A6000", "gpu_vram_gb": 48, "containerDiskInGb": 60, "max_model_len": 32768, "parser": "llama.cpp"},
}


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:54]


def model_defaults(model: str) -> dict[str, Any]:
    return dict(MODEL_DEFAULTS.get(model, {"gpu": "NVIDIA RTX A5000", "gpu_vram_gb": 24, "containerDiskInGb": 30, "max_model_len": 32768, "parser": "hermes"}))


def key() -> str:
    for line in CONNECTORS.read_text().splitlines():
        if line.startswith("RUNPOD_API_KEY="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return value
    raise RuntimeError("RUNPOD_API_KEY missing from ~/.legion/connectors.env")


def request(url: str, method: str = "GET", body: dict[str, Any] | None = None, timeout: int = 90) -> Any:
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=payload, method=method)
    for name, value in (("Authorization", f"Bearer {key()}"), ("Content-Type", "application/json"), ("User-Agent", UA)):
        req.add_header(name, value)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw or b"{}")


def rest(method: str, path: str, body: dict[str, Any] | None = None, timeout: int = 90) -> Any:
    return request(f"{REST}{path}", method, body, timeout)


def gql(query: str) -> dict[str, Any]:
    data = request(GRAPHQL, "POST", {"query": query}, timeout=60)
    if data.get("errors"):
        raise RuntimeError(f"RunPod GraphQL error: {data['errors']}")
    return data.get("data") or {}


def account() -> dict[str, Any]:
    return gql("query { myself { clientBalance currentSpendPerHr spendLimit endpoints { id name workersMin workersMax idleTimeout templateId } } }")["myself"]


def assert_preflight() -> dict[str, Any]:
    me = account()
    if not isinstance(me.get("clientBalance"), (int, float)) or me["clientBalance"] <= 0:
        raise RuntimeError("funded RunPod account required")
    if me.get("currentSpendPerHr") != 0:
        raise RuntimeError(f"active account spend blocks a disposable test: {me.get('currentSpendPerHr')}")
    return me


def append(record: dict[str, Any]) -> None:
    record["ts"] = int(time.time())
    with CATALOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def resources(name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    endpoints = [item for item in rest("GET", "/endpoints") if item.get("name") == name]
    templates = [item for item in rest("GET", "/templates") if item.get("name") == f"{name}-template"]
    return endpoints, templates


def require_pod_check(config: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
        json.dump(config, handle)
        handle.flush()
        import subprocess
        subprocess.run(["pod-check", handle.name], check=True)


def endpoint_config(name: str, image: str, model: str, parser: str, gpu: str, disk: int, baked: bool, extra_env: dict[str, str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    defaults = model_defaults(model)
    env = {
        "MAX_MODEL_LEN": str(defaults["max_model_len"]),
        "GPU_MEMORY_UTILIZATION": "0.95",
        "KV_CACHE_DTYPE": str(defaults.get("kv_cache_dtype", "fp8")),
        "ENABLE_AUTO_TOOL_CHOICE": "true",
        "TOOL_CALL_PARSER": parser,
        "MAX_CONCURRENCY": "1",
    }
    if not baked and not extra_env:
        env["MODEL_NAME"] = model
    if extra_env:
        env.update(extra_env)
    template = {"name": f"{name}-template", "imageName": image, "isServerless": True, "containerDiskInGb": disk, "env": env}
    endpoint = {"name": name, "computeType": "GPU", "gpuTypeIds": [gpu], "gpuCount": 1, "workersMin": 0, "workersMax": 1, "idleTimeout": 10, "flashboot": True, "executionTimeoutMs": 900000, "scalerType": "QUEUE_DELAY", "scalerValue": 4}
    return template, endpoint


def _retry(action, description: str, attempts: int = 3) -> None:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            action()
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return
            error = exc
        except Exception as exc:  # cleanup must retain the last failure for escalation
            error = exc
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"cleanup failed: {description}: {error}")


def cleanup_ids(endpoint_ids: list[str], template_ids: list[str]) -> None:
    """The non-negotiable teardown order for disposable GPU test resources."""
    for endpoint_id in endpoint_ids:
        _retry(lambda eid=endpoint_id: rest("PATCH", f"/endpoints/{eid}", {"workersMax": 0}), f"workersMax=0 for {endpoint_id}")
        readback = rest("GET", f"/endpoints/{endpoint_id}")
        if readback.get("workersMax") != 0:
            raise RuntimeError(f"cleanup refused: {endpoint_id} workersMax read back as {readback.get('workersMax')}")
        _retry(lambda eid=endpoint_id: rest("DELETE", f"/endpoints/{eid}"), f"delete endpoint {endpoint_id}")
    for endpoint_id in endpoint_ids:
        try:
            rest("GET", f"/endpoints/{endpoint_id}")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
        else:
            raise RuntimeError(f"cleanup refused: endpoint {endpoint_id} remains")
    for template_id in template_ids:
        _retry(lambda tid=template_id: rest("DELETE", f"/templates/{tid}"), f"delete template {template_id}")
    for template_id in template_ids:
        try:
            rest("GET", f"/templates/{template_id}")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
        else:
            raise RuntimeError(f"cleanup refused: template {template_id} remains")
    for _ in range(6):
        if account().get("currentSpendPerHr") == 0:
            return
        time.sleep(10)
    raise RuntimeError("cleanup completed resource deletion but billing did not return to $0/hr")


def cleanup(name: str) -> None:
    endpoints, templates = resources(name)
    cleanup_ids([endpoint["id"] for endpoint in endpoints], [template["id"] for template in templates])


def job(endpoint_id: str, body: dict[str, Any], timeout: int = 900) -> dict[str, Any]:
    created = request(f"{RUN}/{endpoint_id}/run", "POST", body, timeout=60)
    job_id = created.get("id")
    if not job_id:
        raise RuntimeError(f"RunPod did not return a job id: {created}")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = request(f"{RUN}/{endpoint_id}/status/{job_id}", timeout=60)
        if current.get("status") in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
            return current
        time.sleep(10)
    raise TimeoutError(f"RunPod job {job_id} did not finish within {timeout}s")


def output_message(result: dict[str, Any]) -> dict[str, Any]:
    output = result.get("output")
    if isinstance(output, list) and output:
        output = output[-1]
    if isinstance(output, dict) and isinstance(output.get("choices"), list) and output["choices"]:
        message = output["choices"][0].get("message")
        if isinstance(message, dict):
            return message
    raise RuntimeError(f"unexpected worker output shape: {str(output)[:500]}")


def invoke_chat(endpoint_id: str, openai_input: dict[str, Any], protocol: str) -> dict[str, Any]:
    if protocol == "runpod-vllm":
        payload = {"input": {"openai_route": "/v1/chat/completions", "openai_input": openai_input}}
    elif protocol == "llama-openai":
        payload = {"input": openai_input}
    else:
        raise RuntimeError(f"unsupported worker protocol: {protocol}")
    return job(endpoint_id, payload)


def real_tool_test(endpoint_id: str, model: str, protocol: str) -> tuple[float, float]:
    tools = [
        {"type": "function", "function": {"name": "lookup_ticket", "description": "Fetch a support ticket by ID.", "parameters": {"type": "object", "properties": {"ticket_id": {"type": "string"}}, "required": ["ticket_id"], "additionalProperties": False}}},
        {"type": "function", "function": {"name": "lookup_user", "description": "Fetch a user by ID.", "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"], "additionalProperties": False}}},
    ]
    started = time.monotonic()
    first = invoke_chat(endpoint_id, {"model": model, "messages": [{"role": "user", "content": "Use the ticket tool to retrieve ticket CASE-731. Do not guess the ticket status."}], "tools": tools, "tool_choice": "required", "temperature": 0, "max_tokens": 160}, protocol)
    cold_s = time.monotonic() - started
    if first.get("status") != "COMPLETED":
        raise RuntimeError(f"tool-call request {first.get('status')}: {first.get('error')}")
    message = output_message(first)
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise RuntimeError(f"expected one structured tool call, got {calls!r}")
    call = calls[0].get("function") or {}
    if call.get("name") != "lookup_ticket":
        raise RuntimeError(f"wrong tool selected: {call.get('name')!r}")
    try:
        arguments = json.loads(call.get("arguments", ""))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"malformed tool arguments: {call.get('arguments')!r}") from exc
    if arguments != {"ticket_id": "CASE-731"}:
        raise RuntimeError(f"wrong tool arguments: {arguments!r}")
    messages = [{"role": "user", "content": "Use the ticket tool to retrieve ticket CASE-731. Do not guess the ticket status."}, message, {"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps({"ticket_id": "CASE-731", "status": "resolved"})}]
    started = time.monotonic()
    second = invoke_chat(endpoint_id, {"model": model, "messages": messages, "tools": tools, "temperature": 0, "max_tokens": 80}, protocol)
    warm_s = time.monotonic() - started
    if second.get("status") != "COMPLETED":
        raise RuntimeError(f"tool-result request {second.get('status')}: {second.get('error')}")
    text = output_message(second).get("content") or ""
    if "resolved" not in text.lower():
        raise RuntimeError(f"model did not use the tool result: {text!r}")
    return cold_s, warm_s


def latest(model: str) -> dict[str, Any]:
    records = [json.loads(line) for line in CATALOG.read_text().splitlines() if line.strip()]
    matches = [record for record in records if record.get("model") == model]
    if not matches:
        raise RuntimeError(f"no catalog record for {model!r}")
    return matches[-1]


def deploy_record(identity: str) -> dict[str, Any]:
    records = [json.loads(line) for line in CATALOG.read_text().splitlines() if line.strip()]
    exact = [record for record in records if record.get("entry") == identity]
    if exact:
        return exact[-1]
    proven = [record for record in records if record.get("model") == identity and record.get("serves") is True and record.get("tool_call_ok") is True]
    if not proven:
        raise RuntimeError("catalog entry is not tool-verified")
    return proven[-1]


def git(*args: str, cwd: Path = ROOT) -> str:
    completed = subprocess.run(["git", *args], cwd=cwd, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def publish_bake_branch(model: str, revision: str, engine: str, spec_file: str, spec: dict[str, Any]) -> str:
    """Publish a tiny immutable bake spec, never a local container layer."""
    if not shutil.which("git"):
        raise RuntimeError("git is required to publish a RunPod source-build branch")
    if not git("remote"):
        raise RuntimeError("configure an origin remote before publishing a RunPod source-build branch")
    if git("status", "--porcelain"):
        raise RuntimeError("commit or stash source changes before publishing a reproducible bake branch")
    branch = f"bake/{slug(model)}-{revision[:12]}-{engine}"
    temp_root = Path(tempfile.mkdtemp(prefix="worker-catalog-bake-"))
    try:
        exists = subprocess.run(["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"], cwd=ROOT).returncode == 0
        if exists:
            git("worktree", "add", str(temp_root), branch)
            git("rebase", "main", cwd=temp_root)
        else:
            git("worktree", "add", "--detach", str(temp_root), "HEAD")
            git("switch", "-c", branch, cwd=temp_root)
        (temp_root / spec_file).write_text(json.dumps(spec, indent=2) + "\n")
        git("add", spec_file, cwd=temp_root)
        if git("status", "--porcelain", cwd=temp_root):
            git("commit", "-m", f"build: bake {model} at {revision[:12]}", cwd=temp_root)
        git("push", "--force-with-lease", "origin", f"HEAD:refs/heads/{branch}", cwd=temp_root)
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(temp_root)], cwd=ROOT, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return branch


def test(args: argparse.Namespace) -> int:
    name = f"catalog-test-{slug(args.model)}-{uuid.uuid4().hex[:8]}"
    extra_env: dict[str, str] = {}
    if args.env_json:
        loaded = json.loads(Path(args.env_json).read_text())
        if not isinstance(loaded, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in loaded.items()):
            raise RuntimeError("--env-json must contain a JSON object of string environment values")
        extra_env = loaded
    record: dict[str, Any] = {"entry": args.entry or slug(args.model), "model": args.model, "image_tag": args.image, "parser": args.parser, "gpu": args.gpu, "baked": args.bake, "worker_protocol": args.worker_protocol, "test_name": name, "serves": False, "tool_call_ok": False}
    error: Exception | None = None
    old_handlers = {signal.SIGINT: signal.getsignal(signal.SIGINT), signal.SIGTERM: signal.getsignal(signal.SIGTERM)}
    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    for signum in old_handlers:
        signal.signal(signum, interrupt)
    try:
        assert_preflight()
        if args.endpoint_id:
            endpoint = rest("GET", f"/endpoints/{args.endpoint_id}")
            if not str(endpoint.get("name", "")).startswith("catalog-test-"):
                raise RuntimeError("refusing to tear down a source endpoint without the catalog-test- prefix")
            name = endpoint["name"]
            record["test_name"] = name
            template_id = endpoint.get("templateId")
            if not template_id:
                raise RuntimeError("source endpoint lacks a templateId; refusing an untraceable GPU test")
            endpoint_id = endpoint["id"]
        else:
            if not args.image:
                raise RuntimeError("--image is required unless testing the exact --endpoint-id created by RunPod source build")
            gate = {"name": name, "image": args.image, "model_baked": args.bake, "gpu_vram_gb": args.gpu_vram_gb, "workersMin": 0, "workersMax": 1, "idleTimeout": 10, "flashBootType": "FLASHBOOT", "spend_limit": 80}
            require_pod_check(gate)
            template_body, endpoint_body = endpoint_config(name, args.image, args.model, args.parser, args.gpu, args.disk, args.bake, extra_env)
            template = rest("POST", "/templates", template_body)
            template_id = template["id"]
            endpoint_body["templateId"] = template_id
            endpoint_id = rest("POST", "/endpoints", endpoint_body)["id"]
        readback = rest("GET", f"/endpoints/{endpoint_id}")
        if (readback.get("workersMin"), readback.get("workersMax"), readback.get("idleTimeout")) != (0, 1, 10):
            raise RuntimeError("scale-to-zero readback failed")
        cold_s, warm_s = real_tool_test(endpoint_id, args.model, args.worker_protocol)
        record.update(serves=True, tool_call_ok=True, cold_s=round(cold_s, 1), warm_s=round(warm_s, 1))
    except Exception as exc:
        error = exc
        record["error"] = str(exc)[:500]
    finally:
        try:
            if args.endpoint_id and 'endpoint_id' in locals() and 'template_id' in locals():
                cleanup_ids([endpoint_id], [template_id])
            else:
                cleanup(name)
            record["teardown_ok"] = True
        except Exception as cleanup_error:
            record["teardown_ok"] = False
            record["teardown_error"] = str(cleanup_error)[:500]
            if error is None:
                error = cleanup_error
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            append(record)
    print(json.dumps(record, indent=2, sort_keys=True))
    if error:
        raise error
    return 0


def deploy(args: argparse.Namespace) -> int:
    record = deploy_record(args.entry)
    image = record.get("image_tag") or record.get("image")
    if not image:
        raise RuntimeError("catalog entry lacks an image tag")
    name = args.name or f"worker-{slug(record['model'])}"
    assert_preflight()
    defaults = model_defaults(record["model"])
    gate = {"name": name, "image": image, "model_baked": bool(record.get("baked")), "gpu_vram_gb": defaults["gpu_vram_gb"], "workersMin": 0, "workersMax": 1, "idleTimeout": 10, "flashBootType": "FLASHBOOT", "spend_limit": 80}
    require_pod_check(gate)
    template_body, endpoint_body = endpoint_config(name, image, record["model"], record.get("parser", "hermes"), record.get("gpu", "NVIDIA RTX A5000"), int(record.get("disk", 30)), bool(record.get("baked")))
    template = rest("POST", "/templates", template_body)
    endpoint_body["templateId"] = template["id"]
    endpoint = rest("POST", "/endpoints", endpoint_body)
    readback = rest("GET", f"/endpoints/{endpoint['id']}")
    if readback.get("workersMin") != 0:
        raise RuntimeError("deployed endpoint violated workersMin=0")
    print(json.dumps({"endpoint_id": endpoint["id"], "template_id": template["id"], "workersMin": readback.get("workersMin")}, indent=2))
    return 0


def build(args: argparse.Namespace) -> int:
    if "/" not in args.model or Path(args.model).expanduser().exists():
        raise RuntimeError("RunPod GitHub source builds accept public Hugging Face org/model IDs only; local weights need a controlled private builder")
    revision = args.revision
    if not revision:
        url = f"https://huggingface.co/api/models/{urllib.parse.quote(args.model, safe='/')}"
        with urllib.request.urlopen(url, timeout=30) as response:
            revision = json.load(response)["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Hugging Face revision must be a 40-character commit SHA")
    if args.engine == "llamacpp" and not args.gguf_file:
        raise RuntimeError("--gguf-file is required for a baked llama.cpp image")
    if args.gguf_file and ("/" in args.gguf_file or not args.gguf_file.endswith(".gguf")):
        raise RuntimeError("--gguf-file must be a repository-relative .gguf filename")
    engine_tag = f"vllm-{VLLM_VERSION.lstrip('v').replace('.', '-')}" if args.engine == "vllm" else "llamacpp-b4721"
    tag = f"{slug(args.model)}-{engine_tag}{'-baked' if args.bake else '-base'}"
    defaults = {"workersMin": 0, "workersMax": 1, "idleTimeout": 10, "flashBoot": True, **model_defaults(args.model)}
    require_pod_check({"name": f"catalog-{slug(args.model)}", "image": f"{args.image_repo}:{tag}", "model_baked": args.bake, "gpu_vram_gb": defaults["gpu_vram_gb"], "workersMin": 0, "workersMax": 1, "idleTimeout": 10, "flashBootType": "FLASHBOOT", "spend_limit": 80})
    request_file = ROOT / "builds" / f"{tag}.json"
    request_file.parent.mkdir(exist_ok=True)
    spec_file = "bake-spec.json" if args.engine == "vllm" else "gguf-bake-spec.json"
    spec = {"bake": True, "model": args.model, "revision": revision}
    if args.engine == "llamacpp":
        spec["gguf_file"] = args.gguf_file
    branch = publish_bake_branch(args.model, revision, args.engine, spec_file, spec) if args.bake and args.publish else git("branch", "--show-current")
    dockerfile = "images/Dockerfile" if args.engine == "vllm" else "images/llamacpp.Dockerfile"
    plan = {"model": args.model, "revision": revision, "engine": args.engine, "vllm_version": VLLM_VERSION if args.engine == "vllm" else None, "gguf_file": args.gguf_file or None, "bake": args.bake, "source_branch": branch, "source_dockerfile": dockerfile, "requested_image_tag": f"{args.image_repo}:{tag}", "endpoint_defaults": defaults, "pod_check": "passed", "status": "source_published_waiting_for_runpod_build" if args.publish else "ready_to_publish"}
    request_file.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_p = sub.add_parser("build")
    build_p.add_argument("model"); build_p.add_argument("--engine", choices=("vllm", "llamacpp"), default="vllm"); build_p.add_argument("--gguf-file"); build_p.add_argument("--bake", action="store_true"); build_p.add_argument("--revision", default=""); build_p.add_argument("--image-repo", default=DEFAULT_IMAGE); build_p.add_argument("--publish", action="store_true", help="push the immutable bake-spec branch to origin"); build_p.set_defaults(func=build)
    test_p = sub.add_parser("test")
    test_p.add_argument("model"); test_p.add_argument("--entry", help="stable catalog entry to update with this verdict"); test_p.add_argument("--image"); test_p.add_argument("--endpoint-id", help="the disposable catalog-test-* endpoint created by a RunPod Git source build"); test_p.add_argument("--bake", action="store_true"); test_p.add_argument("--parser", default="hermes"); test_p.add_argument("--gpu", default="NVIDIA RTX A5000"); test_p.add_argument("--gpu-vram-gb", type=int, default=24); test_p.add_argument("--disk", type=int, default=30); test_p.add_argument("--worker-protocol", choices=("runpod-vllm", "llama-openai"), default="runpod-vllm"); test_p.add_argument("--env-json", help="runtime environment map for non-vLLM custom workers"); test_p.set_defaults(func=test)
    deploy_p = sub.add_parser("deploy")
    deploy_p.add_argument("entry"); deploy_p.add_argument("--name"); deploy_p.set_defaults(func=deploy)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
