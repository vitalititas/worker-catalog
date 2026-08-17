import importlib.util
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "catalog_worker.py"
SPEC = importlib.util.spec_from_file_location("catalog_worker", MODULE)
catalog_worker = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(catalog_worker)


def test_output_message_accepts_list_wrapped_runpod_output():
    result = {"output": [{"choices": [{"message": {"content": "ok"}}]}]}
    assert catalog_worker.output_message(result) == {"content": "ok"}


def test_endpoint_config_keeps_baked_models_offline():
    _, endpoint = catalog_worker.endpoint_config("test", "image", "model", "hermes", "NVIDIA RTX A5000", 30, True)
    assert endpoint["workersMin"] == 0
    template, _ = catalog_worker.endpoint_config("test", "image", "model", "hermes", "NVIDIA RTX A5000", 30, True)
    assert "MODEL_NAME" not in template["env"]


def test_endpoint_config_accepts_a_non_vllm_runtime_environment():
    template, endpoint = catalog_worker.endpoint_config(
        "test", "asymintel/runpod-llamacpp-worker:v4", "muse", "hermes",
        "NVIDIA RTX A6000", 60, False, {"HF_REPO": "example/model", "HF_FILE": "model.Q4_K_M.gguf"},
    )
    assert endpoint["workersMin"] == 0
    assert template["env"]["HF_REPO"] == "example/model"
    assert "MODEL_NAME" not in template["env"]


def test_job_allows_a_cold_worker_to_return_its_job_id(monkeypatch):
    calls = []

    def fake_request(url, method="GET", body=None, timeout=90):
        calls.append((url, method, timeout))
        if url.endswith("/run"):
            return {"id": "cold-job"}
        return {"status": "COMPLETED", "output": {}}

    monkeypatch.setattr(catalog_worker, "request", fake_request)
    assert catalog_worker.job("endpoint", {"input": {}}, timeout=900)["status"] == "COMPLETED"
    assert calls[0] == (f"{catalog_worker.RUN}/endpoint/run", "POST", 300)


def test_cleanup_purges_queue_before_scaling_down(monkeypatch):
    calls = []

    def fake_request(url, method="GET", body=None, timeout=90):
        calls.append((url, method, body))
        return {"removed": 1}

    # First GET is readback; subsequent endpoint/template GETs verify deletion.
    endpoint_gets = iter(({"workersMax": 0}, catalog_worker.urllib.error.HTTPError("url", 404, "missing", {}, None)))
    def cleanup_rest(method, path, body=None, timeout=90):
        if method == "GET" and path == "/endpoints/endpoint":
            value = next(endpoint_gets)
            if isinstance(value, Exception):
                raise value
            return value
        if method == "GET" and path == "/templates/template":
            raise catalog_worker.urllib.error.HTTPError("url", 404, "missing", {}, None)
        return {}

    monkeypatch.setattr(catalog_worker, "request", fake_request)
    monkeypatch.setattr(catalog_worker, "rest", cleanup_rest)
    monkeypatch.setattr(catalog_worker, "account", lambda: {"currentSpendPerHr": 0})
    catalog_worker.cleanup_ids(["endpoint"], ["template"])
    assert calls == [(f"{catalog_worker.RUN}/endpoint/purge-queue", "POST", {})]


def test_endpoint_config_keeps_kim_context_and_kv_defaults():
    template, _ = catalog_worker.endpoint_config(
        "kim", "image", "SvenBrnn/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-gptq-w4a16",
        "gemma4", "NVIDIA RTX A6000", 60, True,
    )
    assert template["env"]["MAX_MODEL_LEN"] == "262144"
    assert template["env"]["KV_CACHE_DTYPE"] == "fp8"


def test_llamacpp_worker_uses_pinned_tool_parsing_engine():
    root = Path(__file__).resolve().parents[1]
    assert "server-cuda-b10450" in (root / "images" / "llamacpp.Dockerfile").read_text()
    source = (root / "images" / "llamacpp_worker.py").read_text()
    assert '"--jinja"' in source
    assert '"--flash-attn", "on", "--jinja"' in source
    assert "--break-system-packages" in (root / "images" / "llamacpp.Dockerfile").read_text()
