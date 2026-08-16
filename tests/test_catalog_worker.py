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


def test_endpoint_config_keeps_kim_context_and_kv_defaults():
    template, _ = catalog_worker.endpoint_config(
        "kim", "image", "SvenBrnn/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-gptq-w4a16",
        "gemma4", "NVIDIA RTX A6000", 60, True,
    )
    assert template["env"]["MAX_MODEL_LEN"] == "262144"
    assert template["env"]["KV_CACHE_DTYPE"] == "fp8"
