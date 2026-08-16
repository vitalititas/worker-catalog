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
