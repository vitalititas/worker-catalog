#!/usr/bin/env python3
"""Tokenizer-counted context proof for the retained Kim endpoint.

Dry-run is the default.  ``--execute`` submits exactly one job only when the
existing queue is empty and the account has no active GPU spend.  This avoids
turning an allocation wait into duplicate A6000 work.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer

import catalog_worker as runpod


ENDPOINT = "khepsu6kjhkffd"
MODEL = "SvenBrnn/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-gptq-w4a16"
TARGET_CONTENT_TOKENS = 249_000  # leaves room for chat framing and 128 output tokens under 262144
NEEDLES = (
    "N0=violet-2f6d9a",
    "N50=ember-91c4e7",
    "N100=cedar-7ba281",
    "N150=onyx-c6945e",
    "N200=saffron-18de73",
    "N240=glacier-a5f209",
)
POSITIONS = (0, 50_000, 100_000, 150_000, 200_000, 240_000)


def stable_filler(tokenizer) -> list[int]:
    """Find a token whose decoded repetitions re-tokenize identically."""
    for candidate in ("§", "¤", "※", "\u200b", " x"):
        unit = tokenizer.encode(candidate, add_special_tokens=False)
        if not unit:
            continue
        repeated = tokenizer.decode(unit * 64, clean_up_tokenization_spaces=False)
        if tokenizer.encode(repeated, add_special_tokens=False) == unit * 64:
            return unit
    raise RuntimeError("could not construct a stable tokenizer-counted filler")


def context_messages(tokenizer) -> tuple[list[dict[str, str]], int]:
    filler = stable_filler(tokenizer)
    ids: list[int] = []
    for position, needle in zip(POSITIONS, NEEDLES, strict=True):
        while len(ids) + len(filler) <= position:
            ids.extend(filler)
        ids.extend(tokenizer.encode(f"\n[{needle}]\n", add_special_tokens=False))
    tail = (
        "\nReturn one JSON object with key `markers` whose value is the six "
        "marker strings exactly as supplied, in order. Do not add commentary.\n"
    )
    tail_ids = tokenizer.encode(tail, add_special_tokens=False)
    while len(ids) + len(filler) + len(tail_ids) <= TARGET_CONTENT_TOKENS:
        ids.extend(filler)
    ids.extend(tail_ids)
    content = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    if tokenizer.encode(content, add_special_tokens=False) != ids:
        raise RuntimeError("decoded context does not round-trip through the model tokenizer")
    messages = [{"role": "user", "content": content}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    # Transformers 5 returns a BatchEncoding here; len(rendered) is its field
    # count (``input_ids``, ``attention_mask``), not the prompt length.
    prompt_tokens = len(rendered["input_ids"]) if hasattr(rendered, "__getitem__") and "input_ids" in rendered else len(rendered)
    if not 200_000 < prompt_tokens < 262_016:
        raise RuntimeError(f"unsafe prompt budget: {prompt_tokens} tokens")
    return messages, prompt_tokens


def verify_configuration() -> dict[str, object]:
    endpoint = runpod.rest("GET", f"/endpoints/{ENDPOINT}")
    template = runpod.rest("GET", f"/templates/{endpoint['templateId']}")
    env = template.get("env") or {}
    expected = {"MODEL_NAME": MODEL, "MAX_MODEL_LEN": "262144", "KV_CACHE_DTYPE": "fp8"}
    actual = {key: env.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"Kim template drift: {actual!r}")
    if (endpoint.get("workersMin"), endpoint.get("workersMax"), endpoint.get("idleTimeout")) != (0, 1, 10):
        raise RuntimeError("Kim no longer has the approved scale-to-zero settings")
    return {"endpoint": endpoint["id"], "template": template["id"], "image": template.get("imageName"), "env": actual}


def response_text(job: dict[str, object]) -> str:
    output = job.get("output")
    if isinstance(output, list) and output:
        output = output[-1]
    if isinstance(output, dict):
        choices = output.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                return str(message.get("content") or "")
        return json.dumps(output, sort_keys=True)
    return str(output or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="submit one proof job after all live safety checks")
    args = parser.parse_args()

    config = verify_configuration()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    messages, local_tokens = context_messages(tokenizer)
    plan = {"config": config, "local_prompt_tokens": local_tokens, "needles": list(NEEDLES), "execute": args.execute}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0

    health = runpod.request(f"{runpod.RUN}/{ENDPOINT}/health")
    if (health.get("jobs") or {}).get("inQueue", 0):
        raise RuntimeError("Kim already has queued work; refusing a duplicate context job")
    if runpod.account().get("currentSpendPerHr") != 0:
        raise RuntimeError("account has active GPU spend; serialize the Kim proof")

    started = time.monotonic()
    job = runpod.job(ENDPOINT, {"input": {"messages": messages, "sampling_params": {"temperature": 0, "max_tokens": 128}, "stream": False}}, timeout=1_800)
    elapsed = round(time.monotonic() - started, 1)
    if job.get("status") != "COMPLETED":
        raise RuntimeError(f"context job {job.get('status')}: {job.get('error')}")
    text = response_text(job)
    missing = [needle for needle in NEEDLES if needle not in text]
    usage = job.get("output", {}).get("usage", {}) if isinstance(job.get("output"), dict) else {}
    observed = usage.get("prompt_tokens")
    if not isinstance(observed, int) or observed < 200_000:
        raise RuntimeError(f"server did not report a >200k prompt-token count: {observed!r}")
    if missing:
        raise RuntimeError(f"context probe missed markers: {missing!r}")
    result = {**plan, "elapsed_s": elapsed, "server_prompt_tokens": observed, "reply": text, "billing_after": runpod.account().get("currentSpendPerHr")}
    report = Path(__file__).with_name("reports") / f"kim-context-{int(time.time())}.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
