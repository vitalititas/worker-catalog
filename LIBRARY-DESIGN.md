# Sovereign Model-Image Library — design (claude, 2026-08-16)
Goal: a library of RunPod-servable custom worker images + a catalog directory, so ANY tool-verified
model deploys instantly on a config we own. Fixes the SWE-8B-class failure (stock vLLM can't load some
models) by owning the vLLM version per image.

## Cost model (LEARN FROM THE INCIDENT)
- Building + pushing an image = CHEAP (CPU + registry, NO GPU). Build the whole library for ~nothing.
- Only DEPLOYING an endpoint burns GPU. Test-deploy each model ONCE, TEARDOWN IMMEDIATELY (~5c each).
- **STRICT TEARDOWN DISCIPLINE:** every test-deploy wraps in try/finally that PATCHes workersMax=0 then
  DELETEs endpoint+template on ANY exit path (success, fail, exception, timeout). My probe orphaned
  billing endpoints by skipping teardown-on-error — do not repeat. Verify $0/hr after each teardown.
- NEVER run test-deploys in parallel (compounds orphan risk + billing). One at a time, teardown between.

## Structure (~/worker-catalog/)
- `catalog.jsonl` — append-only: {model, image_tag, parser, gpu, serves, cold_s, warm_s, tool_call_ok, notes}
- `images/` — Dockerfile(s). ONE parameterized base image (pinned current vLLM) + per-model bake variants.
- `build-image.sh <model> [--bake]` — build (RunPod-side/registry) + push, record image_tag. No GPU.
- `test-model.sh <model|image>` — deploy ONCE, health + a REAL tool-call test, record verdict, TEARDOWN.
- `deploy.sh <catalog-entry>` — deploy a proven catalog model as a live endpoint (workersMin 0).

## The vLLM-version fix
Stock runpod/worker-v1-vllm:v2.25.1 fails qwen3 distills. Custom image pins a vLLM version that loads
them (test which). That is the whole reason to own the image. Tool-parser also baked per model family.

## First catalog population (tool-verified coders from prior testing)
Qwen/Qwen2.5-Coder-7B-Instruct (PROVEN serves+gated), Qwen/Qwen2.5-Coder-14B, Qwen/Qwen2.5-Coder-32B,
SWE-UT-8B-Qwen3-Coder-Distill (needs newer vLLM), jica98/qwen3.5-4B-super-coder, + Kim's gemma + others.
Record each: serves on stock? needs custom image? tool-calling verified?
