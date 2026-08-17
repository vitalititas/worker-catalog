# Run contract — abliterated delegate-coder pair

**Owner:** codexicus
**Run type:** two serialized RunPod Serverless deployments; one GPU test active at a time.

## 1. Objective

Provide K3's delegate-coder tier with two independent, raw-abliterated 20–30B-class coding
endpoints. Guardrails are applied later by fleet prompts/MDs; neither image adds a refusal layer.

| endpoint | immutable weights | engine | target image | GPU |
|---|---|---|---|---|
| `worker-muse-glimmer-30b-heretic-plus-q4` | `mradermacher/Muse-Glimmer-30B-heretic-plus-GGUF@39c460a5b3d3d075e8586f6803c8013e6d761cb9`, `Muse-Glimmer-30B-heretic-plus.Q4_K_M.gguf` (16,935,292,768 bytes) | llama.cpp | `ghcr.io/vitalititas/worker-catalog:muse-glimmer-30b-heretic-plus-q4` | RTX A6000 48GB |
| `worker-qwen3-coder-30b-a3b-ablit-q4` | `mradermacher/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated-GGUF@d7da518983e85529d7e7de7c9ef3901bcf9d8f51`, `Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated.Q4_K_M.gguf` (18,556,689,600 bytes) | llama.cpp | `ghcr.io/vitalititas/worker-catalog:qwen3-coder-30b-a3b-ablit-q4` | RTX A6000 48GB |

Both sources are 30B-class coding/abliterated GGUFs and remain below the 25GB cold-image ceiling.
Muse's own card documents ATEM XML tool calls; Qwen3 uses its Qwen tool template. They are **not**
vLLM inputs: GGUF requires llama.cpp, while vLLM serves safetensors/quantized checkpoint formats.
The existing `asymintel/runpod-llamacpp-worker:v4` is intentionally excluded: it downloads
`HF_REPO`/`HF_FILE` at cold boot and therefore is neither baked nor volume-backed.

## 2. Success

For each endpoint, prove all of the following before keeping it:

1. `pod-check <config>` exits 0 before create.
2. REST readback is exactly `workersMin=0`, `workersMax=1`, `idleTimeout=10`.
3. A real code-completion request returns successfully.
4. A tool-choice request emits a structured, correctly attributed call and consumes its tool
   result; otherwise the endpoint is recorded as **serve-only**, not an agentic delegate.
5. After idling down, billing reports `currentSpendPerHr == 0`, not merely zero workers.

## 3. Budget ceiling

**$10 total active-GPU spend across both serialized tests.** Record account balance before each
submission and stop immediately if the cumulative decrease reaches $10. No volume is created:
it would be a recurring idle cost and would invalidate the $0-idle claim.

## 4. Teardown / rollback

`workersMin=0` is non-negotiable. On an image build, deployment, load, completion, tool-call,
budget, or billing failure: PATCH `workersMax=0`, read it back, purge/cancel the test job queue,
DELETE the failed endpoint and template, then confirm billing returns to `$0/hr`. Do not start the
second GPU test while the first has unresolved spend or teardown.

## 5. Build and endpoint defaults

Build each image server-side/off the home uplink from `images/llamacpp.Dockerfile`, pinning the
repository revision and exact GGUF filename above. Build copies weights into the image; runtime
HF downloading is prohibited. Deploy each image with `CTX=32768`, `-ngl 99`, FlashBoot enabled,
one A6000 48GB GPU, `workersMin=0`, `workersMax=1`, and `idleTimeout=10`.

The hardware/context choices are fit constraints, not an assertion of optimal quality. Increase
context only after a measured VRAM/load test; never silently substitute a lower quant or a
different revision.
