# Sovereign Worker Catalog

Per-model record of what serves on RunPod, how, and its measured numbers. `catalog.jsonl` is append-only: unknown is recorded as `null`, never inferred.

- `images/Dockerfile` — one parameterized worker image pinned to vLLM `v0.27.1`; it serves a runtime HF model by default or bakes one public HF revision with `--bake`.
- `build-image.sh <org/model> --bake --publish` — resolves the immutable HF revision and pushes only a tiny `bake-spec.json` on a dedicated Git branch. In RunPod select that branch and `images/Dockerfile`; its builder, not this machine, fetches vLLM and the weights. The same Dockerfile builds every model.
- `test-model.sh <model> --endpoint-id <id> [--bake]` — tests the exact disposable `catalog-test-*` endpoint made by the RunPod Git-source build; it validates structured required-argument tool selection and tool-result use, then unconditionally scales down, deletes endpoint/template, and verifies billing returns to `$0/hr`. `--image` is reserved for an already-known registry image.
- `deploy.sh <model> [--name NAME]` — creates a scale-to-zero endpoint only from a catalog record with both `serves=true` and `tool_call_ok=true`.

RunPod's source-build flow is console/GitHub-integrated rather than exposed by its REST API. Its documented UI selects a branch and Dockerfile but not Docker build args, hence the immutable per-branch bake spec. The source repo contains no weights or credentials; private/local weights require a controlled private builder and must never be committed.
