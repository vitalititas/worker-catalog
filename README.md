# Sovereign Worker Catalog

Per-model record of what serves on RunPod, how, and its measured numbers. `catalog.jsonl` is append-only: unknown is recorded as `null`, never inferred.

- `images/Dockerfile` — one parameterized worker image pinned to vLLM `v0.27.1`; it serves a runtime HF model by default or bakes one public HF revision with `--bake`.
- `build-image.sh <org/model> [--bake]` — resolves the immutable HF revision and writes a RunPod GitHub-source-build request. It never uses the home network for image layers.
- `test-model.sh <model> --image <registry/image:tag> [--bake]` — one serialized, disposable endpoint; validates structured required-argument tool selection and tool-result use, then unconditionally scales down, deletes resources, and verifies billing returns to `$0/hr`.
- `deploy.sh <model> [--name NAME]` — creates a scale-to-zero endpoint only from a catalog record with both `serves=true` and `tool_call_ok=true`.

RunPod's source-build flow is console/GitHub-integrated rather than exposed by its REST API. The source repo therefore contains only code and immutable build requests; private/local weights require a controlled private builder and must never be committed.
