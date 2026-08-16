# Baked GGUF fallback for architectures that lack a vLLM-compatible quant.
# Build selection is the tiny immutable gguf-bake-spec.json on a source branch.
FROM ghcr.io/ggml-org/llama.cpp:server-cuda-b4721

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip curl \
 && pip3 install --no-cache-dir runpod huggingface_hub requests \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY images/llamacpp_worker.py /llamacpp_worker.py
COPY images/bake_gguf.py /bake_gguf.py
COPY gguf-bake-spec.json /gguf-bake-spec.json
RUN python3 /bake_gguf.py /gguf-bake-spec.json

ENTRYPOINT []
CMD ["python3", "-u", "/llamacpp_worker.py"]
