# Upgrade the llama.cpp runtime in an already-baked GGUF worker without copying
# or re-downloading its immutable weights. Both image arguments must be pinned tags.
ARG BASE_IMAGE
ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda-b10450
FROM ${LLAMA_IMAGE} AS llama
FROM ${BASE_IMAGE}

COPY --from=llama /app/ /app/
COPY images/llamacpp_worker.py /llamacpp_worker.py

ENTRYPOINT []
CMD ["python3", "-u", "/llamacpp_worker.py"]
