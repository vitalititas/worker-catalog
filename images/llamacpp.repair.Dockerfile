# Rebase an already-baked GGUF worker onto a newer llama.cpp runtime.
# Copying /app alone is invalid: the b10450 binaries require the libc/libstdc++
# from their own base image. Both image arguments must be pinned tags.
ARG BASE_IMAGE
ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda-b10450
FROM ${BASE_IMAGE} AS baked
FROM ${LLAMA_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip curl \
 && pip3 install --no-cache-dir 'runpod==1.10.0' requests \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=baked /models/ /models/
COPY --from=baked /baked-gguf-path /baked-gguf-path
COPY images/llamacpp_worker.py /llamacpp_worker.py

ENTRYPOINT []
CMD ["python3", "-u", "/llamacpp_worker.py"]
