# Parameterized sovereign RunPod worker.  The model is supplied at runtime by
# default; --bake source builds set BAKE_MODEL/BAKE_REVISION instead.
ARG VLLM_VERSION=v0.27.1
FROM vllm/vllm-openai:${VLLM_VERSION}

ARG VLLM_VERSION
ARG BAKE_MODEL=""
ARG BAKE_REVISION=""
ENV VLLM_VERSION=${VLLM_VERSION} \
    TOKENIZERS_PARALLELISM=false \
    VLLM_HOST=127.0.0.1 \
    VLLM_PORT=8000

RUN python3 -m pip install --no-cache-dir 'runpod~=1.12.0' 'aiohttp>=3.9' 'huggingface-hub>=0.24'
COPY worker.py /worker.py

# A source build can bake a public Hugging Face revision without putting a
# token in the image. Private/local weights use a controlled private builder,
# never this public Git source path.
RUN if [ -n "$BAKE_MODEL" ]; then \
      export BAKE_MODEL BAKE_REVISION; \
      python3 -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ["BAKE_MODEL"], revision=os.environ.get("BAKE_REVISION") or None, local_dir="/models/model")' && \
      printf '%s' /models/model > /baked-model-path; \
    fi

ENTRYPOINT ["python3", "/worker.py"]
