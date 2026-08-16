# Parameterized sovereign RunPod worker. The model is runtime-supplied by
# default. Git-source builds select a baked public revision through the small
# bake-spec.json file committed on their dedicated build branch; RunPod's
# source-build UI exposes branch/Dockerfile selection but no build-arg field.
ARG VLLM_VERSION=v0.27.1
FROM vllm/vllm-openai:${VLLM_VERSION}

ARG VLLM_VERSION
ENV VLLM_VERSION=${VLLM_VERSION} \
    TOKENIZERS_PARALLELISM=false \
    VLLM_HOST=127.0.0.1 \
    VLLM_PORT=8000

RUN python3 -m pip install --no-cache-dir 'runpod~=1.12.0' 'aiohttp>=3.9' 'huggingface-hub>=0.24'
COPY images/worker.py /worker.py
COPY images/bake_model.py /bake_model.py
COPY bake-spec.json /bake-spec.json

# A public immutable HF revision can be baked without putting a token in the
# image. Private/local weights use a controlled private builder, never public
# Git source. bake_model.py is intentionally the only model-specific step.
RUN python3 /bake_model.py /bake-spec.json

ENTRYPOINT ["python3", "/worker.py"]
