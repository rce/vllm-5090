# vLLM serving Qwen3.8-27B on a single RTX 5090 (Blackwell, sm120, 32 GiB).
#
# The 27B checkpoints are NOT baked into the image -- they are mounted from the
# host Hugging Face cache at run time (see run.sh). The image is ~19 GB; the
# weights are 25-31 GB and belong in ~/.cache/huggingface.
FROM docker.io/vllm/vllm-openai:v0.28.0

# Qwen3.8 needs transformers >= 5.8.0 for the Qwen3-VL processor (vLLM parses
# config.json itself). v0.28.0 ships 5.15.1, so there is nothing to install.

ENV HF_HOME=/root/.cache/huggingface \
    VLLM_LOGGING_LEVEL=INFO

# --- Defaults tuned for one 31.4 GiB-usable RTX 5090 -------------------------
# NVFP4 is the precision that actually fits one card with room for a usable KV
# pool. Override MODEL to try another checkpoint (see README).
ENV MODEL=Inferact/Qwen3.8-27B-NVFP4 \
    SERVED_MODEL_NAME=Qwen3.8-27B \
    MAX_MODEL_LEN=32768 \
    MAX_NUM_SEQS=8 \
    GPU_MEMORY_UTILIZATION=0.92 \
    KV_CACHE_DTYPE=fp8 \
    ENFORCE_EAGER=1 \
    LANGUAGE_MODEL_ONLY=0 \
    SPEC_DECODE=0 \
    HOST=0.0.0.0 \
    PORT=8000 \
    EXTRA_ARGS=

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
