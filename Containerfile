# vLLM on a single RTX 5090 (Blackwell, sm120, 32 GiB).
#
# The image is model-agnostic: which model runs, and with which flags, comes
# from the profile files in profiles/ that run.sh passes in. The ENV block below
# is only the fallback for a profile that leaves something unset.
#
# Checkpoints are NOT baked into the image -- they are mounted from the host
# Hugging Face cache at run time. The image is ~19 GB; the weights are 23-31 GB
# and belong in ~/.cache/huggingface.
FROM docker.io/vllm/vllm-openai:v0.28.0

# Qwen3.x needs transformers >= 5.8.0 for the Qwen3-VL processor (vLLM parses
# config.json itself). v0.28.0 ships 5.15.1, so there is nothing to install.
#
# Note there is no --trust-remote-code anywhere in this setup: none of the
# checkpoints ship .py files or declare auto_map, so the flag would be a no-op.

ENV HF_HOME=/root/.cache/huggingface \
    VLLM_LOGGING_LEVEL=INFO

ENV MODEL=Inferact/Qwen3.8-27B-NVFP4 \
    SERVED_MODEL_NAME=Qwen3.8-27B \
    MAX_MODEL_LEN=32768 \
    MAX_NUM_SEQS=8 \
    GPU_MEMORY_UTILIZATION=0.92 \
    KV_CACHE_DTYPE=fp8 \
    TOOL_CALL_PARSER=qwen3_coder \
    ENFORCE_EAGER=1 \
    LANGUAGE_MODEL_ONLY=0 \
    SPEC_DECODE=0 \
    SPEC_CONFIG={"method":"mtp","num_speculative_tokens":3} \
    HOST=0.0.0.0 \
    PORT=8000 \
    EXTRA_ARGS=

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
