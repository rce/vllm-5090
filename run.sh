#!/usr/bin/env bash
# Start the Qwen3.8-27B vLLM server on the local RTX 5090.
#
#   ./run.sh                                  # NVFP4, 32K ctx, defaults
#   MODEL=Qwen/Qwen3.8-27B-FP8 ./run.sh       # official FP8 checkpoint
#   MAX_MODEL_LEN=65536 ./run.sh              # any Containerfile env var
#   ./run.sh --max-num-batched-tokens 4096    # raw `vllm serve` flags
set -euo pipefail

IMAGE=${IMAGE:-localhost/qwen38-vllm:latest}
NAME=${NAME:-qwen38}
HF_CACHE=${HF_HOME:-$HOME/.cache/huggingface}

# Forward only the tuning vars the caller actually set, so the image defaults
# stay authoritative for everything else.
env_args=()
for v in MODEL SERVED_MODEL_NAME MAX_MODEL_LEN MAX_NUM_SEQS GPU_MEMORY_UTILIZATION \
         KV_CACHE_DTYPE ENFORCE_EAGER LANGUAGE_MODEL_ONLY SPEC_DECODE EXTRA_ARGS HF_TOKEN; do
  [[ -n "${!v:-}" ]] && env_args+=(-e "$v=${!v}")
done

exec podman run --rm -it \
  --name "$NAME" \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --ipc=host \
  -p "${PORT:-8000}:8000" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  "${env_args[@]}" \
  "$IMAGE" "$@"
