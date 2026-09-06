#!/usr/bin/env bash
# Assemble the `vllm serve` command from env vars. Values come from the profile
# file run.sh passes in (profiles/*.env), falling back to the Containerfile
# defaults. Anything passed as arguments to the container is appended verbatim,
# so `./run.sh --max-model-len 8192` still works as an override.
set -euo pipefail

args=(
  "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  # Both Qwen3.x chat templates open every assistant turn with <think>; without
  # this the whole reasoning block lands in message.content rather than in a
  # separate `reasoning` field.
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser "$TOOL_CALL_PARSER"
)

# Only the dense 27B needs this: CUDA graph capture allocates outside the
# --gpu-memory-utilization budget and OOMs at startup when weights already fill
# the card. The 35B-A3B MoE has room and runs with graphs on.
[[ "$ENFORCE_EAGER" == "1" ]] && args+=(--enforce-eager)

# Drops the vision tower. Costs image/video input, buys KV cache.
[[ "$LANGUAGE_MODEL_ONLY" == "1" ]] && args+=(--language-model-only)

# The MTP draft head ships inside both checkpoints; SPEC_CONFIG is per-model
# because the working draft depth and MoE backend differ.
[[ "$SPEC_DECODE" == "1" ]] && args+=(--speculative-config "$SPEC_CONFIG")

# shellcheck disable=SC2206  # word splitting is the point
[[ -n "${EXTRA_ARGS:-}" ]] && args+=($EXTRA_ARGS)

echo "+ vllm serve ${args[*]}" >&2
exec vllm serve "${args[@]}" "$@"
