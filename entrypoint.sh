#!/usr/bin/env bash
# Assemble the `vllm serve` command from the env vars set in the Containerfile.
# Anything passed as arguments to the container is appended verbatim, so
# `podman run ... --max-model-len 8192` still works as an override.
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
  # The chat template opens every assistant turn with <think>; without this the
  # whole reasoning block lands in message.content instead of reasoning_content.
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
)

# On one 5090 CUDA graph capture allocates outside the --gpu-memory-utilization
# budget and OOMs at startup. Eager mode is what makes the server come up.
[[ "$ENFORCE_EAGER" == "1" ]] && args+=(--enforce-eager)

# Drops the vision tower. Costs image/video input, buys ~50% more KV pool.
[[ "$LANGUAGE_MODEL_ONLY" == "1" ]] && args+=(--language-model-only)

# The MTP draft head ships inside the checkpoint. Roughly doubles single-stream
# throughput, but needs LANGUAGE_MODEL_ONLY=1 here: it allocates its own 2.37
# GiB vocab embedding, which does not fit alongside the vision tower.
[[ "$SPEC_DECODE" == "1" ]] && args+=(
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
)

# shellcheck disable=SC2206  # word splitting is the point
[[ -n "${EXTRA_ARGS:-}" ]] && args+=($EXTRA_ARGS)

echo "+ vllm serve ${args[*]}" >&2
exec vllm serve "${args[@]}" "$@"
