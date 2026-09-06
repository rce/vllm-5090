#!/usr/bin/env bash
# Start one of the profiles in profiles/ on the local RTX 5090.
#
#   ./run.sh                                   # default profile (Qwen3.8-27B)
#   ./run.sh -p qwen3.6-35b-a3b                # the MoE comparison model
#   ./run.sh -p qwen3.6-35b-a3b SPEC_DECODE=1  # override a profile value
#   ./run.sh --max-num-batched-tokens 4096     # raw `vllm serve` flags
#
# Env vars also override the profile: MAX_MODEL_LEN=8192 ./run.sh
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROFILE=${PROFILE:-qwen3.8-27b}

if [[ ${1:-} == -p || ${1:-} == --profile ]]; then
  PROFILE=$2
  shift 2
fi

ENV_FILE="$HERE/profiles/$PROFILE.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "run.sh: no such profile '$PROFILE'. Available:" >&2
  for f in "$HERE"/profiles/*.env; do echo "  $(basename "$f" .env)" >&2; done
  exit 1
fi

# KEY=VALUE arguments become per-run overrides; the rest go to `vllm serve`.
overrides=()
serve_args=()
for a in "$@"; do
  if [[ "$a" == [A-Z_]*=* ]]; then overrides+=(-e "$a"); else serve_args+=("$a"); fi
done

# Env vars set in the calling shell also override the profile. podman applies
# --env-file first and -e after, so these win.
for v in MODEL SERVED_MODEL_NAME MAX_MODEL_LEN MAX_NUM_SEQS GPU_MEMORY_UTILIZATION \
         KV_CACHE_DTYPE TOOL_CALL_PARSER ENFORCE_EAGER LANGUAGE_MODEL_ONLY \
         SPEC_DECODE SPEC_CONFIG EXTRA_ARGS HF_TOKEN; do
  [[ -n "${!v:-}" ]] && overrides+=(-e "$v=${!v}")
done

exec podman run --rm -it \
  --name "${NAME:-vllm-$PROFILE}" \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --ipc=host \
  -p "${HOST_PORT:-8000}:8000" \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  --env-file "$ENV_FILE" \
  "${overrides[@]}" \
  "${IMAGE:-localhost/vllm-5090:latest}" \
  "${serve_args[@]}"
