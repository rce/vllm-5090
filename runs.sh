#!/usr/bin/env bash
# Re-measure the `runs` table on the results page: single-stream decode and
# the KV pool for every configuration, each server started and stopped in
# turn. About 25 minutes on the card; the card must be free when it starts.
#
#   ./runs.sh                      # every configuration below
#   ./runs.sh qwen36-moe-default   # just the named ids
#
# Each line of the table is `id | profile | run.sh overrides | shape`. Ids are
# what the page keys on (the hero tiles look for qwen36-moe-default and
# qwen38-dense-default), so keep them stable. Adding a configuration means
# adding a line here.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
OUT=${OUT:-docs/results.json}
NAME=vllm-runs

CONFIGS=(
  "qwen36-moe-default   | qwen3.6-35b-a3b        |                                                          | MoE, 35B total / 3B active"
  "qwen36-moe-mtp       | qwen3.6-35b-a3b        | SPEC_DECODE=1                                            | MoE, 35B total / 3B active"
  "nemotron-default     | nemotron-3.5-lightning |                                                          | MoE, 30B total / 3B active, Mamba hybrid"
  "qwen38-dense-default | qwen3.8-27b            |                                                          | dense 27B"
  "qwen38-dense-mtp     | qwen3.8-27b            | SPEC_DECODE=1 LANGUAGE_MODEL_ONLY=1                      | dense 27B"
  "qwen38-fp8-official  | qwen3.8-27b            | MODEL=Qwen/Qwen3.8-27B-FP8 MAX_MODEL_LEN=4096 MAX_NUM_SEQS=2 GPU_MEMORY_UTILIZATION=0.93 LANGUAGE_MODEL_ONLY=1 EXTRA_ARGS=--max-num-batched-tokens_1024 | dense 27B"
)

wait_ready() {
  for _ in $(seq 1 240); do
    curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && return 0
    podman ps --format '{{.Names}}' | grep -q "^$NAME\$" || { echo "server exited:"; podman logs "$NAME" 2>&1 | tail -20; return 1; }
    sleep 5
  done
  echo "server did not come up in 20 minutes"; return 1
}

trim() { local s=$1; s=${s#"${s%%[![:space:]]*}"}; s=${s%"${s##*[![:space:]]}"}; printf '%s' "$s"; }

for line in "${CONFIGS[@]}"; do
  IFS='|' read -r id profile overrides shape <<<"$line"
  id=$(trim "$id"); profile=$(trim "$profile"); overrides=$(trim "$overrides"); shape=$(trim "$shape")
  if (($#)) && ! printf '%s\n' "$@" | grep -qx "$id"; then continue; fi
  # An EXTRA_ARGS value may need spaces; they are written as _ in the table.
  ov=(); sets=()
  for kv in $overrides; do
    if [[ $kv == EXTRA_ARGS=* ]]; then v=${kv#EXTRA_ARGS=}; kv="EXTRA_ARGS=${v//_/ }"; fi
    ov+=("$kv"); sets+=(--set "$kv")
  done
  echo; echo "=== $(date +%H:%M:%S) $id: run.sh -p $profile ${ov[*]:-}"
  podman rm -f "$NAME" >/dev/null 2>&1 || true
  DETACH=1 NAME=$NAME ./run.sh -p "$profile" "${ov[@]}" >/dev/null
  if wait_ready; then
    # What the weights take on the card, from vLLM's own load report; the
    # checkpoint's size on disk is not the same thing for mixed-precision files.
    gib=$(podman logs "$NAME" 2>&1 | grep -o 'Model loading took [0-9.]* GiB' | head -1 | grep -o '[0-9.]*' || true)
    weights=(); [[ -n "$gib" ]] && weights=(--weights-gb "$(python3 -c "print(round($gib * 1.073741824, 1))")")
    ./bench.py --single --id "$id" --profile "$profile" "${sets[@]}" --shape "$shape" "${weights[@]}" --out "$OUT" \
      || echo "$id: measurement failed"
  else
    echo "$id: server failed to start"
  fi
  podman stop -t 30 "$NAME" >/dev/null 2>&1 || true
  podman rm -f "$NAME" >/dev/null 2>&1 || true
done
echo; echo "=== $(date +%H:%M:%S) runs done"
