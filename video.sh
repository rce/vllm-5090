#!/usr/bin/env bash
# Run video.py inside the video-5090 container on the local RTX 5090.
#
#   ./video.sh --model wan2.2-5b
#   ./video.sh --model wan2.2-5b --frames 121,241 --label "wan2.2-5b 720p" --out docs/video.json
#
# The repo is mounted at /work so clips land in video/out/ and --out can
# point at docs/video.json directly. Weights come from the host HF cache.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tty=(); [[ -t 0 ]] && tty=(-t)
exec podman run --rm -i "${tty[@]}" \
  --name "${NAME:-video-5090}" \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --ipc=host \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -v "$HERE:/work" \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  "${IMAGE:-localhost/video-5090:latest}" \
  "$@"
