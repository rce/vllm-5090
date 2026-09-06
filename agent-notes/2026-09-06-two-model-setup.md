# 2026-09-06 — Qwen3.8-27B and Qwen3.6-35B-A3B on one RTX 5090

## Goal

Serve Qwen3.8-27B in a container on this box (single RTX 5090, 32 GiB), then add
Qwen3.6-35B-A3B alongside it for comparison.

## What the card actually allows

31.4 GiB usable, ~1 GiB of that already held by the desktop compositor. That
number decides everything below.

- `Qwen/Qwen3.8-27B` BF16 is 55.6 GB. Not attempted — CPU offload would need
  ~24 GB on a host with 30 GiB total.
- `Qwen/Qwen3.8-27B-FP8` is ~28.6 GiB. It *starts* and then OOMs on the first
  request. This is the trap worth remembering: vLLM's memory profiling
  underestimates the real activation peak when weights already hold 91% of the
  card, so you get a clean `Application startup complete` and a 500 later.
  Backing off to 4K context / 2 seqs / 1024-token batches gives a working but
  uninteresting server.
- NVFP4 is what fits with room to work in.

## Design

The image is model-agnostic. Model choice and flags live in `profiles/*.env`;
`run.sh` feeds one to podman via `--env-file` and `entrypoint.sh` turns env vars
into a `vllm serve` command line. Adding a model is a new env file, not a
rebuild. Verified that podman passes JSON-valued vars through `--env-file`
intact and that `-e` overrides beat `--env-file`.

Weights are mounted from the host HF cache rather than baked in — the image
stays ~19 GB.

## Findings worth keeping

**The dense 27B needs `--enforce-eager`.** CUDA graph capture allocates outside
the `--gpu-memory-utilization` budget, so with 24.6 GiB of weights it dies during
capture regardless of that fraction. The MoE has room and runs with graphs on —
which is a good part of why it is so much faster.

**MTP on the 27B needs `--language-model-only`.** It OOMs asking for exactly
2.37 GiB = 248,320 x 5,120 in BF16. `qwen3_5_mtp.py` gives the draft head its own
`VocabParallelEmbedding` despite `mtp_use_dedicated_embeddings: false` in the
checkpoint. Looks like a vLLM bug worth reporting upstream; not investigated
further.

**The 35B-A3B flag set is not mine.** It is the vLLM recipe's verified
`rtx_5090` profile, lifted verbatim from
`variants.nvfp4.hardware_overrides.rtx_5090` in the recipe JSON. Reading the
JSON API beat reconstructing flags from the prose page. `--block-size 128` and
`VLLM_HAS_FLASHINFER_CUBIN=1` come from FlashInfer's TRT-LLM attention kernels;
the 64K context cap comes from the 32 GB card.

**No `--trust-remote-code`.** It appears in the recipe's suggested command, but
neither checkpoint ships `.py` files or declares `auto_map`, so it is a no-op
here. Left out deliberately.

## Numbers

In `docs/results.json`. Headline: the MoE is ~10x the dense 27B's decode speed
(237 vs 25 tok/s) and holds 6x the KV cache, because only 3B params are active
and graphs fit.

## Not done

- `unsloth/Qwen3.8-27B-NVFP4` (mixed precision, ~21.3 GB) — the recipe suggests
  it leaves roughly twice the KV pool of the Inferact build. Untested here.
- Concurrency / batched throughput. Everything measured is single-stream.
- Long-context behaviour. Nothing measured above a short prompt; the KV pool
  figures are vLLM's startup report, not observed at depth.
- YaRN context extension past native 262K.
