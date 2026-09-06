# Qwen3.8-27B on one RTX 5090

vLLM 0.28.0 serving Qwen3.8-27B behind an OpenAI-compatible API on `:8000`,
tuned for this box: a single RTX 5090 — 32,607 MiB total, **31.4 GiB usable**,
of which ~1 GiB is already spoken for by the desktop compositor.

Every number below was measured on this machine, not copied from a recipe.

## Build and run

```sh
podman build -t qwen38-vllm .
hf download Inferact/Qwen3.8-27B-NVFP4   # once, ~25 GB into ~/.cache/huggingface
./run.sh
```

Weights are **not** baked into the image — the host Hugging Face cache is
mounted at `/root/.cache/huggingface`, so the image stays ~19 GB and switching
checkpoints costs a download, not a rebuild.

```sh
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B",
       "messages":[{"role":"user","content":"Give me three primes above 100."}],
       "temperature":1.0,"top_p":0.95,"max_tokens":2048}'
```

## Which checkpoint fits

Qwen3.8-27B is a dense 27B hybrid-attention VLM: 48 of 64 layers are Gated
DeltaNet linear attention, 16 are full attention, plus a vision tower and an
in-checkpoint MTP draft head. Dense means the whole thing lives in VRAM.

| Checkpoint | Weights | Result on this card |
| --- | --- | --- |
| `Qwen/Qwen3.8-27B` (official BF16) | 55.6 GB | Never starts — ~24 GB would have to spill to a 30 GiB host |
| `Qwen/Qwen3.8-27B-FP8` (official FP8) | ~28.6 GiB | Runs, but only text-only at **4K context** |
| `Inferact/Qwen3.8-27B-NVFP4` | ~24.6 GiB | **Default.** 32K context with vision |
| `unsloth/Qwen3.8-27B-NVFP4` | ~21.3 GiB | Untried here; mixed precision, should leave more KV room |

Measured configurations:

| Config | Context | KV pool | Vision | Decode |
| --- | --- | --- | --- | --- |
| NVFP4, defaults | 32,768 | 60,681 tok | yes | ~25 tok/s |
| NVFP4, `SPEC_DECODE=1 LANGUAGE_MODEL_ONLY=1` | 32,768 | 70,087 tok | no | **~50 tok/s** |
| Official FP8, `MAX_MODEL_LEN=4096 MAX_NUM_SEQS=2 GPU_MEMORY_UTILIZATION=0.93 LANGUAGE_MODEL_ONLY=1 EXTRA_ARGS='--max-num-batched-tokens 1024'` | 4,096 | 9,557 tok | no | ~38 tok/s |

### About the official FP8 build

It is worth being precise about how it fails, because it does not fail at
startup. At 8K context it profiles fine, reports a 21,845-token KV pool and
prints `Application startup complete` — then dies on the *first real request*
with `Tried to allocate 394.00 MiB. GPU 0 ... 151.75 MiB is free`. Memory
profiling underestimates the true activation peak when the weights already
occupy 91% of the card.

Backing off to 4K context, 2 sequences and a 1024-token batch does produce a
working server. It is just not a useful one next to NVFP4's 32K window and
vision support, so NVFP4 is the default. To try it anyway:

```sh
MODEL=Qwen/Qwen3.8-27B-FP8 MAX_MODEL_LEN=4096 MAX_NUM_SEQS=2 \
  GPU_MEMORY_UTILIZATION=0.93 LANGUAGE_MODEL_ONLY=1 \
  EXTRA_ARGS='--max-num-batched-tokens 1024' ./run.sh
```

NVFP4 is a real kernel path on sm120, not an emulation fallback — vLLM logs
`Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM`.

## Why these defaults

`--enforce-eager` decides whether the server starts at all. CUDA graph capture
allocates *outside* the `--gpu-memory-utilization` budget, so on one card it
dies during capture no matter how that fraction is set. Everything else is a
lever on KV pool size, not a fix for that.

`--reasoning-parser qwen3` is likewise not optional in practice: the chat
template opens every assistant turn with `<think>`, so without it the whole
reasoning block lands in `message.content`. With it, reasoning arrives in a
separate `reasoning` field on the message and `content` holds just the answer.

vLLM auto-disables DeepGemm for `model_type=qwen3_5_text` on Blackwell and falls
back to CUTLASS on its own; no `VLLM_USE_DEEP_GEMM=0` needed.

## Knobs

Env vars read by `entrypoint.sh`; set them in front of `./run.sh`. Trailing
arguments to `./run.sh` are appended to `vllm serve` too.

| Var | Default | Notes |
| --- | --- | --- |
| `MODEL` | `Inferact/Qwen3.8-27B-NVFP4` | Any HF id or mounted path |
| `MAX_MODEL_LEN` | `32768` | Native context is 262,144, well beyond one card |
| `MAX_NUM_SEQS` | `8` | Lower = more KV pool per sequence |
| `GPU_MEMORY_UTILIZATION` | `0.92` | |
| `KV_CACHE_DTYPE` | `fp8` | `auto` for BF16 KV, at ~20% less KV pool |
| `ENFORCE_EAGER` | `1` | Turning this off OOMs at startup on one card |
| `LANGUAGE_MODEL_ONLY` | `0` | `1` drops the vision tower: no image/video input, ~9K more KV tokens |
| `SPEC_DECODE` | `0` | MTP draft head — see below |
| `EXTRA_ARGS` | — | Raw `vllm serve` flags |

### MTP speculative decoding

`SPEC_DECODE=1` doubles single-stream throughput (25 → 50 tok/s, ~0.48 draft
token acceptance), but **requires `LANGUAGE_MODEL_ONLY=1` on this card**:

```sh
SPEC_DECODE=1 LANGUAGE_MODEL_ONLY=1 ./run.sh
```

With vision loaded it OOMs asking for exactly 2.37 GiB. That is
`vocab_size × hidden_size` in BF16 — 248,320 × 5,120 — because
`vllm/model_executor/models/qwen3_5_mtp.py` gives the draft head its own
`VocabParallelEmbedding` even though the checkpoint sets
`mtp_use_dedicated_embeddings: false`. Lowering `GPU_MEMORY_UTILIZATION` does
not help; that allocation happens after profiling, outside the budget.

### Thinking control

Per request via `chat_template_kwargs`:

- `{"enable_thinking": false}` — answer directly, no thinking block
- `{"reasoning_effort": "low"}` — adaptive; `xhigh` (default), `medium`, `low`

Server-wide via `EXTRA_ARGS='--default-chat-template-kwargs {"enable_thinking":false}'`.

### Sampling

Qwen's recommendation, matching the shipped `generation_config.json`:

- Thinking: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`
- Non-thinking: `temperature=0.7`, `top_p=0.80`, `top_k=20`, `presence_penalty=1.5`

## References

- [vLLM recipe for Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [Model card](https://huggingface.co/Qwen/Qwen3.8-27B)
