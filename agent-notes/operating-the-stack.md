# Operating the stack

Reference material for running the two profiles on this box. Everything below
was measured or read off a running server, not assumed.

Hardware: one RTX 5090, 32,607 MiB total, **31.4 GiB usable**, ~1 GiB of that
held by the desktop compositor. 30 GiB host RAM. That usable figure decides
nearly every choice here.

## Build and run

```sh
podman build -t vllm-5090 .

hf download Inferact/Qwen3.8-27B-NVFP4      # ~25 GB
hf download nvidia/Qwen3.6-35B-A3B-NVFP4    # ~23 GB

./run.sh                                    # Qwen3.8-27B
./run.sh -p qwen3.6-35b-a3b                 # Qwen3.6-35B-A3B
```

Weights are not baked into the image; the host Hugging Face cache is mounted at
`/root/.cache/huggingface`. The image stays ~19 GB and adding a model is a
download plus an env file, not a rebuild. Only one profile can hold the GPU at
a time.

```sh
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B",
       "messages":[{"role":"user","content":"Give me three primes above 100."}],
       "temperature":1.0,"top_p":0.95,"max_tokens":2048}'
```

## Profiles

| Profile | Checkpoint | Shape |
| --- | --- | --- |
| `qwen3.8-27b` (default) | `Inferact/Qwen3.8-27B-NVFP4` | Dense 27B, all params active |
| `qwen3.6-35b-a3b` | `nvidia/Qwen3.6-35B-A3B-NVFP4` | MoE, 35B total / **3B active** |

Each profile is a plain env file in `profiles/`, read by `run.sh` and turned
into a `vllm serve` command line by `entrypoint.sh`. Override for one run,
either as an argument or from the environment:

```sh
./run.sh -p qwen3.6-35b-a3b SPEC_DECODE=1
MAX_MODEL_LEN=16384 ./run.sh
./run.sh --max-num-batched-tokens 4096      # raw vllm serve flags
```

| Var | Notes |
| --- | --- |
| `MODEL` | Any HF id or mounted path |
| `MAX_MODEL_LEN` | Native context is 262,144 on both; the card is the limit |
| `MAX_NUM_SEQS` | Lower = more KV pool per sequence |
| `GPU_MEMORY_UTILIZATION` | |
| `KV_CACHE_DTYPE` | `auto` for BF16 KV, at ~20% less KV pool |
| `TOOL_CALL_PARSER` | `qwen3_coder` for 27B, `qwen3_xml` for 35B-A3B |
| `ENFORCE_EAGER` | Required for the 27B, not for the MoE — see below |
| `LANGUAGE_MODEL_ONLY` | `1` drops the vision tower: no image/video input, more KV |
| `SPEC_DECODE` / `SPEC_CONFIG` | MTP draft head; depth and MoE backend are per-model |
| `EXTRA_ARGS` | Raw `vllm serve` flags |

## Qwen3.6-35B-A3B

The flag set in `profiles/qwen3.6-35b-a3b.env` is **not hand-tuned**. It is the
vLLM recipe's verified `rtx_5090` profile, lifted from
`variants.nvfp4.hardware_overrides.rtx_5090` in
[the recipe JSON](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B.json). Reading
the JSON API beat reconstructing flags from the prose page — worth remembering
for the next model.

FlashInfer's TRT-LLM attention kernels are what require `--block-size 128` and
`VLLM_HAS_FLASHINFER_CUBIN=1`; 32 GB is what caps context at 64K. NVFP4 on this
model needs vLLM ≥ 0.28.0 — 0.24.0 loads the checkpoint but FlashInfer only
selects its XQA decode kernel on sm120 from 0.28.0.

`SPEC_DECODE=1` buys ~25% single-stream (237 → 300 tok/s, 0.40 acceptance) and
costs most of the KV pool (380,817 → 135,084 tokens). Still more headroom than
the 27B has with MTP off.

## Qwen3.8-27B

`ENFORCE_EAGER=1` decides whether this profile starts at all. CUDA graph capture
allocates *outside* the `--gpu-memory-utilization` budget, so with 24.6 GiB of
weights it dies during capture no matter how that fraction is set. The MoE has
room and runs with graphs on — a good part of why it is so much faster.

`SPEC_DECODE=1` here **requires `LANGUAGE_MODEL_ONLY=1`**. With vision loaded it
OOMs asking for exactly 2.37 GiB — that is `248,320 × 5,120` in BF16, because
`vllm/model_executor/models/qwen3_5_mtp.py` gives the draft head its own
`VocabParallelEmbedding` even though the checkpoint sets
`mtp_use_dedicated_embeddings: false`. Lowering `GPU_MEMORY_UTILIZATION` does
not help; that allocation happens after profiling, outside the budget. Looks
like an upstream bug worth reporting; not chased further.

### The official Qwen3.8-27B weights

The obvious first choice, and the one thing that does not really work here.

| Checkpoint | Weights | Result |
| --- | --- | --- |
| `Qwen/Qwen3.8-27B` (BF16) | 55.6 GB | Never starts — ~24 GB would spill to a 30 GiB host |
| `Qwen/Qwen3.8-27B-FP8` | ~28.6 GiB | Runs, but only text-only at 4K context |

The FP8 failure is not a startup failure, which is what makes it nasty. At 8K
context it profiles fine, reports a 21,845-token KV pool and prints
`Application startup complete` — then dies on the *first real request* with
`Tried to allocate 394.00 MiB. GPU 0 ... 151.75 MiB is free`. Memory profiling
underestimates the true activation peak when weights already occupy 91% of the
card. Backing right off does produce a working server:

```sh
MODEL=Qwen/Qwen3.8-27B-FP8 MAX_MODEL_LEN=4096 MAX_NUM_SEQS=2 \
  GPU_MEMORY_UTILIZATION=0.93 LANGUAGE_MODEL_ONLY=1 \
  EXTRA_ARGS='--max-num-batched-tokens 1024' ./run.sh
```

## Benchmarking

`bench.py` is the standard measurement — a synthetic concurrency sweep against
a running server. Stdlib only, so it runs anywhere including inside the
container.

```sh
./bench.py --label "qwen3.6-35b-a3b baseline" --out docs/results.json
./bench.py --concurrency 1,2,4,8,16,32 --output-tokens 512
```

The page's headline charts (single-stream decode, KV pool) are the `runs`
table, produced by `bench.py --single`: one request at a time on an idle
server, 300 output tokens cut from a longer essay (not `ignore_eos`: a model
forced past its end writes repetitive text, and the first scripted MTP run
measured 452 tok/s at 0.56 acceptance on it, against 296 at 0.40 on natural
text), a cold request discarded and the median of three kept; the KV pool and
concurrency read from the `vllm:cache_config_info`
gauge on `/metrics`, the weights from the HF cache, vision / CUDA graphs / MTP
from the profile plus overrides. `runs.sh` does it for every configuration on
the page, starting and stopping each server (needs `DETACH=1`, which `run.sh`
now takes), about 25 minutes for the six. Until 2026-09-07 that table was
measured by hand, which is why Nemotron was missing from it.

```sh
./runs.sh                                   # all six
./runs.sh nemotron-default qwen36-moe-mtp   # just these ids
```

Design choices worth knowing when reading the numbers:

- **`ignore_eos: true`.** Every request emits exactly `--output-tokens` tokens,
  so tok/s is a property of the server rather than of how chatty the model felt.
- **Salted prompts.** Each request gets a unique prefix. The MoE profile has
  `--enable-prefix-caching` on; identical prompts would measure the cache.
- **TTFT counts `reasoning` deltas too**, since these models stream reasoning
  before content.
- **Per-stream decode excludes prefill** (`(tokens-1)/(wall-ttft)`); `total_tok_s`
  is the whole batch over wall time, which is the number that matters for
  serving several callers.
- A level that fails stops the sweep and is recorded with its error, so the
  output doubles as "does this configuration work at all at this concurrency".

## Notes that apply to both

NVFP4 is a real kernel path on sm120, not an emulation fallback — vLLM logs
`Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM`.

`--reasoning-parser qwen3` is not optional in practice: both chat templates open
every assistant turn with `<think>`, so without it the whole reasoning block
lands in `message.content`. With it, reasoning arrives in a separate `reasoning`
field and `content` holds just the answer.

There is **no `--trust-remote-code`** anywhere, despite it appearing in the
recipe's suggested command. Neither checkpoint ships `.py` files or declares
`auto_map`, so it would be a no-op — and it is the flag that lets a checkpoint
execute arbitrary Python from the Hub at load time, so it should be opt-in per
model, never boilerplate.

vLLM auto-disables DeepGemm for `model_type=qwen3_5_text` on Blackwell and falls
back to CUTLASS on its own; no `VLLM_USE_DEEP_GEMM=0` needed.

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

- [vLLM recipe: Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [vLLM recipe: Qwen3.6-35B-A3B](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B)
