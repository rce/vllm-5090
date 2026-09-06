# 2026-09-06 — The divergence probe, and what it found

`quality.py` is the "is it still the same model" companion to `bench.py`,
built from the recommendation in `2026-09-06-eval-harness-research.md`. The
reader-facing explanation of the method is `docs/divergence.md`; this note is
the engineering record — what was tried, what broke, and the numbers.

## Design, briefly

- **Teacher-forced scoring** via `/v1/completions` with the prompt passed as
  token ids, `echo: true`, `max_tokens: 0`, `prompt_logprobs: 20`. The whole
  prompt+continuation goes through prefill and the server returns the top-20
  distribution at every position. `return_tokens_as_token_ids` keeps the keys
  numeric so alignment never depends on detokenisation.
- **Generation-side comparison**: each capture also stores its own greedy
  256-token output with per-step top-20, so decode-only features (MTP, CUDA
  graphs, KV read-back) are exercised too. Reported as identical count and
  first-split index; positions past the split are not compared.
- **KL divergence over a coarsened support**: shared top-20 tokens plus one
  "other" bucket, so the reported KLD is a lower bound on the true value.
  Support mass is reported alongside (0.998 here).
- **Reference text** is the root capture's greedy output; `--ref` captures
  score that text instead of their own. `compare` refuses captures whose
  prompt token ids or scored continuation ids differ.
- **Bootstrap CI** over prompts (2,000 resamples), because positions within a
  prompt are not independent.
- **Prompt set** `quality/prompts.jsonl`: 96 prompts across 11 categories
  (Python, other code, maths, reasoning, structured output, instruction
  following, writing, summarisation of four original passages, Finnish,
  sv/de/ja/es, and agent-style system prompts with tool JSON and a PR diff).
  Short by design — under 600 tokens each — so the FP8 27B reference at 4K
  context can score all of them.

Verified in the vLLM 0.28.0 source before building: `prompt_logprobs` requests
skip the prefix-cache lookup (full recompute every time), the actual token is
always included in each position's dict, and `/tokenize` with `messages`
applies the chat template with `enable_thinking` honoured.

## The server is not deterministic, and no flag fixes that

The first self-comparison (same server, same requests, minutes apart) came back
with mean KLD 5.5e-3 and a max of 0.27. `compare(A, A)` is exactly zero, and
four identical sequential raw requests reproduced the effect, so it is the
server. `quality.py jitter` was added to measure it directly: N identical
scoring requests, one at a time, KLD between consecutive answers.

Bisect on Qwen3.6-35B-A3B, 6 prompts × 4 repeats × 128 tokens = 2,304
positions each, one flag changed from the profile at a time:

| configuration | mean KLD | p99 | max | top-1 flips |
| --- | ---: | ---: | ---: | ---: |
| profile default (fp8 KV, flashinfer/trtllm attention, marlin MoE, graphs, async) | 2.34e-3 | 3.78e-2 | 0.282 | 0.35% |
| `KV_CACHE_DTYPE=auto` | 2.10e-3 | 3.63e-2 | 0.209 | 0.61% |
| default attention backend (no `--attention-backend`/`--attention-config`) | 2.79e-3 | 5.80e-2 | 0.175 | 0.74% |
| `ENFORCE_EAGER=1` | 2.40e-3 | 4.16e-2 | 0.248 | 0.65% |
| no `--async-scheduling`, no chunked prefill, no prefix caching | 3.38e-3 | 4.30e-2 | 1.17 | 0.56% |
| `--moe-backend flashinfer_cutlass` | does not start | | | |
| `VLLM_BATCH_INVARIANT=1` | does not start | | | |

Every configuration that starts lands in the same band. The maximum is always
at the same position (`py-iso-duration`, token 5), which is a near-tie the
router resolves differently from run to run. The flag is not the cause; the
model is. Two failures are worth recording:

- **`flashinfer_cutlass` MoE**: `NvFp4 MoE backend 'FLASHINFER_CUTLASS' does
  not support the deployment configuration since kernel does not support
  quantization scheme QuantKey(u8, scale(f8e4m3fn, static, GroupShape(row=1,
  col=16)), scale2(f32, static, per_tensor), symmetric) x None`. The only NVFP4
  MoE backend that works for this checkpoint on this card is Marlin.
- **`VLLM_BATCH_INVARIANT=1`**: `VLLM_BATCH_INVARIANT forces NVFP4 linear to
  use the CUTLASS backend for deterministic execution`, then
  `CutlassNvFp4LinearKernel does not support W4A16`. The research note guessed
  it would fall back to emulation; it refuses instead. The reason is the
  checkpoint, not the GPU: `nvidia/Qwen3.6-35B-A3B-NVFP4` is
  `quant_algo: MIXED_PRECISION` with 161 `W4A16_NVFP4` layers (experts, shared
  expert) and 130 `FP8` layers (attention projections). W4A16 means 4-bit
  weights dequantised against 16-bit activations — Marlin is what that format
  *is*, not a fallback. The startup line "Your GPU does not have native support
  for FP4 computation but FP4 quantization is being used. Weight-only FP4
  compression will be used" is misleading on a 5090; it fires for the W4A16
  path regardless of the card. Batch invariance is therefore not available for
  this checkpoint at all, on any hardware.

Consequence for the probe: the floor for Qwen3.6 is ~2e-3 mean KLD and ~0.5–1%
top-1 flips, and anything compared against it has to clear that. The
reference for the Qwen3.6 family is the profile default with `KV_CACHE_DTYPE=auto`
(the least lossy configuration that runs); the floor is measured both by
`jitter` (sequential) and by a full second capture on the same server
(concurrency 4), and the latter is the one that matters for reading the table.

## Qwen3.6-35B-A3B: every flag is at the floor, except the one that broke the instrument

Reference: profile default with `KV_CACHE_DTYPE=auto`. 96 prompts, 256-token
greedy continuations, top-20, thinking on, four requests in flight. 24,576
scored positions per row.

| candidate | mean KLD | 95% CI | p99 | top-1 agree | Δp p5 / p95 | identical gens | first split p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| same server again (floor) | 4.16e-3 | 3.74–4.60e-3 | 0.058 | 98.61% | −0.042 / +0.041 | 3/96 | 72 |
| fp8 KV (profile default) | 4.87e-3 | 4.34–5.38e-3 | 0.069 | 98.60% | −0.046 / +0.046 | 6/96 | 74 |
| `ENFORCE_EAGER=1` | 4.45e-3 | 3.98–4.93e-3 | 0.065 | 98.66% | −0.043 / +0.044 | 6/96 | 70 |
| `SPEC_DECODE=1` (MTP) | 2.54 | 1.92–3.16 | 19.9 | 74.0% | −1.000 / +0.031 | 4/96 | 72 |

Reading it:

- **The floor at capture concurrency is ~2× the sequential jitter floor**
  (4.2e-3 vs 2.3e-3). Batching composition changes the reduction order, so
  the comparisons inherit both sources of noise. The generation side agrees:
  only 3 of 96 greedy outputs survive 256 tokens unchanged against the same
  server, and the median first split is token 72.
- **fp8 KV is at the floor.** 4.87e-3 against 4.16e-3; the confidence
  intervals nearly touch, Δp is symmetric to three decimals, perplexity
  1.1268 vs 1.1269. If the fp8 cache costs anything it is below what this
  probe can see on this model. That is the answer the profile needed.
- **Eager is at the floor**, as it should be — same kernels, no graph capture.
- **MTP is not a model result.** See the next section.

### The MTP row is a `prompt_logprobs` artefact

Teacher-forced, MTP looks catastrophic; generation-side it is exactly at the
floor (4/96 identical, split at 72, prefix KLD 2.48e-3 vs 2.50e-3). Those
cannot both describe the model. Taking the capture apart:

- Damage is bimodal per prompt: 80 prompts at the floor, 16 prompts with
  86–97% of positions wrong.
- Within a damaged prompt the wrong positions are contiguous from the start
  of the continuation — e.g. positions 0–105 wrong then clean for `if-table`
  (40-token prompt) and 0–101 for `math-marbles` (44-token prompt), both
  cleaning up at sequence position ~145. Shorter prompts are wrong
  throughout.
- The wrong distributions are not shifted by a position (top-1 vs token
  i±1, i±2 all under 10%) and match no other prompt's continuation. They are
  simply not this sequence's logits.

In `vllm/v1/worker/gpu_model_runner.py` (0.28.0), `sample_tokens` calls
`propose_draft_token_ids` — the MTP head's forward — *before*
`_bookkeeping_sync`, and `_bookkeeping_sync` is what slices
`hidden_states[offset : offset + num_logits]` to compute prompt logprobs.
The captures below say what that does in practice.

**Scored one request at a time (`--concurrency 1`)** the damage grows: 59/96
prompts, mean KLD 4.14 [3.47–4.81], top-1 50.7%, perplexity 101.7 vs 1.13,
Δp p50 −0.41. Generation side still at the floor (prefix KLD 2.6e-3, top-1
99.6%). Classifying each prompt by which row of the reference its returned
row *i* matches best:

| prompt length | rows 0..255 of the continuation |
| --- | --- |
| ≤ 47 tokens (43 prompts) | row *i* matches token *i+1* on ~54–63% of rows, token *i* on ~5%; the rest match nothing within ±8 |
| 48–63 tokens (16 prompts) | rows 0..~100 match nothing at any offset; rows past ~100 correct |
| ≥ 65 tokens (37 prompts) | correct throughout |

"Row *i* predicts token *i+1*" is the MTP draft head's job — it predicts
two tokens ahead from every position. So for short prompts the
`prompt_logprobs` response is the draft head's distribution, not the target
model's; 54–63% top-1 on the reference's own greedy text is about what an
MTP head scores. For mid-length prompts the leading ~100 rows are neither
head — a partially overwritten buffer. The ~48 / ~64 token thresholds and
the ~100-row extent are presumably padding-batch geometry (the padded
drafter batch rounds request sizes up) and were not chased further.

At concurrency 4 the same two signatures appear on fewer prompts (38/96),
and *which* prompts depends on batch composition rather than length alone —
a 374-token prompt was hit and several 22–44-token prompts were not.
Consistent with the damage being tied to a request's row offset in the
batch buffer, not to the request itself.

`disable_padded_drafter_batch=True` refuses to start alongside
`--async-scheduling` ("Async scheduling is not compatible with
disable_padded_drafter_batch=True"). Whether dropping async scheduling
(with or without the padded-drafter change) makes the artefact go away is being
measured; result to follow.

Recorded on the results page with `compare --suspect`, which keeps the row in
the table and leaves it off the chart. The general lesson is in
`docs/divergence.md`: when the teacher-forced view and the generation view
disagree violently, suspect the instrument.

## Qwen3.8-27B: the dense model is deterministic, and NVFP4 costs 0.10

The eval-harness research note wrote off NVFP4-vs-higher-precision as
"impossible on 32 GiB". For the 27B it is not: `Qwen/Qwen3.8-27B-FP8` loads
with `MAX_MODEL_LEN=4096 MAX_NUM_SEQS=2 GPU_MEMORY_UTILIZATION=0.92
LANGUAGE_MODEL_ONLY=1` and ~0.3 GiB to spare. FP8 weight quantisation is
in the 1e-3 class in every published measurement, so that checkpoint is a
usable stand-in for the BF16 original.

Two lessons getting it to run:

- **`prompt_logprobs` is a memory bomb.** vLLM materialises full-vocabulary
  logits for every prompt position in a prefill chunk, outside the profiled
  KV budget: 150k vocab × fp32 ≈ 0.6 MB per token, so the default 1024-token
  chunk is ~0.6 GB of un-budgeted allocation. With 0.3 GiB headroom the FP8
  server died on the first batch (`memory allocation failed with OOM ...
  771751936 bytes (free: 326369280)`), and the NVFP4+MTP server (MTP head
  weights on top, 0.92 utilisation) died the same way at prompt 43. Fix:
  `--max-num-batched-tokens 256` and capture with `--concurrency 1`. It costs
  ~7 s per prompt instead of ~2 s. Now in the `quality.py` docstring.
- The run is slow but it only has to happen once per reference; the captures
  are reusable and every comparison below except the first two was computed
  offline from stored captures.

### Floors

| server | run-to-run mean KLD | top-1 flips | full rerun |
| --- | ---: | ---: | ---: |
| FP8, eager, bf16 KV | 0.0 | 0/3072 | 96/96 identical, KLD 0.0 everywhere |
| NVFP4, eager, bf16 KV | 0.0 | 0/3072 | — |

No router, no nondeterminism. The dense server gives the same logits to the
last bit every time.

### Matrix (reference: official FP8, 24,576 positions)

| candidate | mean KLD | 95% CI | p99 | top-1 | Δp p5 / p95 | ppl | identical | first split p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| same server again | 0 | — | 0 | 100% | 0 / 0 | 1.2783 | 96/96 | — |
| NVFP4 (profile default, bf16 KV) | 0.103 | 0.095–0.113 | 1.05 | 89.9% | −0.294 / +0.134 | 1.4257 | 0/96 | 8 |
| NVFP4, fp8 KV (profile default) | 0.101 | 0.092–0.111 | 1.01 | 89.7% | −0.295 / +0.131 | 1.4246 | 0/96 | 9 |
| NVFP4, text-only, CUDA graphs, seqs 32 | 0.102 | 0.093–0.112 | 1.03 | 89.8% | −0.292 / +0.132 | 1.4238 | 0/96 | 9 |
| NVFP4, text-only, MTP (batched 256, concurrency 1) | 0.103 | 0.094–0.113 | 1.03 | 89.7% | −0.293 / +0.137 | 1.4261 | 0/96 | 9 |

Per category, NVFP4 vs FP8: math 0.060, reasoning 0.078, structured 0.088,
summarise 0.091, code-py 0.099, instruction 0.112, code-other 0.119,
finnish 0.120, agentic 0.131, writing 0.139, multilingual 0.143. Top-1
agreement 85–93% everywhere. It is a general cost.

The MTP row is a real measurement, unlike its Qwen3.6 counterpart: against
plain NVFP4 it sits at 0.027 [0.025–0.031], i.e. the between-configuration
floor below, and no prompt shows the "row *i* predicts token *i+1*" signature
(at most 8% of rows on any prompt, against 54–63% on the damaged Qwen3.6
prompts). But that is not evidence the dense model is immune. Qwen3.8's chat
template is ~42 tokens longer, so the shortest prompt here is 64 tokens — and
on Qwen3.6 every prompt of 65 tokens or more was untouched. The clean row is
consistent with the length rule, nothing more. (The first attempt at
batched-tokens 1024, concurrency 4 OOMed in `prompt_logprobs` at prompt 43;
the retry needed the same 256/1 settings as the FP8 reference.)

Kernel path (from the server log): `Detected ModelOpt NVFP4 checkpoint
(quant_algo=NVFP4)`, `Using FlashInferCutlassNvFp4LinearKernel for NVFP4
GEMM`. That is the real W4A4 path — 4-bit activations too — not a
weight-only fallback. The Qwen3.6 checkpoint, by contrast, is
`MIXED_PRECISION` with `W4A16_NVFP4` experts and runs through Marlin
weight-only (see the bisect section); its activations stay 16-bit.

For scale: llama.cpp's Q4_K_M-class quantisations land around 0.014 against
BF16 on the same measurement; this is 7× that, with a matching perplexity
rise of 11.5% on the model's own greedy text. Whether it *matters* is a
benchmark question the probe cannot answer, but it is not a subtle
difference — the greedy outputs diverge inside the first ten tokens on 90%
of prompts.

### The between-configuration floor on W4A4 is 0.027, not 0

Because the server is deterministic, the NVFP4 captures can be compared
against each other offline:

| pair | mean KLD | 95% CI | top-1 | Δp p5 / p95 | ppl A / B | first split p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eager bf16 KV → eager fp8 KV | 0.0265 | 0.0245–0.0297 | 94.7% | −0.104 / +0.100 | 1.4257 / 1.4246 | 26 |
| eager bf16 KV → graphs bf16 KV | 0.0266 | 0.0246–0.0303 | 94.6% | symmetric | 1.4257 / 1.4238 | 22 |
| eager fp8 KV → graphs bf16 KV | 0.0267 | 0.0247–0.0298 | 94.5% | symmetric | 1.4246 / 1.4238 | 24 |

Three configurations, three pairwise distances, all 0.027. If fp8 KV had a
cost of its own, the bf16→fp8 pair would stand out; it does not. Perplexity
is flat, Δp is symmetric, and none of it shows up against the FP8 reference
(0.101–0.103 for all three). This is noise, and it is 6× the MoE server's
noise despite the dense server being bit-reproducible.

Hypothesis, not proven: W4A4 quantises activations per 16-value block with
an fp8 scale. A value near a bucket edge lands on either side depending on
the last bits of the preceding layer's output, so any change in kernel
selection (CUDA graphs + torch.compile fusions, fp8 KV, a different batch
shape) becomes a scattering of bucket flips — each a rounding error, but a
large one, and then propagated. The W4A16 MoE never quantises activations
and its between-configuration distances (4.5–4.9e-3) sit right on its
run-to-run floor (4.2e-3). A way to test it: serve the same Inferact
checkpoint through a weight-only path (if a Marlin W4A16 fallback can be
forced for `quant_algo: NVFP4`) and repeat the eager→graphs pair.

Recorded on the page as "Qwen3.8 NVFP4 eager → CUDA graphs" with
`kind=floor`, since that is the floor a flag comparison on this model has to
clear. The same-server 0.0 is real but is the wrong bar.

### What this changes about the fp8 KV question

Qwen3.6: fp8 KV is at the run-to-run floor (4.9e-3 vs 4.2e-3). Qwen3.8: fp8
KV is at the between-configuration floor (0.027, same as flipping CUDA graphs)
and adds nothing against the FP8 truth. Both profile defaults stand.
The open contrast is *why* the dense model's configuration wobble is 6×
the MoE's; both are 3:1 linear/full-attention hybrids with head_dim 256, so
it is not the attention layout. The W4A4-vs-W4A16 hypothesis above is the
best candidate.

## Nemotron 3.5 Lightning: floor only

`quality.py jitter` on the profile default (fp8 KV, fp16 SSM cache): mean
KLD 8.68e-3, p50 1.1e-6, p99 0.185, max 1.15, top-1 flips 41/3072 (1.3%).
Four times the Qwen3.6 floor: 6-of-128 expert routing (Qwen3.6: 8 of 256)
plus a recurrent Mamba state carried in fp16 across most of its 52 layers.
No comparisons were run against it;
any Nemotron flag comparison needs to clear ~0.01 before it means anything.
