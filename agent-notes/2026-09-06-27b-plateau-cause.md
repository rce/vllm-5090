# 2026-09-06 — Why the 27B plateaued at ~180 tok/s

Follow-up to `2026-09-06-qwen3.8-27b-concurrency.md`, which recorded the
plateau and two guesses at its cause. One guess was wrong, the other was right
but incomplete.

## The chunked-prefill guess was wrong, and the logs said so for free

The hypothesis was that the 27B profile lacked `--enable-chunked-prefill` and
`--enable-prefix-caching`, which the MoE profile carries from the recipe's
`rtx_5090` override. Checking the engine config dump before running anything:

```
27B default: enable_prefix_caching=True, enable_chunked_prefill=True
```

**Both are vLLM V1 defaults.** The MoE profile's explicit flags are redundant —
they change nothing. Worth remembering before copying flags from a recipe and
assuming they do something.

The real difference in the config dumps:

```
27B: cudagraph_mode=NONE                 compilation mode NONE   (enforce_eager=True)
MoE: cudagraph_mode=FULL_AND_PIECEWISE   VLLM_COMPILE            (enforce_eager=False)
```

## CUDA graphs are worth 2.3x

Controlled: both text-only, `MAX_NUM_SEQS=32`, no MTP, same prompt. The only
difference is eager vs graphs.

| Concurrency | eager total | graphs total | eager tok/s/stream | graphs tok/s/stream |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 26.3 | **61.1** | 26.5 | 62.2 |
| 2 | 49.1 | **107.0** | 25.3 | 55.2 |
| 4 | 98.3 | **203.1** | 25.3 | 53.7 |
| 8 | 189.4 | **391.5** | 24.9 | 53.9 |
| 16 | 354.4 | 393.7 | 24.2 | 49.9 |
| 32 | 358.9 | 388.8 | 22.7 | 49.8 |

2.3x single-stream and 2.1x at concurrency 8. The gap narrows at 16+ because
graphs cost KV pool (53,399 tokens against eager's 121,362), so the graph build
runs out of sequence room sooner. Eager catches up on aggregate throughput but
never on per-caller latency.

## So the 180 plateau was three things stacked

Unpicking the default profile (vision on, eager, `MAX_NUM_SEQS=8`) one change at
a time, at concurrency 32:

| Change | total tok/s |
| --- | ---: |
| default | 181 |
| `MAX_NUM_SEQS=32` | 238 |
| …and text-only (KV 60,681 → 121,362) | 359 |
| …and CUDA graphs | 389 |

The sequence cap and the vision tower's KV cost did most of it; graphs did the
rest at high concurrency and *all* of it at low concurrency. None of it was the
scheduler.

## MTP or graphs, not both

They cannot be combined on this card. At 32K context the KV cache is squeezed to
0.81 GiB against the 1.87 GiB one full-length sequence needs; the engine refuses
to start with a clear `ValueError` rather than dying later. Dropping to 16K
context and capping graph capture at 16 still leaves 1.2 GiB against 1.35 GiB
needed. Not a near miss worth tuning around — the two features want the same
couple of gigabytes.

Which to pick depends on the workload, and they genuinely differ:

| | single stream | concurrency 8 |
| --- | ---: | ---: |
| MTP, text-only, eager | **72.3** | 260.2 |
| CUDA graphs, text-only | 61.1 | **391.5** |

MTP wins when one caller wants an answer fast. Graphs win by 50% once several
callers are in flight. Speculative decoding gets less useful as the batch fills,
which is the usual shape.

## Profile unchanged, again

All of this is text-only, and the default profile keeps vision. Dropping the
vision tower on a native VLM is a capability decision, not a tuning one, so it
stays opt-in — but if the 27B is ever used as a text model in earnest, the
combination to reach for is `LANGUAGE_MODEL_ONLY=1 ENFORCE_EAGER=0
MAX_NUM_SEQS=32`, which is 2x the default at every concurrency level.
