# 2026-09-06 — Concurrency baseline for Qwen3.6-35B-A3B

First run of `bench.py`, against `profiles/qwen3.6-35b-a3b.env` unchanged (no
MTP). 700-word prompt (786 tokens), exactly 256 output tokens per request,
2 rounds per level.

| Concurrency | TTFT p50 | TTFT p95 | tok/s per stream | total tok/s | latency p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.067 s | 0.068 s | 280.1 | 261.9 | 0.98 s |
| 2 | 0.131 s | 0.132 s | 215.4 | 388.7 | 1.32 s |
| 4 | 0.140 s | 0.142 s | 206.9 | 746.4 | 1.38 s |
| 8 | 0.225 s | 0.228 s | 173.9 | **1223.0** | 1.69 s |
| 16 | 1.016 s | 1.918 s | 169.2 | 1210.9 | 3.40 s |

Every level succeeded — 62 requests, zero failures.

## Reading it

**The knee is at 8.** System throughput climbs steeply to ~1,223 tok/s at
concurrency 8 and then stops: 16 delivers 1,211 tok/s, statistically the same
number. Past 8 the extra requests are queueing, not computing.

**Concurrency 16 costs latency for nothing.** TTFT p50 jumps 4.5x (0.225 s →
1.016 s) and p95 8.4x (0.228 s → 1.918 s) while total throughput does not move.
That is the profile's `MAX_NUM_SEQS=8` doing exactly what it says — the ninth
concurrent request waits for a slot.

**Per-stream decode degrades gracefully.** 280 → 169 tok/s from 1 to 16 streams,
so even a saturated server still feels fast to any individual caller. The 8x
increase in load costs each stream only ~40% of its speed.

**Single-stream here is 280 tok/s**, against the ~237 tok/s measured earlier
with the ad-hoc script. The difference is workload, not variance: the earlier
number used a ~10-token prompt and let the model stop on its own, this one uses
a 786-token prompt and forces exactly 256 tokens. This is the number to quote
going forward, and the reason `bench.py` exists.

## Follow-up: MAX_NUM_SEQS=32

Confirmed — the cap was the constraint, and lifting it is close to free. Same
sweep, `MAX_NUM_SEQS=32`, KV pool 370,189 tokens (down from 379,046; more
sequence state, no meaningful loss).

| Concurrency | total tok/s @ 8 | total tok/s @ 32 | TTFT p95 @ 8 | TTFT p95 @ 32 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 262 | 259 | 0.068 s | 0.068 s |
| 2 | 389 | 376 | 0.132 s | 0.200 s |
| 4 | 746 | 709 | 0.142 s | 0.208 s |
| 8 | 1,223 | 1,180 | 0.228 s | 0.284 s |
| 16 | 1,211 | **1,861** | 1.918 s | **0.486 s** |
| 32 | — | **2,546** | — | 0.884 s |

At concurrency 16 the higher cap wins on *both* axes at once: +54% throughput
and a quarter of the tail latency. That is the queueing disappearing, not a
tradeoff. At 32 it is still climbing — 2,546 tok/s and no plateau in sight, so
the real ceiling has not been found yet.

The cost is a consistent 3-5% at concurrency 1-8. Small, and possibly noise,
but it leans the same way at every level so probably real: more sequence slots
means more per-step scheduling and state.

## Long-context follow-up: the worry was unfounded

The reason not to just raise the cap was that long prompts might preempt and
thrash. Tested it — 14,500-word prompts (~16.3K tokens), 256 output tokens,
one round per level.

| Concurrency | total tok/s @ 8 | total tok/s @ 32 | TTFT p95 @ 8 | TTFT p95 @ 32 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 158 | 157 | 0.64 s | 0.63 s |
| 2 | 203 | 195 | 1.24 s | 1.31 s |
| 4 | 260 | 254 | 2.51 s | 2.55 s |
| 8 | 298 | 304 | 4.96 s | 4.92 s |
| 16 | 309 | 306 | 11.32 s | 11.73 s |
| 32 | 304 | **339** | 24.56 s | **22.09 s** |

They are the same curve. Every request succeeded in both configurations — no
preemption thrash, no failures, no OOM. At the top level the higher cap is
slightly *better* on both axes (+11% throughput, 10% lower tail).

The reason is that this workload is **prefill-bound, not slot-bound**. Each
request drags 16.3K tokens through prefill and then emits only 256, so the
server spends nearly all its time on prompts and the number of decode slots
barely matters. Both configurations converge on the same ~300 tok/s ceiling —
about a tenth of the 2,546 tok/s the short-prompt workload reaches, which is
the honest cost of long context here.

Note also that at concurrency 32 the sequence cap was never the binding
constraint for the seqs=32 run: 32 × 16.3K = 524K KV tokens against a 370K
pool, so vLLM was scheduling in waves on KV capacity either way.

**So the profile default moves to 32.** The recipe's 8 has no measured
advantage on this card in either regime, and costs 54% of throughput in the
short-prompt one. This diverges from the recipe deliberately and the reason is
recorded here; revert the one line in `profiles/qwen3.6-35b-a3b.env` if a
workload ever shows otherwise.

## Worth trying next

- Push past 32 (64, 128) to find where throughput actually turns over.
- A long-prompt sweep (4K, 16K, 32K) — the case where the recipe's cap of 8
  probably earns its keep, and the one this workload cannot speak to.
- The same sweep with `SPEC_DECODE=1`. MTP helps single-stream, but speculative
  decoding usually loses its advantage under batch load; worth confirming rather
  than assuming.
- The 27B profile for comparison, expecting a much lower plateau.
