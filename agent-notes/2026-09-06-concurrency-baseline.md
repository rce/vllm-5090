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

## Worth trying next

- `MAX_NUM_SEQS=16` or `32` and re-run — the current cap is the thing bounding
  the plateau, and there is KV pool to spare (380,817 tokens).
- The same sweep with `SPEC_DECODE=1`. MTP helps single-stream, but speculative
  decoding usually loses its advantage under batch load; worth confirming rather
  than assuming.
- The 27B profile for comparison, expecting a much lower plateau.
- Longer prompts (4K, 16K) to see prefill cost separate from decode.
