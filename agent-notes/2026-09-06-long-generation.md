# 2026-09-06 — Long generation: 4,096 output tokens

Same 700-word prompt as every other sweep, every request forced to emit
4,096 tokens, so the only thing that moved against the standard sweeps is
the output length (the sweeps now carry an `output` dimension; the page
filters on it). Total tok/s, 256-token sweep → 4K sweep, same profile:

| conc | Qwen3.6-35B-A3B | Nemotron 3.5 Lightning | Qwen3.8-27B (text-only, graphs) |
| ---: | ---: | ---: | ---: |
| 1 | 259 → **279** | 363 → **384** | 61 → 62 |
| 2 | 376 → 424 | 486 → 579 | 107 → 110 |
| 4 | 708 → 794 | 837 → 960 | 203 → 213 |
| 8 | 1,180 → 1,325 | 1,293 → 1,461 | 392 → **300** |
| 16 | 1,861 → 2,044 | 1,873 → 2,343 | 394 → 328 |
| 32 | 2,546 → **2,934** | 2,653 → **3,677** | 389 → 350 |

Three things:

- **Both MoE hybrids get faster with longer outputs**, +15% for Qwen3.6 and
  +39% for Nemotron at 32 streams. Prefill and per-request overhead are a
  fixed cost that 4K tokens amortise, and neither model's per-token decode
  cost grows much with sequence length: 3 of 4 layers carry a constant-size
  recurrent state, and the KV cache is only for the rest.
- **Nemotron's Mamba advantage shows up here and nowhere else.** At 256
  output tokens the two tied at 32 streams (2,653 vs 2,546); at 4K Nemotron
  leads by 25% (3,677 vs 2,934) and by 117 vs 93 tok/s per stream. The
  Nemotron note predicted the 1.5M-token KV pool would only matter on long
  sequences; this is the first sweep where it does.
- **The 27B runs out of KV.** 53,399 tokens of fp8 KV at `MAX_MODEL_LEN`
  32768 is room for about ten 4.9K-token sequences. Past 8 streams the
  scheduler queues and preempts: TTFT p95 jumps from 0.6 s at 8 streams to
  89 s at 16 and 243 s at 32, per-stream decode holds at ~52 tok/s, and
  total throughput *falls* from 392 (256-token sweep) to 300–350. Every
  request still completes; it just waits. For long outputs on the 27B the
  fix is a smaller `MAX_MODEL_LEN` (more KV) or fewer streams, and the
  sweep says the ceiling is ~350 tok/s either way.

Per-stream decode at 32 streams, 4K output: Qwen3.6 93 tok/s, Nemotron
117, Qwen3.8 52 (with ~10 actually running).
