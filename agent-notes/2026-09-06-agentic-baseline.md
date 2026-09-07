# 2026-09-06 — Agentic baselines: tool calls and a replayed agent loop

Two new repeatable measurements in `agentic.py`, run once on each of the four
configurations recommended as standard comparison points. Everything here is
merged into `docs/agentic.json` (originally `docs/results.json`) under `toolcalls` and `loops` and shown on the
results page.

The four configurations:

| Label | Profile | Notes | KV pool |
| --- | --- | --- | ---: |
| `qwen3.6-35b-a3b` | `qwen3.6-35b-a3b` | seqs 32, CUDA graphs | 370,189 |
| `nemotron-3.5-lightning` | `nemotron-3.5-lightning` | seqs 32, CUDA graphs | 1,497,245 |
| `qwen3.8-27b text-only, CUDA graphs` | `qwen3.8-27b` + `LANGUAGE_MODEL_ONLY=1 ENFORCE_EAGER=0 MAX_NUM_SEQS=32` | vision off | 53,399 |
| `qwen3.8-27b default` | `qwen3.8-27b` | vision on, eager, seqs 8 | 60,681 |

Neither test says anything about how *good* the model's decisions are. `tools`
checks that the plumbing works; `loop` measures the serving cost of an agent's
shape of traffic. Task success is the eval-harness question
(`2026-09-06-eval-harness-research.md`).

## 1. Tool-call round trip (`agentic.py tools`)

Six requests through `/v1/chat/completions` with two tools declared
(`get_weather(city, unit)`, `read_file(path)`), temperature 0, the chat
template's default thinking mode, checking the parsed response rather than the
raw text. A fail here is usually the tool-call or reasoning parser, not the
model — hence the leak-marker check (`<tool_call>`, `<think>`, `<TOOLCALL>`,
… appearing in `content`).

| Check | What passes |
| --- | --- |
| `single_call` | one `tool_calls` entry naming `get_weather` with `city` |
| `tool_result_roundtrip` | after a `tool` message with the result, a plain answer that quotes it |
| `parallel_calls` | asked for two cities, two calls in one turn |
| `no_false_call` | "say hello" with tools declared → plain answer, no call |
| `streaming_call` | same as `single_call` over SSE, reassembled from deltas |
| `path_argument` | `read_file` with the exact path from the prompt |

| Configuration | Passed | Failed check | Latency per check | Reasoning field |
| --- | ---: | --- | --- | --- |
| qwen3.6-35b-a3b | 6/6 | — | 0.15–0.58 s | yes |
| nemotron-3.5-lightning | 5/6 | `parallel_calls` | 0.18–0.66 s | yes |
| qwen3.8-27b text-only, CUDA graphs | 6/6 | — | 0.74–2.35 s | yes |
| qwen3.8-27b default | 6/6 | — | 1.71–7.16 s | yes |

**Nemotron makes one call at a time.** Asked for Helsinki and Oslo it returned
one `get_weather("helsinki")` and stopped with `finish=tool_calls`, presumably
to call Oslo next turn. That is a model habit, not a parser fault: the one call
it did make parsed cleanly. A client loop that feeds results back gets there in
two round trips instead of one. Worth knowing when comparing turns/minute
against Qwen, which batches both calls.

**No leaks anywhere.** All four configurations return `reasoning` as a separate
field and never let `<think>` or `<tool_call>` text into `content`. The
`qwen3_xml` tool parser and the `qwen3` / `nemotron_v3` reasoning parsers are
the right pairing for these models on 0.28.0.

**The 27B default profile is slow for this.** 7 s for a single tool call is
eager mode plus a 27B dense model thinking before it calls. The text-only +
CUDA graphs variant is 2–5× faster on every check with identical results, which
is the argument for making it the standard 27B comparison point.

## 2. Replayed agent loop (`agentic.py loop`)

A synthetic coding-agent transcript, replayed turn by turn: a 600-word system
prompt, a user task, then per turn an assistant `read_file` call and ~400 words
of Python as the tool result (~1.2K tokens). After each turn the server is asked
for the next step — 128 tokens, `ignore_eos`, thinking off, tools declared.
Sixteen turns, so the context grows from ~2.3K to ~20.6K tokens. Streaming
gives per-turn TTFT; `/metrics` gives the prefix-cache hit rate over the run.

Two modes. **Warm** is the natural case: every turn's prompt is the previous
turn's prompt plus one exchange, so a prefix cache should hit on all of it.
**Cold** prefixes the system prompt with a per-run, per-turn salt so nothing is
reusable — the cost of the same traffic without prefix caching, or with a cache
that has been evicted. Run at 1 agent and at 4 agents in parallel.

### One agent

| Configuration | Mode | TTFT turn 1 | TTFT turn 16 | TTFT p95 | Cache hit | Turns / min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen3.6-35b-a3b | warm | 0.072 s | 0.248 s | 0.233 s | 80.8% | 96.6 |
| qwen3.6-35b-a3b | cold | 0.132 s | 0.843 s | 0.797 s | 1.2% | 65.9 |
| nemotron-3.5-lightning | warm | 0.144 s | 0.143 s | 0.203 s | 79.7% | 125.9 |
| nemotron-3.5-lightning | cold | 0.179 s | 0.905 s | 0.877 s | 0.0% | 72.0 |
| qwen3.8-27b text-only, CUDA graphs | warm | 0.341 s | 0.310 s | 0.362 s | 83.0% | 26.0 |
| qwen3.8-27b text-only, CUDA graphs | cold | 0.235 s | 2.601 s | 2.461 s | 0.0% | 17.7 |
| qwen3.8-27b default | warm | 0.288 s | 0.358 s | 0.413 s | 83.0% | 11.2 |
| qwen3.8-27b default | cold | 0.287 s | 2.849 s | 2.699 s | 0.0% | 9.2 |

### Four agents in parallel

| Configuration | Mode | TTFT turn 1 | TTFT turn 16 | TTFT p95 | Cache hit | Turns / min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen3.6-35b-a3b | warm | 0.374 s | 0.712 s | 0.632 s | 79.6% | 228.3 |
| qwen3.6-35b-a3b | cold | 0.377 s | 2.155 s | 2.380 s | 0.0% | 103.2 |
| nemotron-3.5-lightning | warm | 0.542 s | 0.379 s | 0.538 s | 82.5% | 264.7 |
| nemotron-3.5-lightning | cold | 0.381 s | 1.522 s | 1.589 s | 9.3% | 108.6 |
| qwen3.8-27b text-only, CUDA graphs | warm | 0.870 s | **9.896 s** | 9.604 s | **1.3%** | 28.9 |
| qwen3.8-27b text-only, CUDA graphs | cold | 0.682 s | 9.905 s | 9.616 s | 0.0% | 28.7 |
| qwen3.8-27b default | warm | 0.786 s | **11.629 s** | 10.973 s | **7.7%** | 21.1 |
| qwen3.8-27b default | cold | 0.797 s | 11.648 s | 10.955 s | 0.0% | 20.4 |

The ~80% ceiling on the hit rate is structural. Each turn's new tokens (the
call, the ~1.2K-token tool result, the response) cannot be cached before they
are first seen, and only whole blocks are cached — and on these hybrid models
the block is large: vLLM sets the attention block to match the Mamba/DeltaNet
page (*"Setting attention block size to 2128 tokens to ensure that attention
page size is >= mamba page size"* — 2176 on Qwen3.6, 1568 on Qwen3.8), so the
trailing partial block of every prompt is recomputed every turn. Over 16 turns
that tail is about a fifth of everything prefilled.

### Findings

**Prefix caching works on Nemotron despite `--mamba-cache-mode none`.** The
profile keeps `none` from the sm120 override, and the open question in
`2026-09-06-nemotron-3.5-lightning.md` was whether that disabled prefix caching
for the Mamba hybrid. It does not: the server log says *"Mamba cache mode is
set to 'align' … by default when prefix caching is enabled"* and reports
`enable_prefix_caching=True`. Warm TTFT is flat at 0.14 s from turn 1 to turn
16 — the flattest curve of the four — and the hit rate is the same ~80% as the
Qwen MoE.

**The 27B is fine for one agent and falls over at four.** One agent warm: 83%
hits, TTFT 0.31–0.36 s at 20K context, indistinguishable in shape from the MoEs
(just 3–4× slower per turn from the dense decode). Four agents warm: 1.3% hits
(text-only) / 7.7% (default) and TTFT identical to cold — 9.9 s and 11.6 s at
the last turn. The KV pool is 53–61K tokens; four agents at ~20K each need
~80K, so the blocks from one agent's turn are evicted before its next turn
arrives. A dense 27B with a 32K window on a 32 GB card cannot hold four long
agent contexts, whatever the flags — both 27B profiles hit the same wall. If
four agents is the target workload the answer is the MoEs, or shorter contexts.

**The two 27B profiles differ only in decode speed.** Same TTFT curve, same
hit rate, same collapse at four agents; but the default profile (eager, seqs 8)
does 11 turns/min against 26 for text-only + CUDA graphs at one agent, because
128 tokens of eager dense decode is ~4 s. There is no reason to benchmark the
default profile for agent work.

**Warm-vs-cold is worth 3–8× on TTFT at 20K context.** Qwen3.6 0.25 s vs
0.84 s, Nemotron 0.14 s vs 0.91 s, 27B 0.31 s vs 2.60 s (0.36 vs 2.85 default). For an agent that
makes many short requests over a growing context this is the dominant serving
effect, larger than any of the flag differences measured elsewhere in this
repository. Anything that breaks prefix caching (per-request system prompt
changes, timestamps in the prompt, reordered tool definitions) costs this much.

**Turns per minute, one agent, warm:** Nemotron 126, Qwen3.6 97, 27B 26
(text-only) / 11 (default). At four agents: 265, 228, 29 / 21. The MoEs scale
to four agents at 2.4× the single rate; the 27B text-only does not scale at all
(the default profile does, from a much lower base, because eager decode was
leaving the card idle).

**Unexplained: Nemotron's 4-agent cold run shows 9.3% cache hits** where every
other cold run is at 0–1%. The cold salt makes every prompt unique from the
first token, so nothing should hit. Possibly the Mamba `align` mode counting
something differently at block granularity. Not chased; it does not change
the ranking.

## Caveats

- One run each, no repeats. TTFT at one agent is stable to ~10 ms; the 4-agent
  numbers depend on scheduling luck and should be treated as ±20%.
- The transcript is synthetic and the same for every model. The token counts
  differ slightly by tokenizer (Nemotron: 19.2K at turn 16, Qwen: 20.6K).
- `ignore_eos` with 128 tokens makes every turn cost the same decode; real
  agents vary. Thinking is off, so a reasoning model's real turns would be
  longer.
- Hit rate is `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total`
  over the level, including the warm-up turn's eviction effects.

## Running it

```
./agentic.py tools --label "<config>" --out docs/agentic.json --dim seqs=32 --dim graphs=on
./agentic.py loop  --label "<config>" --out docs/agentic.json --agents 1,4 --dim seqs=32 --dim graphs=on
```

The loop takes ~2 min on the MoEs, ~6 min on the 27B text-only and ~10 min on
the 27B default profile (eager, seqs 8) — detach it, it outlives a 10-minute
tool timeout. Labels are the merge key, same as
`bench.py`.
