# 2026-09-06 — What to measure besides tokens per second

`bench.py` answers "how fast" and "does it fall over". It says nothing about
whether the answers are any good. This note surveys what is available for that
in September 2026 and ends with a concrete recommendation.

Everything below was checked against live sources today. Where a claim is
second-hand or unverified it says so. Several of the models involved postdate my
training data, so I read their cards rather than recalling them.

## The short version of the landscape

Three things have changed since the last time most people looked at this.

**The classic benchmark set is dead for this model class.** Not "getting weak" —
gone from the model cards. `Qwen/Qwen3.8-27B` (14 Aug 2026) reports **no** MMLU,
MMLU-Pro, MMLU-Redux, SuperGPQA, AIME, HMMT, MATH-500, IFEval, MMMU, MMBench,
MathVista, DocVQA, ChartQA or OCRBench. Four months earlier `Qwen3.6-35B-A3B`
still reported all of them. Artificial Analysis moved MMLU-Pro, GPQA-Diamond,
AIME, MATH-500 and LiveCodeBench to "Legacy Evaluations" and now weights agents
at 30% of its index, dropping GPQA-Diamond in v4.2 with the words "has now been
saturated".

**The centre of gravity moved to agentic and held-out evals.** SWE-bench Pro,
Terminal-Bench, τ³-Bench, MCPMark, BrowseComp, OSWorld 2.0. AA raised its
private/held-out weighting to ~40% explicitly to make gaming harder.

**Almost none of that is runnable here.** Agentic benchmarks want Docker-in-
Docker, a scratch VM, network access, or a graded environment. The GPU is
already the whole budget.

So the interesting question is not "which harness should I install" but "what
can a single 5090 actually say that is true".

## Harnesses

Checked by fetching docs and repos today. The column that matters most is
whether it can talk to `localhost:8000/v1` without loading weights itself.

| Tool | Latest release | Points at an OpenAI endpoint? | Verdict here |
| --- | --- | --- | --- |
| [inspect-ai](https://github.com/UKGovernmentBEIS/inspect_ai) + [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals) | 0.3.263, 2026-09-04 | Yes, cleanly | **The one to adopt** |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 0.4.13, 2026-08-31 | Yes, and torch-free | Healthy, but skip — see below |
| [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) | 0.7.3, 2026-08-28 | Yes (`OPENAI_BASE_URL`) | The VLM option, heavy |
| [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) | — | Yes (`--api-mode`) | The other VLM option, heavy |
| [OpenCompass](https://github.com/open-compass/opencompass) | 0.5.4, 2026-08-26 | Python config only | Institutional scale; skip |
| [promptfoo](https://www.promptfoo.dev/docs/providers/vllm/) | 0.122.2, 2026-08-28 | Yes | Wrong shape — see below |
| [lighteval](https://github.com/huggingface/lighteval) | 0.13.0, **2025-11-24** | `--model-base-url` is a no-op | De-prioritised; skip |
| [openbench](https://github.com/groq/openbench) | 0.5.3, **2025-12-09** | Yes (wraps inspect-ai) | Dormant; a trap — see below |
| [HELM](https://github.com/stanford-crfm/helm) | 0.5.16, 2026-04-30 | Yes, in YAML | **Maintenance mode since 2026-06-01** |
| [simple-evals](https://github.com/openai/simple-evals) | never released | **No** `base_url` at all | Frozen since 2025 |
| [evalchemy](https://github.com/mlfoundations/evalchemy) | never released | — | **Abandoned** — last commit 2025-12-23 |

Maintenance figures are from the GitHub and PyPI APIs today. The download gap is
worth noting: lm-eval pulls ~1.56M/month from PyPI against lighteval's ~19.8k.

### inspect-ai

UK AI Security Institute. Shipping two to three releases a week — 0.3.263 on
4 Sept 2026. Two ways in, both documented at
[inspect.aisi.org.uk/providers.html](https://inspect.aisi.org.uk/providers.html):

```sh
export MYSERVER_API_KEY=local
export MYSERVER_BASE_URL=http://localhost:8000/v1
inspect eval inspect_evals/gpqa_diamond --model openai-api/myserver/Qwen3.6-35B-A3B

# or, equivalently, against the vllm provider:
inspect eval inspect_evals/gpqa_diamond --model vllm/Qwen3.6-35B-A3B \
  --model-base-url http://localhost:8000/v1
```

Gotcha: it is `--model-base-url`, not `-M base_url=`. The latter raises a
TypeError.

There is also a mode where the `vllm/` provider starts and stops a server for
you. Don't use it — it would fight `run.sh` for the card.

It has quietly become the substrate everyone else builds on. METR abandoned its
own stack for it (Vivaria's README: "transitioning its internal tooling from
Vivaria to Inspect… we are ramping down new feature development"), lighteval is
migrating onto it, openbench wraps it, Epoch's MirrorCode is built on it. That
is a reasonable proxy for it still being here in a year.

`inspect_evals` carries 100+ community evals including GPQA, MMLU-Pro, IFEval,
AIME 2024/2025/2026, LiveBench, HumanEval, SWE-bench Verified,
LiveCodeBench-Pro, SciCode, GAIA, AgentDojo, BFCL, MMMU, MathVista, DocVQA and
∞Bench. Its MATH task is dead — the dataset was pulled on a DMCA claim.

What makes it the right pick is not the eval list, it is the log format. Every
sample, its prompt, its completion, its score and its finish reason land in a
viewable log. That is what you need when a score looks wrong, which it will.

### lm-evaluation-harness — why not

It is in good health — 0.4.13 on 31 Aug 2026, 89 commits in the last 90 days —
and it has the single best remote-endpoint story of anything here. `pip install
"lm_eval[api]"` pulls **no torch and no transformers**, and since 0.4.10
`tokenizer_backend="auto"` probes vLLM's `/tokenizer_info` and `/tokenize`
endpoints, so it does not even download a tokenizer.

```sh
pip install "lm_eval[api]"
lm_eval --model local-completions --tasks ifeval \
  --model_args model=Qwen3.6-35B-A3B,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=32,tokenized_requests=False
```

Note `base_url` is the *full endpoint path*, not the `/v1` root. So the reason
to skip it is not health, it is fit:

- The tasks it is uniquely good at are the loglikelihood ones — MMLU, HellaSwag,
  ARC, multiple-choice ranking. Those are exactly the benchmarks that are dead
  for a 2026 30B model.
- `local-chat-completions` cannot do loglikelihood at all, because chat APIs
  don't expose prompt logprobs. So the interesting half of the harness needs the
  raw `/v1/completions` path, which bypasses the chat template — wrong for models
  whose template opens every turn with `<think>`.
- Backend choice moves scores. Issue
  [#2851](https://github.com/EleutherAI/lm-evaluation-harness/issues/2851)
  reports the same model on the same task (`leaderboard_ifeval`) scoring
  differently under the HF backend and the vLLM backend.
- The Hugging Face Open LLM Leaderboard, which was the reason to want
  comparable lm-eval numbers, **died on 2025-03-13**. Its datasets froze that
  March and there is no v3. So "matching the leaderboard" is no longer a reason.

One thing it has that nothing else does: NVIDIA's SCORE robustness tasks —
`score_non_greedy_robustness_{mmlu_pro,agieval,math}` runs five seeds at
temperature 0.7 and reports a consistency rate next to the accuracy, plus
option-order and prompt-paraphrase robustness variants. That is a ready-made
way to measure how much of a score is noise. Worth remembering if the noise
question ever becomes the point.

### promptfoo — wrong shape

It is a product-testing tool: you write assertions about your prompts and it
tells you when a change broke them. Genuinely good at that. But its scoring
leans on LLM-as-judge, and judges are the one thing this box cannot supply
(below). Not a model-comparison harness.

### The dead and the dying

Worth stating plainly, because several of these still look alive from a distance.

- **HELM entered maintenance mode on 1 June 2026** — that is its own README's
  wording, and it redirects readers to Inspect, lm-eval, and Evalchemy. Zero
  commits in 90 days.
- **Evalchemy is abandoned.** Zero commits in 90 days, last commit 2025-12-23,
  never released to PyPI. HELM pointing at it is stale advice.
- **lighteval is de-prioritised**, not dead: four commits in 90 days, last
  release 2025-11-24. Its migration onto inspect-ai converted about 157 of 1,040
  task configs and stopped in January. It also **requires torch even for remote
  endpoints**, and its `--model-base-url` flag is declared but never forwarded —
  a silent no-op. Use `--model-args base_url=...` if you use it at all.
- **openbench is a trap in its released form.** PyPI 0.5.3 dates from 2025-12-09
  and pins `inspect-ai==0.3.125`, roughly 138 releases behind current. Install
  from git or not at all.
- **OpenAI's simple-evals has no `base_url` support whatsoever** — a code search
  returns zero hits — and its graders are hardcoded to OpenAI models. An
  `OPENAI_BASE_URL` hack would silently grade your model with itself, which is
  the exact failure this note warns about elsewhere.
- **OpenCompass** is alive and active but institutional-scale, and configured in
  Python rather than on a command line. It is worth the setup only if the VLM
  side becomes the point — it absorbed VLMEvalKit's multimodal dataset loading
  in August 2026.

## Which benchmarks are still worth running

### Dead — do not run these

MMLU · MMLU-Redux · GSM8K · MATH-500 · AIME 2024 · AIME 2025 · HumanEval ·
HumanEval+ · MBPP · IFEval · MT-Bench · AlpacaEval 2 · WildBench ·
single-needle NIAH · LV-Eval · ToolBench · GAIA v1 · DocVQA · ChartQA ·
OCRBench v1 · MMBench · MMVet · Video-MME.

Three separate reasons, worth keeping distinct:

- **Saturated.** HumanEval is 96%+. MMLU-Redux spreads 92.7–93.7 across four
  different 30B models — a one-point band. There is no signal left to read.
- **Contaminated.** GSM8K→GSM1k (Scale) found up to ~13 points of drop on freshly
  written equivalent problems for some model families. AIME 2024 is the standard
  example.
- **Broken.** "Are We Done with MMLU?" ([arXiv 2406.04127](https://arxiv.org/abs/2406.04127))
  re-annotated 5,700 MMLU questions and found a **6.49% error rate**, rising to
  **57% wrong in Virology**. A few-point MMLU delta is frequently bad answer
  keys, not capability. Epoch's error analysis extrapolates ~8% label errors on
  GPQA-Diamond too.

NIAH deserves its own line. It is saturated *and* misleading: HELMET found
synthetic needle retrieval does not predict downstream long-context performance.
A NIAH pass is not evidence of a working 262K context.

### Live, and discriminating at 20–40B

Numbers below are from the model cards themselves, fetched today. Note that
NVIDIA's numbers for Qwen3.6 differ from Qwen's own by 3–7 points on the same
benchmarks — **cross-card comparison is not safe**, only within-card.

| Axis | Benchmark | Spread across ~30B models |
| --- | --- | --- |
| Knowledge | **HLE** | Gemma4-26B 8.7 → Nemotron 3.5 11.7 → Qwen3.6 21.4 → Qwen3.8-27B 30.8 |
| | GPQA-Diamond | gpt-oss-20b ~66 → Nemotron 75.4 → Qwen3.6 86.0 → Qwen3.8 89.2 |
| Calibration | **AA-Omniscience** | Nemotron 17.5 / Qwen3.6 19.5 / Gemma4-26B 22.2 — penalises hallucination, rewards abstention. Orthogonal axis. |
| Math | AIME 2026, recent HMMT sittings | Qwen3.6 AIME26 92.7; HMMT Feb'25 90.7 → Nov'25 89.1 → **Feb'26 83.6** |
| Instruction | **IFBench** | Qwen3.6 63.7 / Nemotron 71.9 / Gemma4-26B 77.3 / Qwen3.8 79.5 |
| Long context | AA-LCR, MRCR v2 | Nemotron 52.0 / Gemma4-26B 57.6 / Qwen3.6 61.1. Gemma4 MRCR v2 8-needle @128K: **44.1** |
| Coding | SWE-bench Pro, LiveCodeBench with a date window | Qwen3.6 49.5 → Qwen3.8 61.7 |
| Agentic | Terminal-Bench (pin the version), τ³, MCPMark, BrowseComp | BrowseComp: Gemma4-26B 26.3 / Nemotron 37.0 / Qwen3.6 48.7 |

That HMMT row is the most instructive line in the whole table. Same model, same
benchmark family, three sittings: 90.7 → 89.1 → 83.6 as the questions get newer.
The seven-point drop on the freshest sitting **is** the contamination
measurement. Reporting multiple sittings is a technique worth stealing.

GPQA-Diamond has 198 items. Each one is worth 0.5 points and the label error
rate is around 8%. **Gaps under two points are not real.** More on that below.

### Version pinning is not optional

Terminal-Bench is on **4.0** (26 Aug 2026), and it moved house — the live repo
is `harbor-framework/terminal-bench`, while `laude-institute/terminal-bench`
now redirects to the legacy v1 tree, and the PyPI `terminal-bench` package is
the legacy one. The model cards report 2.0 (Qwen3.6: 51.5) and 2.1 (Qwen3.8:
73.0, Nemotron: 24.58). Their release notes describe removing tasks
"for saturation (2), refusals (2), public solutions (2)". Cross-version numbers
are meaningless. LiveCodeBench is still frozen at v6 (May 2023 – Apr 2025,
1,055 problems) — a window that closed 17 months ago, before every model here
was trained. If you run it, use `--start_date` past the model's cutoff, which is
the whole point of its versioned windows.

### LiveBench is not what its README says

The repo is alive — commits on 2–4 Sept 2026, adding configs for new frontier
models. But every question dataset on
[huggingface.co/livebench](https://huggingface.co/livebench) was last updated
**2025-04-07**, while the README still claims monthly refreshes. That claim is
roughly 17 months stale. A 2026 LiveBench score is a static-benchmark score with
an April 2025 cutoff. I could not find a 2026 question release — livebench.ai is
a JS app that does not render to a fetch, and the changelog URL 404s. **Flagging
this as the least certain finding in this note**; if someone can render the site,
it is worth rechecking.

### Arena / LMArena

`lmarena.ai` now redirects to `arena.ai`. Mid-size open models are not usefully
ranked there — gemma-4-31b sits at #66 and gemma-4-26b-a4b at #86, 50–70 Elo
below the top and inside each other's error bars. The Leaderboard Illusion paper
([arXiv 2504.20879](https://arxiv.org/abs/2504.20879)) documented that Meta
tested 27 private Llama-4 variants pre-release, that Google and OpenAI each drew
~19–20% of all battle data against ~29.7% shared by 83 open-weight models, and
that arena-specific data access produced up to 112% relative gains *on the arena
distribution*. I found no published rebuttal. Not useful here.

## Four practical problems

### 1. You cannot buy your way out of contamination

There is no off-the-shelf benchmark that is credibly clean for a model trained
in 2026. The date-windowed options are running out (LiveCodeBench v6 ends Apr
2025), the held-out splits are held out from you too (SWE-bench Pro's private
set, ARC-AGI's semi-private set), and canary strings demonstrably failed —
GPT-4-base, Claude 3.5 and Gemini have all been shown reproducing the BIG-bench
canary GUID verbatim, which proves the marked data was trained on anyway.

What actually works at this scale: **hold your own private item set and never
publish it**. Fifty questions you wrote yourself, in a file that never leaves
the box, are worth more than 12,000 questions everyone has trained on. This is
also Hamel Husain's standing advice for application evals
([hamel.dev/blog/posts/evals](https://hamel.dev/blog/posts/evals/)) and it
applies just as well to model selection.

Secondary technique, cheap here because the server exposes logprobs: **Min-K%++**
([github.com/zjysteven/mink-plus-plus](https://github.com/zjysteven/mink-plus-plus))
is black-box membership inference — average the log-prob of the k% *lowest*-
probability tokens. Genuinely novel text has surprising tokens; memorised text
does not. It needs logits and nothing else.

### 2. Non-determinism is larger than the effect you are measuring

This is the finding that should change how the results page is read.

**vLLM is not reproducible by default, and for a *server* there is exactly one
lever.** From [the reproducibility docs](https://docs.vllm.ai/en/latest/usage/reproducibility/):
"Online serving APIs cannot achieve reproducibility" except via batch
invariance. `VLLM_ENABLE_V1_MULTIPROCESSING=0` only helps offline. And even then
it only holds on the same hardware and the same vLLM version.

Batch invariance is `VLLM_BATCH_INVARIANT=1`
([docs](https://docs.vllm.ai/en/latest/features/batch_invariance/), tracking
issue [vllm#27433](https://github.com/vllm-project/vllm/issues/27433), still
marked beta and updated 4 Sept 2026). It comes from Thinking Machines' [Defeating
Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/),
whose demonstration is worth quoting: 1,000 runs of Qwen3-235B at temperature 0
produced **80 distinct completions**, diverging at token 103. With
batch-invariant kernels, 1,000 identical.

**Here is the problem for this repo.** The flags that make batch invariance work
are close to the exact complement of the flags in `profiles/`. Reading the vLLM
source, batch invariance:

- disables custom all-reduce and cascade attention
- force-disables prefix caching for the FlashInfer and Triton-MLA backends
- does not support speculative decoding — so MTP is out
- has FlashInfer **disabled in its own determinism test matrix**, pending
  [FlashInfer #2424](https://github.com/flashinfer-ai/flashinfer/issues/2424)
- overrides `--linear-backend` for NVFP4 and forces `CutlassNvFp4LinearKernel`,
  **silently falling back to `EmulationNvFp4LinearKernel` if that is
  unsupported** on the device

The `qwen3.6-35b-a3b` profile uses `--attention-backend flashinfer`,
`--moe-backend marlin`, `--enable-prefix-caching` and `--kv-cache-dtype fp8`.
Every one of those is on the collision list. So **a determinism profile is a
different server configuration from a throughput profile**, and it will be
slower. Whether the Cutlass NVFP4 kernel is even supported on sm120 I could not
determine statically — `nvfp4_scaled_mm_sm120_kernels.cu` exists in the tree,
which is suggestive, but the gate is a runtime `cutlass_fp4_supported()` call.
Watch the startup log for a line about falling back to emulation.

Nobody has published the batch-invariance overhead on consumer Blackwell with
NVFP4. vLLM ships `benchmarks/benchmark_batch_invariance.py`, which runs the
same workload with the flag off and on. That is a measurable, publishable gap.

**How big is the noise?** [arXiv 2506.09501](https://arxiv.org/abs/2506.09501)
measured up to **9 percentage points** of accuracy variation and 9,000 tokens of
response-length variation on DeepSeek-R1-Distill-Qwen-7B under bf16 greedy
decoding, purely from changing GPU count, GPU type and batch size. Nine points
is larger than most of the model differences worth arguing about.

Note also that greedy is not even available here: both Qwen cards recommend
`temperature=1.0` for thinking, and the Qwen3/QwQ cards warn that greedy causes
endless repetition. So `temperature=0` reproducibility is not the goal. Fixed
seed plus batch invariance is.

### 3. Sample sizes make most single runs meaningless

Miller's [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640)
(Anthropic) is the reference. Its headline recommendation is that new evals
should contain **at least 1,000 questions** to resolve a 3-point difference at
80% power. At n=198 — GPQA-Diamond — even ten samples per item only brings the
minimum detectable effect down from about 13 points to about 7.5.

Binomial 95% half-widths at p=0.5, which is the arithmetic worth memorising:

| n | Benchmark | 95% half-width |
| ---: | --- | ---: |
| 30 | AIME, one year | **±17.9 pp** |
| 198 | GPQA-Diamond | **±7.0 pp** |
| 500 | MATH-500 | ±4.4 pp |
| 541 | IFEval | ±4.2 pp |
| 12,032 | MMLU-Pro | **±0.9 pp** |

Confirmed empirically: [A Sober Look at Progress in Language Model
Reasoning](https://arxiv.org/abs/2504.07086) measured pass@1 standard deviation
of **up to 15 points on AIME'24 across 20 seeds**, and found that temperature
and top_p choice alone shifted results by up to 15%.

Three things follow, and they are all free:

1. **Pair every comparison.** Score both configurations on the identical item
   set and analyse per-item differences, not the two aggregate scores. Costs
   nothing, cuts variance by roughly a third. [arXiv
   2512.21326](https://arxiv.org/abs/2512.21326) reports the minimum detectable
   difference on HumanEval dropping from ~12 pp unpaired to ~2–4 pp paired.
2. **Prefer more samples per item over more items** when GPU-bound. Prediction
   noise exceeds data noise by 2–6× in that same work.
3. **Bootstrap, don't CLT.** [Don't Use the CLT in LLM Evals With Fewer Than a
   Few Hundred Datapoints](https://arxiv.org/abs/2503.01747) (ICML 2025) is a
   direct rebuttal to Miller's first recommendation at small n.

And note what the model cards do: Qwen samples AIME **64×** and GPQA-Diamond
**10×**; Qwen3.8's card reports QwenSWEBench at avg@3; Qwen3.6 reports
Terminal-Bench 2.0 as an average of 5 runs. The o1 system card is the cleanest
illustration — the same model on AIME 2024 scores 74% pass@1, 83% cons@64, and
93% with a re-ranker over 1,000 samples. A benchmark number without a stated
protocol means nothing.

### 4. LLM-as-judge is not available on this box

This is the hard constraint, and it is worth being blunt about.

The literature says judges need to be stronger than the thing judged. Hamel
Husain: "Effective judges often use larger models or more compute than the
systems they evaluate." Here the only local models *are* the things under test.

The self-preference result is [Panickssery et al.,
2404.13076](https://arxiv.org/abs/2404.13076): models recognise their own
output above chance (GPT-4 at 73.5%), and **self-preference strength correlates
linearly with self-recognition ability**. Fine-tune a model to recognise itself
better and its self-preference rises in lockstep. Zheng et al.'s weaker number:
GPT-4 favoured itself by ~10% higher win rate, Claude-v1 by 25%. Direction and
magnitude are task-dependent — Koo et al. found the *opposite* ordering between
GPT-4 and GPT-3.5 — so you cannot correct for it with a constant.

Position and verbosity bias are worse than most people assume. MT-Bench measured
consistency-under-swap of **23.8% for Claude-v1** and 46.2% for GPT-3.5 — a
Llama-2-class judge reverses its preference 89% of the time on swap. That judge
is reading position, not content. The "repetitive list" attack (pad a correct
answer with restated non-information) fooled Claude-v1 and GPT-3.5 **91.3%** of
the time.

And on hard pairs, prompted judges are near chance: JudgeBench (ICLR 2025) put
vanilla-prompted **GPT-4o at 50.86%** on a binary task. Most fine-tuned open
judges scored *below* random — PandaLM 13.1%, JudgeLM 25–36%, Prometheus-2
34–40%.

The one encouraging result: small models are usable for **binary, objective**
checks. SLMJury (2026) has Qwen3-4B at 87.8% on closed-ended correctness, 1.7
points behind Phi-4 14B. But the same judges only reach Spearman 0.53–0.62 on
coherence and 0.36–0.42 on fluency. **Binary yes/no is fine. Open-ended quality
scoring is not.**

If a judge is ever unavoidable, the least-bad local option is a panel of small
models from **disjoint families** (Verga et al.,
[2404.18796](https://arxiv.org/abs/2404.18796) — three small models beat a single
GPT-4 judge on agreement with humans at 7× lower cost, and disjoint families are
the mechanism), with position rotation and a tie on disagreement. That is a lot
of machinery to avoid needing.

LiveBench's designers reached the same conclusion from the other end and score
everything against ground truth, citing GPT-4-Turbo judge error rates of up to
46% on hard problems.

**Conclusion: build only programmatically-verifiable scoring here.** Exact match,
regex, unit tests, schema validation, constraint checkers. IFEval's design —
25 instruction types verified entirely in code — is the model to copy even
though the benchmark itself is dead.

## Runtime, on this card

Using the repo's own measured aggregate throughput at concurrency 32:

| Profile | best total tok/s |
| --- | ---: |
| Nemotron-3.5-Lightning | 2,652 |
| Qwen3.6-35B-A3B, seqs 32 | 2,546 |
| Qwen3.8-27B, text-only, graphs | 394 |
| Qwen3.8-27B, default profile | 181 |

**The 27B is 6.5× more expensive to evaluate than either MoE.** That asymmetry
will dominate eval planning more than any benchmark choice does.

One caveat, and it is a real gap: `bench.py` has never measured a long-generation
workload. Its 2,546 figure is a 786-token prompt producing 256 tokens. A
reasoning eval generates 8–32K tokens per item, and decode slows as the KV cache
fills. The 339 tok/s figure from the 16K-prompt sweep is the other end, but that
workload is *prefill*-bound, which is a different shape. So the honest bracket
is wide. Estimates below assume 2,500 tok/s (optimistic) and 500 tok/s
(pessimistic) on the MoE.

| Eval | tokens | MoE @2500 | MoE @500 |
| --- | ---: | ---: | ---: |
| ~500 short-answer items, k=1, non-thinking | 0.5 M | 3 min | 17 min |
| GPQA-Diamond, 198 × k=8 × ~10K | 15.8 M | 1.8 h | 8.8 h |
| AIME 2026, 30 × k=32 × ~15K | 14.4 M | 1.6 h | 8.0 h |
| LiveCodeBench v6, 1055 × k=1 × ~8K | 8.4 M | 0.9 h | 4.7 h |
| MMLU-Pro full, 12,032 × k=1 × ~4K | 48 M | 5.3 h | 26.7 h |

Multiply by ~6.3 for the 27B. Full MMLU-Pro with thinking on is a multi-day job
on the dense model, and it is a dead benchmark.

Note that AIME at k=32 costs about the same as GPQA-Diamond at k=8 and tells you
less, because 30 items carries a ±18-point noise band no matter how many samples
you draw.

### The truncation trap

Unterminated generations are graded wrong, so a truncation bug looks exactly
like a capability gap. It is a documented silent failure in both major harnesses
— inspect_ai [#3582](https://github.com/UKGovernmentBEIS/inspect_ai/issues/3582)
("truncated responses that are not easily visible to end users") and lm-eval
[#2081](https://github.com/EleutherAI/lm-evaluation-harness/issues/2081) (a
`max_tokens=8192` setting that silently didn't take effect). lm-eval's MMLU-Pro
default `max_gen_toks` is 2048, which is fine for an instruct model and
catastrophic for a thinking one.

The reference numbers these models are compared against were produced at large
budgets: Qwen3.6's Terminal-Bench run used `max_tokens=80K`; NVIDIA's NVFP4
evals of the same model used 131,072. Anything smaller is a different
experiment.

**Log the finish-reason distribution on every run.** If `length` is more than a
percent or two of completions, the score is a config artefact.

## Quantisation: can you see NVFP4 damage from here?

Directly, no. BF16 for these models is 55–70 GB. There is no local baseline and
there never will be on this card. But the question is more answerable than that
makes it sound.

### Someone already measured it, for this exact model

[RedHatAI/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-NVFP4)
publishes BF16 vs NVFP4 for the same architecture, **with three seeds per
benchmark and eight on AIME** — the only card I found that reports seeds:

| | BF16 | NVFP4 |
| --- | ---: | ---: |
| GSM8K Platinum | 95.73 | 96.08 |
| IFEval | 93.09 | 92.45 |
| AIME 2025 | 92.92 | 91.25 |
| GPQA Diamond | 84.51 | 84.68 |
| MATH-500 | 84.80 | 85.00 |
| MMLU-Pro Chat | 85.32 | 84.70 |
| LiveCodeBench V6 | 77.33 | 74.67 |

Caveat: that is Red Hat's LLM Compressor checkpoint, not the
`nvidia/Qwen3.6-35B-A3B-NVFP4` ModelOpt one in `profiles/`. Different calibration
data, possibly different layer coverage. But it is the same architecture at the
same precision, and it is a vendor publishing their full `lm_eval` command lines,
which is more than most.

Red Hat's aggregate claim across model sizes
([redhat.com article, Feb 2026](https://developers.redhat.com/articles/2026/02/04/accelerating-large-language-models-nvfp4-quantization)):
**97–99% accuracy recovery at ~30B**, ~99% at 70B+, 95–98% at 7–14B, with MoE
models notably robust. Damage shrinks with model size, which is why an 8B proxy
experiment would be pessimistic and misleading.

### The one place NVFP4 damage reliably shows up

Across Red Hat's Qwen3-32B cards, MMLU-Pro drops 54.39 → 51.13 (94.0% recovery).
That is −3.3 points against a ±0.9 point confidence interval at n=12,032 — well
outside noise. And it reproduces at almost the same magnitude in the **NVFP4A16**
variant (54.48 → 51.61), which leaves activations in BF16. So the damage is
coming from the 4-bit *weights*, not from activation quantisation.

Meanwhile the scary-looking "94% reasoning recovery" on the same card is AIME24
at n=30 — a 6.9-point drop that is **two questions**, inside a ±15-point band.
Statistically nothing.

### Do not reproduce published BF16 baselines

The pitfall is not hypothetical; it shows up inside a single vendor's own
catalogue. For the same base model, Qwen3-32B, Red Hat's own cards report
baseline BBH at **62.35** on one card and **44.29** on another; GPQA at **30.12**
versus **5.53**. Same model, same nominal harness family, 18 and 25 points apart
— from lm-eval version, chat-template handling, `--fewshot_as_multiturn`, and
context truncation. Any reproduction of a published BF16 number will differ by
more than the quantisation effect being measured.

### What to measure instead: divergence

Three independent parties recommend comparing output *distributions* rather than
benchmark scores for this question.

- **Microsoft Research, [Accuracy is Not All You Need](https://arxiv.org/abs/2407.09141)**
  introduced *flips*: the share of answers that change correct↔incorrect between
  baseline and compressed model. Aggregate accuracy stayed within 1% while
  **flips reached 13.6%** — the errors cancel. Spearman correlation between
  flips and KL divergence on MMLU: **0.981**.
- **llama.cpp** shipped `llama-perplexity --kl-divergence` for exactly this
  ([PR #6936](https://github.com/ggml-org/llama.cpp/pull/6936),
  [docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)).
  Its Δp percentiles are the useful diagnostic: symmetric percentiles mean the
  quant added noise, asymmetric ones mean it lost quality.
- **Unsloth** call raw perplexity "incorrect, since output token values can
  cancel out" and KLD "one of the gold standards". Their Divergence-300 metric
  measures KLD over 32-token greedy continuations on held-out coding, math and
  agentic prompts — divergence on the distribution you care about, not Wikipedia.

Calibration for what counts as small: llama.cpp's BF16-vs-FP16 floor is mean KLD
2.5e-5. Unsloth's Q4-class GGUFs land around 0.014 — about 550× the floor.

**And here is the thing that makes this usable on this box.** vLLM's
`/v1/completions` accepts `prompt_logprobs` as a vLLM extra parameter, and
`logprobs` gives top-k on generation. So a divergence probe against the running
server needs no special tooling, no weights loaded twice, and no BF16 baseline —
only two servers compared in sequence, or one server compared against a stored
reference file.

That reframes the question from one you cannot answer to one you can:

- **Cannot answer:** how much did NVFP4 cost against BF16? No local baseline.
  Use the published Red Hat numbers and stop.
- **Can answer, and nobody else has:** does `KV_CACHE_DTYPE=fp8` change the
  model's output distribution? Does `SPEC_DECODE=1`? Does `ENFORCE_EAGER=0`?
  Does `MAX_NUM_SEQS=32` versus 8? Does `--moe-backend marlin`?

Every one of those flags is currently justified in `agent-notes/` on throughput
grounds alone. "It is 54% faster and provably identical in output distribution"
is a much better sentence than "it is 54% faster". MTP in particular *claims*
output-equivalence by construction — speculative decoding is supposed to be
lossless — and a divergence probe would actually check that on this hardware
rather than assuming it.

I could not find anyone publishing that measurement for NVFP4 on consumer
Blackwell. Neither vLLM's docs, nor NVIDIA's, nor the community.

---

# Recommendation

The gap worth exploiting: quality measurement and speed measurement live in
separate tools, and nothing joins them for a local server. That complaint shows
up verbatim on Hacker News. This repo is already half of the answer — it has the
speed half, with a results page and a schema. The other half should live next
to it, not in a separate ecosystem.

## Build first: `quality.py`, a sibling to `bench.py`

Same shape, same rules. Stdlib only, runs against the OpenAI endpoint, merges
into `docs/results.json` under a new key, records the exact protocol next to
every number. Two things in it, in this order.

**1. A divergence probe (build this first, it is the higher-value half).**

Fix a prompt set — 100–200 prompts drawn from what you actually do with these
models. Run it against configuration A, store per-token top-k logprobs. Run it
against configuration B. Report mean KLD, 99.9th-percentile KLD, same-top-1
rate, and flip rate on any prompts with a checkable answer.

Uses `/v1/completions` with `prompt_logprobs`, or `logprobs` on chat completions
for the generated side. No datasets to download, no gated Hugging Face repos, no
judge, no GPU beyond the server that is already running. Deterministic enough to
be useful because you compare distributions rather than sampled text.

This answers the question this repo generates constantly and that no public
harness will answer: *did that flag change the model, or only its speed?*
Nobody has published it for NVFP4 on sm120.

Effort: **one focused day.** Runtime: **minutes per configuration pair.**

**2. A small private correctness set.**

Fifty to a hundred items you write yourself, never published, every one scored
by code — exact match, regex, JSON schema, a unit test, a constraint check.
Cover the things you would actually notice: instruction-following under
awkward constraints, structured output, tool-call JSON validity, a handful of
long-context retrievals at 16K and 60K, and a few image questions for the two
VLMs. No judge anywhere in it.

Score with k≥4 samples per item and paired per-item comparison between models.
Report the bootstrap interval, not a bare number. Log the finish-reason
distribution.

It is uncontaminated by construction, it is the only benchmark that tracks what
you personally care about, and at that size it runs in minutes.

Effort: **one to two days**, mostly writing items. Runtime: **under 10 minutes
per model** on the MoE, under an hour on the 27B.

## Adopt, don't build: inspect-ai for the public numbers

For the handful of standard benchmarks worth having, install
`inspect-ai` + `inspect_evals` and point it at the server with the `openai-api`
provider. Do not reimplement scored benchmarks — the scoring details are where
the bodies are buried, and inspect's per-sample logs are worth the dependency on
their own.

Run, in this order:

1. **IFBench** — best instruction-following discrimination at this size (63.7 →
   79.5 across the peer set), programmatically scored, no judge, cheap.
2. **GPQA-Diamond**, k=8, treating any gap under 2 points as noise.
3. **AIME 2026** and the two most recent HMMT sittings — the sitting-to-sitting
   delta measures your own contamination exposure.
4. **LiveCodeBench with an explicit post-cutoff `--start_date`**, or not at all.

That is a few hours of setup and roughly 4–12 hours of GPU time per model for
the whole set, depending on where reality lands in the throughput bracket.

## Skip, and why

- **LLM-as-judge, entirely.** The only local models are the subjects. Prompted
  judges are near chance on hard pairs and small fine-tuned judges are worse than
  random. This is not a resource constraint you can engineer around.
- **lm-evaluation-harness.** Healthy and torch-free, so this is not a swipe at
  the project. Its unique strength is loglikelihood tasks, and those tasks are
  dead for these models — and the leaderboard that made its numbers worth
  matching shut down in March 2025. Revisit only if the SCORE robustness suite
  becomes the point.
- **HELM** (maintenance mode since 1 June 2026), **evalchemy** (abandoned, no
  commits since Dec 2025), **lighteval** (four commits in 90 days, torch
  required, `--model-base-url` silently ignored), **openbench from PyPI** (pins
  an inspect-ai 138 releases old — use git or nothing), **simple-evals** (no
  `base_url` support at all). None of these is worth the time right now.
- **OpenCompass.** Alive, but institutional-scale and Python-config-only.
  Revisit only if the VLM side becomes the point.
- **promptfoo.** Right tool, wrong problem: product regression testing, and
  judge-centric.
- **MMLU, MMLU-Pro, GSM8K, MATH-500, HumanEval, IFEval, MT-Bench, MMBench,
  DocVQA, NIAH.** Dead, contaminated, or label-broken. MMLU-Pro full is a 5–27
  hour run on the MoE and 30+ hours on the 27B for a four-point spread narrower
  than harness variance.
- **Any attempt at NVFP4-vs-BF16.** Impossible on 32 GiB, and Red Hat already
  published it for this architecture with seeds. Cite their table.
- **An 8B proxy in both precisions.** Damage is strongly size-dependent in the
  direction that makes the proxy pessimistic. It would validate the pipeline, not
  the models.
- **Agentic benchmarks** — Terminal-Bench, SWE-bench Pro, τ³, OSWorld. They are
  where the field has gone and they are the right things to care about, but they
  need container-in-container or a graded environment, and the card is the whole
  budget. Reconsider when there is a second machine. If that day comes, the two
  best-documented local paths are `swebench infer -m hosted_vllm/<model> -c
  model.model_kwargs.api_base=http://localhost:8000/v1` followed by `swebench
  eval` (~120 GB of Docker images), and
  [harbor](https://github.com/harbor-framework/terminal-bench) for Terminal-Bench
  4.0 (Python ≥3.12, 88 dependencies, Docker).
- **Arena / LMArena.** Mid-size open models cluster inside each other's error
  bars, and the sampling asymmetries are documented.

## Worth doing eventually

- Run `benchmarks/benchmark_batch_invariance.py` from the vLLM tree to get the
  batch-invariance overhead on sm120 with NVFP4, and check the startup log for
  the emulation-fallback line. Nobody has published either number. Expect to
  need a separate determinism profile — no MTP, no prefix caching, probably not
  FlashInfer.
- Extend `bench.py` with a long-generation workload (`--output-tokens 8192`).
  The current sweeps cannot predict eval runtime, and the 2,500-vs-500 tok/s
  bracket above is embarrassingly wide for a repo whose whole point is
  measurement.
- The VLM side needs `lmms-eval` or `VLMEvalKit` (both pushed within the last
  48 hours as of today). Both are heavy. Worth it only once there is a reason to
  care about the vision tower beyond "it costs KV cache".

## What I could not confirm

- **LiveBench's refresh cadence.** Repo alive, datasets stale since 2025-04-07,
  README claims monthly. The site is a JS app and the changelog 404s.
- **Whether `CutlassNvFp4LinearKernel` is supported on sm120**, and therefore
  whether batch invariance silently falls back to emulation here. Static reading
  of the vLLM tree could not resolve a runtime capability check.
- **Community sentiment on r/LocalLLaMA** — Reddit returned 403 to everything.
  The "what people actually use" read is from Hacker News and blogs only.
- **PinchBench**, which appears on NVIDIA's Nemotron card. Could not determine
  what it measures.
- Average thinking-token counts per item for these models on any benchmark.
  Nobody publishes them, so the runtime table's 4K/8K/10K/15K figures are
  estimates, not measurements.
