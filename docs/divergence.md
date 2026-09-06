# What the divergence probe measures

`quality.py` answers one question about this repo's server configurations:
**did that flag change what the model says, or only how fast it says it?**

Every knob in `profiles/` — 4-bit weights, 8-bit KV cache, speculative
decoding, CUDA graphs, a different attention kernel — is there because it made
`bench.py` faster. Speed is easy to measure. Whether the model is still the
same model afterwards is the part nobody checks, and it turns out you can check
it with nothing but the running server.

This page explains what the numbers mean. It assumes you know what a
classifier and a probability distribution are; it does not assume you know
how language models work inside.

## A language model is a classifier that runs once per word

Strip away the chat interface and a language model does one thing: given the
text so far, it outputs a probability for every possible next *token* (a word
or a piece of a word — the vocabulary here has about 150,000 of them). Then one
token is picked, appended to the text, and the whole thing runs again.

So "the model's output" for a given text is not a sentence. It is a sequence
of probability distributions, one per position, each over 150,000 classes.
The sentence you see is one sample drawn from that sequence. Two runs can
produce different sentences from the *same* model just by sampling — which is
why comparing generated text is a poor way to tell whether two configurations
are the same model.

The probe compares the distributions instead.

## The trick: fix the text, compare the predictions

Take a prompt and a continuation — a fixed, few-hundred-token piece of text.
Feed the whole thing to configuration A and ask, at every position, "what
probability did you give to each possible next token?" Do the same with
configuration B. Now at every position you have two distributions over the
same classes, computed from identical inputs, with no sampling anywhere.

This is called *teacher forcing*: the model is not choosing the text, it is
being walked through text chosen in advance and asked to predict each step.
It is exactly how the model is scored during training.

vLLM exposes this directly — a completions request with `prompt_logprobs`
returns, for every prompt position, the log-probability of the token that was
actually there plus the top-k alternatives. `quality.py capture` runs the
prompt set in `quality/prompts.jsonl` (96 prompts: code, maths, structured
output, Finnish, an agent-style system prompt, and so on) and stores those
distributions. `quality.py compare` lines two captures up position by position.

Where does the continuation come from? The first capture in a family
generates it — greedily, so the text is the model's own most-likely output on
that prompt, which is the text you actually care about it getting right.
Later captures use `--ref` to score that same text rather than their own.

## The metrics

All of these are computed over every position of every prompt — roughly
25,000 positions per comparison.

### KL divergence: how far apart are the two distributions?

Kullback–Leibler divergence is the standard "distance" between two probability
distributions. It is zero when they are identical and grows as they disagree,
weighting disagreements by how much probability is at stake. A tiny shift in
a token the model was 0.01% sure of contributes almost nothing; flipping the
top choice from 60% to 20% contributes a lot.

The report gives the mean, the median, and the tail (p99, p99.9, max). The
mean is the headline. The tail tells you whether the difference is spread
evenly (noise) or concentrated in a few positions where something structural
happened.

Calibration, from the llama.cpp project which uses the same measurement for
its quantised models:

| comparison | mean KLD |
| --- | --- |
| BF16 vs FP16 weights — numerically almost identical | 0.000025 |
| a well-made 4-bit weight quantisation vs BF16 (Unsloth's Q4 class) | about 0.014 |

So the interesting range spans three orders of magnitude, which is why the
tables show these numbers in scientific notation.

One honest caveat: the server only reports the top 20 tokens per position,
not all 150,000. Everything outside the top 20 on both sides is lumped into a
single "other" bucket. Merging classes can only *lower* KL divergence, so the
reported number is a floor on the true value, never an exaggeration. The
`support mass` column says how much probability the compared tokens covered;
in practice it is 0.999+, because the model is rarely that uncertain.

### Top-1 agreement: would greedy decoding pick the same token?

The fraction of positions where both configurations rank the same token
first. This is the most user-visible number: at temperature 0, every position
where top-1 differs is a position where the two servers would have written
different text from that point on.

99.9% sounds high. It means one position in a thousand flips — so across a
500-token answer there is roughly an even chance the two servers part ways
somewhere, because it only takes one flip and everything after it is
different text.

### Δp of the reference token: did it move the right way?

For each position, take the token that was actually in the text and look at
how much its probability changed from A to B. The report gives percentiles.

The shape matters more than the size. If p5 and p95 are mirror images
(−0.03 and +0.03), B is adding *noise*: it makes the true token more likely
about as often as less likely. If the distribution is skewed negative, B is
systematically less confident in the reference text — that is a quality loss,
not noise. This diagnostic comes from llama.cpp's quantisation work, where it
separates "this quant is a bit fuzzy" from "this quant is worse".

### Perplexity: how surprised was each configuration by the text?

Perplexity is exp(mean negative log-probability of the actual tokens) —
roughly "on average, between how many equally likely options was the model
choosing?" Lower is more confident. Because the reference text is the model's
own greedy output, perplexity is near 1 for the configuration that wrote it,
and a little higher for the others. It is reported for completeness; the KL
figures are more informative.

## The second view: what each server actually generates

Teacher forcing has a blind spot. Feeding text in as a prompt exercises the
*prefill* path — the kernels that process many tokens at once. Generating text
one token at a time exercises the *decode* path, and several of the flags
under test live only there: speculative decoding (MTP) proposes and verifies
tokens during decode; CUDA graphs replay captured decode steps; the 8-bit KV
cache is read back during decode. A configuration could be identical under
teacher forcing and different when generating.

So each capture also records the server's own greedy generation with the
distribution at each step. `compare` reports:

- **identical** — how many prompts produced token-for-token the same output
  from both servers;
- **first split** — for the ones that differed, the token index where they
  first disagreed;
- KL divergence and top-1 agreement over the shared prefix, up to that split.

After the first split the two texts are different and comparing positions
stops meaning anything, which is why the teacher-forced view is the primary
one and this is the supplement.

## Read nothing before reading the noise floor

Here is the finding that shapes everything else. Send the *same* scoring
request to the *same* server several times in a row, one at a time, and the
answers are not identical. `quality.py jitter` measures this.

On Qwen3.6-35B-A3B (the MoE model), 8 prompts × 4 repeats × 128 tokens:

| what | value |
| --- | ---: |
| mean KLD between two answers to the same request | 0.0023 |
| median | 0.000008 |
| 99th percentile | 0.04 |
| worst position | 0.15 |
| positions where the top-1 token flipped | 1.0% |

The median says almost every position is reproduced to five decimal places.
The mean is 300× the median because a few positions are not: a handful of
near-ties resolve differently each time, and once a top-1 flip happens the
model is on a different path.

Run the *full* 96-prompt capture twice on the same server, four requests in
flight at a time — the way the comparisons are actually made — and the floor is
higher still: mean KLD **0.0042** (95% CI 0.0037–0.0046), top-1 agreement
98.6%, and only 3 of 96 greedy generations come out token-for-token identical.
Batching adds its own noise on top of the sequential floor. That second number
is the one the results page draws as the hatched bar.

Why? Two things compound. GPU kernels that sum in parallel do not always add
numbers in the same order, and floating-point addition is not associative, so
results differ in the last bits from run to run. That would be harmless on its
own. But Qwen3.6-35B-A3B is a mixture-of-experts model: at every layer, a
small router picks 8 of 256 expert sub-networks for each token by ranking
scores. When two scores are nearly tied, a last-bit wobble flips the choice,
a different sub-network runs, and the output at that position changes by a
visible amount rather than a rounding error. The flip then feeds into every
later position.

This means:

1. A comparison whose mean KLD is at or below the floor is **indistinguishable
   from running the same configuration twice**. That is the best possible
   result and it is what "the flag is free" looks like.
2. A comparison well above the floor — several times — is a real difference.
3. Anything in between is ambiguous, and the bootstrap confidence interval on
   the mean (computed by resampling prompts) is there to say so.

The floor is a property of the server configuration, not the probe, and it
is different per model. Measure it for the configuration you are comparing
against, not once in general. The three models here span the whole range:

| server | run-to-run mean KLD | top-1 flips |
| --- | ---: | ---: |
| Qwen3.8-27B (dense, FP8 or NVFP4) | **0** — bit-identical, 96/96 generations reproduce | 0 |
| Qwen3.6-35B-A3B (MoE) | 0.0023 | 1.0% |
| Nemotron 3.5 Lightning (MoE, Mamba hybrid) | 0.0087 | 1.3% |

The dense model has no router to flip, so the last-bit wobble stays a
last-bit wobble and the same request always gets the same answer. That makes
it the cleanest possible subject for the probe — every non-zero number in its
comparisons is real — with one trap described in the next section.

## A real difference looks nothing like the floor

Qwen3.8-27B is the one model where a higher-precision version fits on the
card: Qwen's own FP8 checkpoint, which is effectively the original model.
Against it, the 4-bit NVFP4 checkpoint that the profile actually serves:

| | FP8 → NVFP4 |
| --- | ---: |
| mean KLD (95% CI) | **0.103** (0.095–0.113) |
| top-1 agreement | 89.9% |
| perplexity of the reference text | 1.278 → 1.426 |
| Δp of the reference token, 5th / 95th percentile | −0.29 / +0.13 |
| identical greedy generations | 0 of 96 |
| median first split | token 8 |

Every metric agrees, and they agree in the *loss* direction: perplexity went
up, the Δp distribution is lopsided toward the reference token losing
probability, and the greedy outputs part ways within the first ten tokens.
That is about seven times the "well-made 4-bit weight quantisation" reference
above, and it is spread across every prompt category — math is the least
affected (0.06), multilingual and writing the most (0.14) — so it is a
general cost, not one broken domain. Whether that matters is a question for
a task benchmark; what the probe says is that the 4-bit 27B is measurably a
different model from the FP8 one, and the 4-bit MoE server's flags
measurably are not.

**The trap.** The dense server reproduces itself exactly, so the run-to-run
floor is zero — but that is the floor for *the same configuration*. Compare
the NVFP4 server to itself with CUDA graphs on instead of off, or with the KV
cache in fp8 instead of bf16, and the answer is 0.027 either way (top-1
94.6%), with perplexity unchanged and Δp symmetric: noise-shaped, but far
above zero. Any change in which kernels run moves this model by about that
much. The likely reason is that NVFP4 quantises the *activations* to 4 bits
as well as the weights, and a value near a bucket boundary lands on one side
or the other depending on the last bits upstream — the ordinary rounding
wobble gets amplified into a visible one. The MoE server quantises only its
weights (its NVFP4 experts run through a weight-only kernel on this card)
and does not show this.

So the dense model has two floors: zero for "same server again", and about
0.03 for "same weights, any other configuration". The results page draws the
second one hatched, because that is the one a flag comparison has to clear.
Against it, the fp8 KV cache on the dense model costs nothing measurable —
0.101 from FP8 with it, 0.103 without.

## When the two views disagree, believe the generation

The MTP comparison on Qwen3.6 is the case study. Teacher-forced, it came back
looking catastrophic — mean KLD 2.5, top-1 agreement 74%, perplexity 17.6
against 1.13 for the reference. Generation-side, the same server was
indistinguishable from the floor: 4 of 96 outputs identical (floor: 3), first
split at token 72 (floor: 72), KL over the shared prefix 0.0025 (floor:
0.0025). Both cannot be true of the model.

They are not both about the model. Look inside the teacher-forced numbers and
the damage is per prompt, not per position: some prompts are at the floor,
the rest are wrong almost everywhere. Scoring one request at a time makes it
worse (59 of 96 prompts, mean KLD 4.1, perplexity 102), and the shape of the
damage gives the mechanism away. On prompts shorter than about 48 tokens,
the distribution the server returns at position *i* is a good prediction of
the token at *i+1* — one step ahead. That is exactly what the speculative
"draft head" computes: it guesses two tokens forward from each position so
the main model can verify several at once. The server is handing back the
draft head's predictions where the main model's belong. On mid-length
prompts the first hundred or so rows match nothing at any offset — a buffer
partly overwritten — and prompts over ~64 tokens are untouched. The
generation path never reads that buffer, which is why the greedy outputs
match. (The same flag on the 27B measured cleanly — but that model's prompt
template is longer, so none of its prompts fall in the affected band. It is
the length rule holding, not the dense model being immune.)

So: the teacher-forced view is the sensitive instrument, but it is reading a
server feature (`prompt_logprobs`) that not every configuration implements
faithfully. When it disagrees violently with the generation view, suspect the
instrument first. The results page keeps the MTP row so the pattern is
visible; the number in its KLD column is not a quality measurement.

## What this does not tell you

**It measures sameness, not quality.** A configuration with zero divergence
from the reference is exactly as good — or as bad — as the reference. For
the 27B the FP8 reference is close enough to the original to stand in for
it, so the 4-bit cost above is a real measurement. For the MoE models the
reference *is* the 4-bit checkpoint, because nothing larger fits on the card;
what those weights cost against the original is not measured here.

**Lower is not better, only closer.** If configuration B happens to be *more*
accurate than A on some task, the probe still reports the difference as
divergence. It cannot tell the direction. The Δp skew is a hint, not a
verdict.

**It only sees the prompt set.** 96 prompts of a few hundred tokens each, all
under 2K tokens total. Behaviour at 30K tokens of context, or on images, is
not covered. The prompt set is a file; add to it.

**The floor is large enough to hide small effects.** A change that shifts
mean KLD by a few ten-thousandths would be invisible under a floor of 0.002,
and on the dense NVFP4 model anything under about 0.03 is inside the
configuration-to-configuration wobble.
The bootstrap interval tells you when that is happening; it does not make the
effect measurable. See `agent-notes/2026-09-06-divergence-probe.md` for what
was tried to lower the floor.

## Running it

```sh
# 1. Is the server deterministic? Always first.
./quality.py jitter

# 2. Reference capture on configuration A.
./quality.py capture --label q36-ref

# 3. Restart the server as configuration B; score A's text, and generate.
./quality.py capture --label q36-fp8kv --ref quality/captures/q36-ref.json.gz

# 4. Compare, and record the summary on the results page.
./quality.py compare quality/captures/q36-ref.json.gz quality/captures/q36-fp8kv.json.gz \
    --out docs/results.json --dim kv=fp8
```

Captures are a few megabytes each and live in `quality/captures/`, which is
not tracked in git. Only the comparison summaries go into `docs/results.json`.
