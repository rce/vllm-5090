# 2026-09-06 — Video generation: how long a clip costs, how long a clip fits

A pivot from serving LLMs to text-to-video diffusion on the same RTX 5090.
The question for now is purely speed and length: wall time per clip, how it
grows with clip length, and where 32 GB runs out. Nothing here judges the
pictures. Everything is merged into `docs/results.json` under `videos` and
shown on the results page; the clips themselves are in `video/out/`
(untracked).

## The landscape (September 2026)

Found without web search (HF listing pages, model cards, GitHub), so treat
release details as "what the card said on the day".

| Model | Size | Native shape | Fit on 32 GB | Status here |
| --- | --- | --- | --- | --- |
| **Wan2.2 TI2V-5B** (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`) | 5B DiT + umT5-xxl | 1280×704, 24 fps, 121 frames, 50 steps, CFG 5 | yes, with the text encoder dropped | **measured**, below |
| **HunyuanVideo-1.5** (`hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v`) | 8.3B DiT + Qwen2.5-VL-7B + byT5 | 1280×720, 24 fps, 121 frames, 50 steps, CFG 6 (guider) | yes, same trick, plus an attention patch | **measured** to 49 frames, then stopped: too slow to be interesting |
| **LTX-2.5 distilled** (`Lightricks/LTX-2.5-Diffusers`) | 22B DiT + Gemma-12B + 12 GB of text connectors, joint audio | 960×544, 24 fps, 121 frames, 8 steps, no CFG | fp8 weights with bf16 compute (see below); gated licence | **measured**, the fast one |
| **Wan2.2-TI2V-5B-Turbo** (`yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers`) | the 5B above, step- and CFG-distilled (Self-Forcing/DMD) | same shape, 4 steps, no CFG | yes | **measured**; VAE-decode-bound |
| **MiniMax-H3** | 33B dense omni (video + stereo audio) | 768p, 4–15 s | community NVFP4 (12.5 GB DiT) via ComfyUI only | out of scope for the diffusers harness |

Skipped: Sulphur-2 (an uncensored LTX-2.3 finetune), the ComfyUI-only GGUF
repacks of Hunyuan, Wan2.1.

## The harness

`video.py` inside `Containerfile.video` (diffusers pinned to main commit
`c5469b7ceb60`, layered on the vLLM 0.28.0 image because its torch 2.13+cu130
is already proven on sm120). One fixed prompt and seed; clips of growing frame
count at the model's native resolution and step count; per clip: wall time,
per-step time from wrapping `scheduler.step`, VAE decode time, seconds of video
per wall minute, peak allocated VRAM. A CUDA OOM is recorded and stops the
sweep.

### Memory: the text encoder is the problem, not the transformer

Straight `pipe.to("cuda")` on Wan2.2-5B puts 22.5 GB of weights on the card —
11 GB of which is umT5-xxl, used once. 25 frames then OOMs at 1280×704.
`enable_model_cpu_offload()` fixes the GPU and kills the host instead: 121
frames pushed the process to 28.7 GB RSS and the OOM-killer ended it (30 GB of
host RAM here).

What works, and is now how `video.py` loads every model: load the pipeline
twice. First with only the text-encoder components (`transformer=None,
vae=None, ...` — diffusers skips `None` components), encode the one prompt,
free it. Then load it again without the encoders and pass the embeddings in.
Static GPU footprint for Wan2.2-5B drops from 22.5 GB to 12.0 GB (transformer
bf16 9.5 GB + VAE fp32 2.7 GB), host RAM never holds encoder and transformer
at once, and VAE tiling is on for the decode. HunyuanVideo-1.5's `__init__`
dereferences `vae.config` unconditionally, so its encoder stage keeps the
(small) VAE.

### HunyuanVideo-1.5: the attention mask

The diffusers port builds a dense `seq × seq` boolean mask in every attention
layer and passes it to SDPA. That mask is 12 GB at 121 frames of 720p, and
SDPA with an arbitrary mask cannot use the flash kernel (80 ms vs 26 ms per
call at 26K tokens on this card). The mask only hides text padding, and the
transformer reorders tokens so padding is always a trailing suffix — so
`video.py` replaces the processor with one that slices the padded tail off
before attention and zero-fills it after. Padded positions never influence
video tokens, so the output is unchanged. 25 frames: 12.0 → 5.5 s/step,
22.7 → 20.9 GB peak. The model card recommends `flash_hub` via `kernels` for
the same reason; neither is installed in the image.

### LTX-2.5: a 35 GB transformer on a 32 GB card with 30 GB of RAM

The distilled 22B DiT is 35 GB in bf16 — over the card, and over the host's
RAM, so it cannot even be loaded once to be quantised, and the image has no
quantisation library anyway. What `video.py` does: `from_pretrained` with
`torch_dtype=float8_e4m3fn` and `device_map="cuda"`, so each tensor is cast
as it streams off the shard and lands on the GPU (18 GB; host RAM peaks at
~9 GB). Then diffusers' layerwise casting hooks upcast each linear layer to
bf16 for its own forward. Two details: the norms and adaLN tables are
reloaded in bf16 afterwards (fp8 has three mantissa bits), and the hooks are
applied with an empty skip list — the default list exempts anything under a
`norm` path, which is where LTX keeps its adaLN projections, and an fp8
linear without a hook fails at the first matmul. Unscaled fp8 loses some
precision against the fp8-scaled checkpoints Lightricks ships for ComfyUI;
the clips look right, and nothing here measures more than that.

The rest of the pipeline needed the same two-stage treatment with two extra
wrinkles. Gemma-12B (22 GB) and the text connectors (12 GB) both belong to
the encoder side but do not fit the card together, so the encoder stage
swaps them: encode, drop Gemma, run the connectors. And the connectors run
inside `__call__`, not `encode_prompt`, so the generator stage gets a stub
that returns the precomputed result. Static GPU footprint for generation:
19.8 GB (transformer, video VAE, audio VAE, vocoder). Audio is generated
jointly and muxed into the mp4 (`av` was added to the image for it).

## Wan2.2 TI2V-5B, 1280×704, 50 steps, CFG 5

| Frames | Clip | Wall | Denoise | s / step | Decode | Video s / min | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 1.0 s | 56 s | 46 s | 0.93 | 9.7 s | 1.11 | 13.8 GB |
| 49 | 2.0 s | 125 s | 106 s | 2.12 | 19.1 s | 0.98 | 14.4 GB |
| 81 | 3.4 s | 230 s | 199 s | 3.97 | 31.4 s | 0.88 | 15.9 GB |
| 121 | 5.0 s | 377 s | 330 s | 6.60 | 46.5 s | 0.80 | 17.8 GB |
| 161 | 6.7 s | 570 s | 508 s | 10.16 | 62.0 s | 0.71 | 19.8 GB |
| 241 | 10.0 s | 1013 s | 919 s | 18.39 | 93.4 s | 0.59 | 23.6 GB |

Prompt encoding is not in these numbers (0.4 s once). Warm-up clip first.

**The model card's default 5 s clip costs 6.3 minutes.** That is with the
standard 50 steps and CFG (two transformer forwards per step). "Under 9 min on
a consumer GPU" from the card, comfortably.

**Cost grows as ~n^1.3, not linearly.** Per-step time goes 0.93 → 18.4 s for
25 → 241 frames: 9.6× the frames for 19.8× the time. Attention over all
space-time tokens is quadratic; the linear layers are linear; the mix lands
between. Seconds of video per wall minute halves from 1.11 at 1 s to 0.59 at
10 s. Anything longer than ~5 s is better made as chained 5 s clips
(conditioned on the previous clip's last frame, which this TI2V model supports
natively) — linear cost, at the price of drift across seams.

**VAE decode is 17% of the clip, and linear.** 9.7 s → 93 s for 25 → 241
frames, 0.39 s per frame with tiling on. Not the bottleneck, but not free.

**10 s fits with 8 GB to spare.** 23.6 GB peak allocated (27.1 GB reserved)
at 241 frames. The 32 GB ceiling is somewhere beyond 300 frames at this
resolution; the sweep stopped at 241 because the clip already costs 17 minutes,
not because it ran out of memory.

## HunyuanVideo-1.5, 1280×720, 50 steps, CFG 6

| Frames | Clip | Wall | Denoise | s / step | Decode | Video s / min | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 1.0 s | 282 s | 275 s | 5.49 | 7.5 s | 0.22 | 20.9 GB |
| 49 | 2.0 s | 730 s | 716 s | 14.32 | 13.9 s | 0.17 | 23.7 GB |

Stopped by hand after 49 frames: the 121-frame default clip would have taken
~55 minutes, and the answer was already clear. Five times slower than Wan at
the same length, and steeper — 2× the frames cost 2.6× per step (Wan: 2.3×),
so the gap widens with length. That is 8.3B parameters, 54 layers and two
CFG forwards per step against Wan's 5B; the attention patch above is already
in these numbers. Its 121-frame clip would land near an hour on this card;
the 480p variant of the model was not tried. Static footprint 17.9 GB.

## LTX-2.5 distilled, 960×544, 8 steps, no CFG, with audio

| Frames | Clip | Wall | Denoise | s / step | Decode | Video s / min | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 1.0 s | 4.5 s | 4.0 s | 0.51 | 0.5 s | 13.8 | 20.5 GB |
| 49 | 2.0 s | 7.2 s | 6.4 s | 0.80 | 0.8 s | 17.0 | 21.2 GB |
| 97 | 4.0 s | 13.4 s | 11.9 s | 1.49 | 1.5 s | 18.1 | 22.4 GB |
| 121 | 5.0 s | 17.4 s | 15.6 s | 1.95 | 1.9 s | 17.4 | 23.1 GB |
| 169 | 7.0 s | 25.2 s | 22.6 s | 2.82 | 2.8 s | 16.7 | 24.4 GB |
| 241 | 10.0 s | 38.0 s | 33.7 s | 4.22 | 4.5 s | 15.9 | 26.0 GB |
| 361 | 15.0 s | — | | | | | OOM at 26.2 GB |

Prompt encoding 12 s once (loading Gemma-12B and the connectors included).

**The 5 s clip costs 17 seconds, not 6 minutes.** 22× Wan's default
recipe, 4× faster per step than Wan at the same frame count despite being
4× the parameters — and that per-step figure includes upcasting 18 GB of fp8
weights to bf16 every forward. Two things make it cheap: 8 steps without CFG
against 50 with (8 forwards vs 100), and a VAE that compresses 32×32×8
instead of 16×16×4, so a 121-frame clip is ~8K tokens instead of Wan's ~60K
and attention stops mattering.

**Cost is nearly linear in length.** 0.51 → 4.22 s/step for 25 → 241 frames:
9.6× the frames for 8.3× the time. Seconds of video per wall minute is flat
at 16–18 from 2 s to 10 s. So there is no reason to chain clips for cost on
this model; 10 s in one shot is 38 s.

**15 s does not fit.** 361 frames wanted 1.4 GB more than the card had at
29 GB in use. 10 s is the ceiling at this resolution with this loading; a
smaller resolution or offloading the VAE would push it.

**Audio is essentially free.** The audio branch runs inside the same
transformer forward; the audio VAE and vocoder add nothing visible to the
decode column.

## Wan2.2-TI2V-5B-Turbo, 1280×704, 4 steps, no CFG

| Frames | Clip | Wall | Denoise | s / step | Decode | Video s / min | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 1.0 s | 12.4 s | 2.0 s | 0.50 | 10.4 s | 5.1 | 13.8 GB |
| 49 | 2.0 s | 24.7 s | 4.4 s | 1.10 | 20.3 s | 5.0 | 14.4 GB |
| 81 | 3.4 s | 42.1 s | 8.5 s | 2.12 | 33.6 s | 4.8 | 15.9 GB |
| 121 | 5.0 s | 63.6 s | 13.9 s | 3.47 | 49.7 s | 4.8 | 17.8 GB |
| 161 | 6.7 s | 85.3 s | 21.3 s | 5.33 | 64.0 s | 4.7 | 19.7 GB |
| 241 | 10.0 s | 130.6 s | 37.5 s | 9.38 | 93.3 s | 4.6 | 23.6 GB |

**The distillation works as advertised on the denoise side and then the VAE
eats it.** Denoising a 5 s clip is 13.9 s instead of 330 s (4 forwards
instead of 100, same s/step as the base model within noise); but the tiled
fp32 VAE decode is the same 50 s it always was, so the clip takes 64 s
instead of 377 s — 6×, not 24×. Decode is 75–84% of every clip. It is
inherent: without tiling the decode does not fit at all (a 1 s clip OOMs at
27.9 GB, `wan2.2-5b-turbo 1280x704 no VAE tiling` in `results.json`), and
the model card wants the VAE in fp32. A bf16 VAE or a lighter decoder
(TAEHV-style) is where the next 3× is on this model, not the transformer.

**Still 4× slower than LTX at the same length,** and without audio. The
16×16×4 VAE means 121 frames is ~60K tokens, so s/step grows 19× from 25 to
241 frames where LTX's grows 8×.

## The fast tier: preview resolutions and frame rates

The question was whether a near-real-time feedback loop is possible by
giving up resolution and/or frame rate. Same prompt, seed and recipes,
640×352 (a third of the pixels), and for LTX also 12 fps, which it is
conditioned on and honours: 121 frames at 12 fps is a 10 s clip with
half the frames per second of motion, not a 5 s clip played slowly.

### LTX-2.5 distilled, 640×352, 12 fps

| Frames | Clip | Wall | Denoise | s / step | Decode | Video s / min | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 2.1 s | 2.5 s | 2.3 s | 0.29 | 0.3 s | 49 | 20.3 GB |
| 49 | 4.1 s | 3.5 s | 3.2 s | 0.40 | 0.3 s | 70 | 20.7 GB |
| 97 | 8.1 s | 5.6 s | 5.0 s | 0.63 | 0.6 s | 87 | 21.5 GB |
| 121 | 10.1 s | 6.6 s | 6.0 s | 0.76 | 0.6 s | 91 | 21.9 GB |

### LTX-2.5 distilled, 640×352, 24 fps

| Frames | Clip | Wall | Denoise | s / step | Decode | Video s / min | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 1.0 s | 2.6 s | 2.4 s | 0.30 | 0.3 s | 24 | 20.3 GB |
| 49 | 2.0 s | 3.7 s | 3.4 s | 0.42 | 0.3 s | 33 | 20.7 GB |
| 97 | 4.0 s | 5.9 s | 5.4 s | 0.67 | 0.6 s | 41 | 21.5 GB |
| 121 | 5.0 s | 7.1 s | 6.5 s | 0.81 | 0.6 s | 43 | 21.9 GB |
| 241 | 10.0 s | 14.0 s | 12.7 s | 1.59 | 1.3 s | 43 | 24.0 GB |
| 361 | 15.0 s | 21.3 s | 19.6 s | 2.45 | 1.8 s | 42 | 26.1 GB |

15 s fits here (26.1 GB) where it did not at 960×544; the frame count, not
the pixel count, is what the 20 GB of weights leaves room for.

### Wan2.2-TI2V-5B-Turbo, 640×352, 24 fps

| Frames | Clip | Wall | Denoise | s / step | Decode | Video s / min | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 1.0 s | 2.7 s | 0.4 s | 0.11 | 2.2 s | 23 | 13.5 GB |
| 49 | 2.0 s | 5.1 s | 0.8 s | 0.19 | 4.4 s | 24 | 13.5 GB |
| 81 | 3.4 s | 8.5 s | 1.3 s | 0.33 | 7.2 s | 24 | 13.6 GB |
| 121 | 5.0 s | 13.2 s | 2.3 s | 0.56 | 10.9 s | 23 | 13.7 GB |

**Faster than real time exists: LTX at 640×352 / 12 fps.** 10 s of video
(with audio) in 6.6 s of wall, 91 video-seconds per minute; a 2 s clip in
2.5 s. Per-clip overhead is ~2 s (the 8 steps at their minimum cost plus
decode), so anything under 4 s of video is dominated by it and the ratio
improves with length. Prompt encoding is another 10–12 s but is per prompt,
not per clip, and `generate.py` pays it once.

**Resolution buys ~2.5× on LTX; frame rate buys another ~2×** on top for
the same clip duration, because at 12 fps a 10 s clip is 121 frames instead
of 241. The two levers compound: 960×544 / 24 fps → 640×352 / 24 fps → 640×352
/ 12 fps is 38 s → 14 s → 6.6 s for 10 s of video.

**Wan Turbo does not get there.** At 640×352 the denoise is quick (2.3 s for
5 s of video) but the VAE decode is still 11 s; 23 video-seconds per minute,
flat with length, so a 5 s preview costs 13 s. Preview resolution helps Wan
by 5× (decode scales with pixels) where it helps LTX by 3×, but from a base
that is 4× worse, and no audio.

**What "near real time" would take beyond this.** The remaining per-clip
cost on LTX is the 8 denoise steps over the whole clip; a streaming
autoregressive model (Self-Forcing / CausVid style, chunk-by-chunk) would
turn that into a first-frame latency of a few hundred ms and a steady
frames-per-second thereafter, which is a different harness measure than
wall-per-clip. Not measured here.

## Caveats

- One run per level, no repeats. Diffusion at a fixed seed is deterministic
  enough that wall time repeats to ~1%.
- Wan base and Hunyuan run all 50 steps with CFG; the distilled models run
  their own recipes (4 and 8 steps, no CFG). These are different
  configurations, compared as such.
- The 640×352 LTX 24 fps sweep OOMed at 25 frames on its first attempt with
  ~10 GB held by another process on the desktop; the rerun on a clean card is
  what is recorded.
- `prep_s` is ~0 because prompt encoding was moved out of the clip; the
  first step's latent preparation is inside `denoise_s`.
- Peak VRAM is `torch.cuda.max_memory_allocated`; reserved is 1–3.5 GB higher.

## Running it

```
podman build -f Containerfile.video -t video-5090 .
./video.sh --model wan2.2-5b --label "wan2.2-5b 1280x704" --out docs/results.json --dim variant=default
./video.sh --model hunyuan-1.5 --frames 25,49,81,121 --label "hunyuan-1.5 1280x720" --out docs/results.json
./video.sh --model ltx-2.5 --label "ltx-2.5 distilled 960x544" --out docs/results.json --dim variant=default
./video.sh --model wan2.2-5b-turbo --label "wan2.2-5b-turbo 1280x704" --out docs/results.json --dim variant=default
./video.sh --model ltx-2.5 --size 640x352 --frames 25,49,97,121,241,361 --label "ltx-2.5 distilled 640x352" --out docs/results.json --dim variant=preview
./video.sh --model ltx-2.5 --size 640x352 --fps 12 --frames 25,49,97,121 --label "ltx-2.5 distilled 640x352 12fps" --out docs/results.json --dim variant=preview
./video.sh --model wan2.2-5b-turbo --size 640x352 --frames 25,49,81,121 --label "wan2.2-5b-turbo 640x352" --out docs/results.json --dim variant=preview
```

The Wan sweep to 241 frames takes ~40 minutes; detach it. The distilled
models sweep in a few minutes. `--frames`, `--size`, `--steps`,
`--guidance`, `--fps` override the model defaults (`--fps` only changes the
content for LTX, which is conditioned on it; for the others it is playback
speed); `--dim` tags the entry for the page's filters. Clips land in
`video/out/<label>/`. The Hunyuan entry in `results.json` was transcribed
from the run's table after the sweep was stopped by hand, so it lacks the
reserved-memory and export columns.

For actually making a clip rather than measuring one, `generate-video` runs
`generate.py` in the same container with the same model setups:

```
./generate-video ltx-2.5 "a red bicycle against a whitewashed wall, a cat walks past"
./generate-video ltx-2.5 "..." --seconds 8 --fps 12 --size 640x352 --seed 3 --out preview.mp4
./generate-video wan2.2-5b-turbo "..." --size 640x352
```

It prints the same denoise / decode / peak-VRAM line as the sweep and writes
to `video/gen/` (untracked) by default. Expect ~25 s of loading before the
clip on LTX (12 s of it is Gemma-12B encoding the prompt).
