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
| **HunyuanVideo-1.5** (`hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v`) | 8.3B DiT + Qwen2.5-VL-7B + byT5 | 1280×720, 24 fps, 121 frames, 50 steps, CFG 6 (guider) | yes, same trick, plus an attention patch | running |
| **LTX-2.5** (`Lightricks/LTX-2.5-Diffusers`) | 22B DiT + Gemma-12B, joint audio | 24 fps, distilled 8 steps | not in bf16 (35 GB transformer); fp8 needed | **gated** — the HF account has to accept the licence first |
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

## Caveats

- One run per level, no repeats. Diffusion at a fixed seed is deterministic
  enough that wall time repeats to ~1%.
- All 50 steps with CFG. A 4-step Lightning LoRA or a distilled model would
  divide the denoise column by ~10; that is a different configuration, not a
  correction to this one.
- `prep_s` is ~0 because prompt encoding was moved out of the clip; the
  first step's latent preparation is inside `denoise_s`.
- Peak VRAM is `torch.cuda.max_memory_allocated`; reserved is 1–3.5 GB higher.

## Running it

```
podman build -f Containerfile.video -t video-5090 .
./video.sh --model wan2.2-5b --label "wan2.2-5b 1280x704" --out docs/results.json --dim variant=default
./video.sh --model hunyuan-1.5 --frames 25,49,81,121 --label "hunyuan-1.5 1280x720" --out docs/results.json
```

The Wan sweep to 241 frames takes ~40 minutes; detach it. `--frames`,
`--size`, `--steps`, `--guidance` override the model defaults; `--dim` tags
the entry for the page's filters. Clips land in `video/out/<label>/`.
