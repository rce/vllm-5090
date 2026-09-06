#!/usr/bin/env python3
"""Length/speed sweep for video generation models on this card.

Answers, for one model at one resolution, as the clip gets longer:

  1. Does it still fit -- or where does 32 GB run out?
  2. What does a second of video cost: wall time per clip, time per denoise
     step, VAE decode time, seconds of video per wall minute, peak VRAM.

Runs inside the video-5090 container (see Containerfile.video) and drives
diffusers directly. One fixed prompt and seed: the cost of a clip depends on
its shape, not on what is in it. Quality is not measured here at all -- the
clips land in video/out/ for eyeballing, and that is as far as it goes.

  ./video.sh --model wan2.2-5b                              # default sweep
  ./video.sh --model wan2.2-5b --frames 49,121,241 --steps 20
  ./video.sh --model ltx-2.5 --label "ltx-2.5 distilled 960x544" --out docs/results.json
"""

import argparse
import datetime as dt
import json
import os
import sys
import time

import torch

PROMPT = ("A red vintage bicycle leans against a whitewashed wall in a narrow "
          "Mediterranean street at golden hour; a tabby cat walks past, pauses, "
          "and looks up at the camera as laundry sways on a line overhead.")

# Wan's standard negative prompt, from the model card.
WAN_NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
                "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
                "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
                "杂乱的背景，三条腿，背景人很多，倒着走")


# --- model registry -----------------------------------------------------------
# Each entry: the pipeline class, which of its components are the text
# encoder side, how to encode the prompt with them, the model card's default
# shape, the frame-count rule its VAE imposes, and the call kwargs that differ
# per model.
#
# The pipeline is loaded twice. First with only the text-encoder components,
# to encode the one prompt; then without them, for generation. The encoders
# are big (umT5-xxl 11 GB, Qwen2.5-VL-7B 13 GB, Gemma-12B 22 GB) and dead
# weight after one forward pass -- and encoder + transformer together do not
# fit this host's 30 GB of RAM for the larger models, let alone the GPU.

def load_wan(repo, dtype, skip):
    from diffusers import AutoencoderKLWan, WanPipeline
    if "vae" not in skip:  # the model card wants the VAE in fp32
        skip = dict(skip, vae=AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32))
    return WanPipeline.from_pretrained(repo, torch_dtype=dtype, **skip)


def load_ltx2(repo, dtype, skip):
    from diffusers import LTX2Pipeline
    return LTX2Pipeline.from_pretrained(repo, torch_dtype=dtype, **skip)


def load_hunyuan15(repo, dtype, skip):
    from diffusers import HunyuanVideo15Pipeline
    patch_hunyuan15_attention()
    return HunyuanVideo15Pipeline.from_pretrained(repo, torch_dtype=dtype, **skip)


def patch_hunyuan15_attention():
    """Drop the dense attention mask from HunyuanVideo-1.5's attention.

    The diffusers port materialises a seq x seq boolean mask in every layer
    (12 GB at 121 frames of 720p) and hands it to SDPA, which then cannot use
    the flash kernel. The mask only hides text padding, and the transformer
    reorders tokens so padding is always a trailing suffix -- so slice the
    tail off before attention and zero-fill it after. Padded positions never
    reach the video tokens, so the output is the same.
    """
    import torch.nn.functional as F
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.transformers import transformer_hunyuan_video15 as m

    cls = m.HunyuanVideo15AttnProcessor2_0
    if getattr(cls, "_prefix_patched", False):
        return

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(2, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        if encoder_hidden_states is not None:
            eq = attn.add_q_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1))
            ek = attn.add_k_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1))
            ev = attn.add_v_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1))
            if attn.norm_added_q is not None:
                eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None:
                ek = attn.norm_added_k(ek)
            query = torch.cat([query, eq], dim=1)
            key = torch.cat([key, ek], dim=1)
            value = torch.cat([value, ev], dim=1)

        seq_len = query.shape[1]
        mask = F.pad(attention_mask, (seq_len - attention_mask.shape[1], 0), value=True).bool()
        n = int(mask.sum(dim=1).min())
        out = dispatch_attention_fn(query[:, :n], key[:, :n], value[:, :n], attn_mask=None, dropout_p=0.0,
                                    is_causal=False, backend=self._attention_backend,
                                    parallel_config=self._parallel_config)
        if n < seq_len:
            out = F.pad(out, (0, 0, 0, 0, 0, seq_len - n))
        hidden_states = out.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            k = encoder_hidden_states.shape[1]
            hidden_states, encoder_hidden_states = hidden_states[:, :-k], hidden_states[:, -k:]
            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states

    cls.__call__ = __call__
    cls._prefix_patched = True


def component_names(repo):
    from diffusers import DiffusionPipeline
    cfg = DiffusionPipeline.load_config(repo)
    return [k for k, v in cfg.items() if not k.startswith("_") and isinstance(v, list) and v[0]]


def encode_wan(pipe, prompt, negative, cfg):
    pe, ne = pipe.encode_prompt(prompt, negative_prompt=negative, do_classifier_free_guidance=cfg)
    return dict(prompt_embeds=pe, negative_prompt_embeds=ne)


def encode_ltx2(pipe, prompt, negative, cfg):
    pe, pm, ne, nm = pipe.encode_prompt(prompt, negative_prompt=negative, do_classifier_free_guidance=cfg)
    return dict(prompt_embeds=pe, prompt_attention_mask=pm,
                negative_prompt_embeds=ne, negative_prompt_attention_mask=nm)


def encode_hunyuan15(pipe, prompt, negative, cfg):
    pe, pm, pe2, pm2 = pipe.encode_prompt(prompt)
    out = dict(prompt_embeds=pe, prompt_embeds_mask=pm, prompt_embeds_2=pe2, prompt_embeds_mask_2=pm2)
    if cfg:
        ne, nm, ne2, nm2 = pipe.encode_prompt(negative or "")
        out.update(negative_prompt_embeds=ne, negative_prompt_embeds_mask=nm,
                   negative_prompt_embeds_2=ne2, negative_prompt_embeds_mask_2=nm2)
    return out


MODELS = {
    "wan2.2-5b": dict(
        repo="Wan-AI/Wan2.2-TI2V-5B-Diffusers", load=load_wan, encode=encode_wan,
        encoders=("text_encoder", "tokenizer"),
        width=1280, height=704, fps=24, steps=50, guidance=5.0, frame_step=4,
        frames=[25, 49, 81, 121, 161, 241], negative=WAN_NEGATIVE,
        kwargs=dict(),
    ),
    "ltx-2.5": dict(
        repo="Lightricks/LTX-2.5-Diffusers", load=load_ltx2, encode=encode_ltx2,
        encoders=("text_encoder", "tokenizer", "connectors"),
        width=960, height=544, fps=24, steps=8, guidance=1.0, frame_step=8,
        frames=[25, 49, 97, 121, 169, 241], negative=None,
        kwargs=dict(frame_rate=24.0),
    ),
    "hunyuan-1.5": dict(
        repo="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        load=load_hunyuan15, encode=encode_hunyuan15,
        encoders=("text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"),
        encoder_extra=("vae",),  # the pipeline's __init__ dereferences vae.config unconditionally
        guider=True,  # CFG lives in a guider component, not a guidance_scale kwarg
        width=1280, height=720, fps=24, steps=50, guidance=6.0, frame_step=4,
        frames=[25, 49, 81, 121, 161, 241], negative=None,
        kwargs=dict(),
    ),
}


def merge_into(path, entry):
    """Add or replace this entry in a results.json, keyed by label."""
    try:
        with open(path) as f:
            doc = json.load(f)
    except FileNotFoundError:
        doc = {"schema_version": 1}
    items = doc.setdefault("videos", [])
    for i, existing in enumerate(items):
        if existing.get("label") == entry["label"]:
            items[i] = entry
            break
    else:
        items.append(entry)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_dims(items):
    dims = {}
    for it in items or []:
        k, _, v = it.partition("=")
        dims[k] = v
    return dims


def gb(n):
    return round(n / 2**30, 2)


def encode_once(spec, prompt, cfg):
    """Load only the text-encoder side, encode the prompt, free it all again."""
    import gc
    names = component_names(spec["repo"])
    keep = spec["encoders"] + spec.get("encoder_extra", ())
    pipe = spec["load"](spec["repo"], torch.bfloat16, {n: None for n in names if n not in keep})
    pipe.to("cuda")
    with torch.inference_mode():
        embeds = spec["encode"](pipe, prompt, spec["negative"], cfg)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return embeds


def one_clip(pipe, spec, frames, args, embeds, out_path):
    """Generate one clip. Returns a level dict; never raises on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # Per-step timestamps by wrapping the scheduler: every pipeline calls
    # scheduler.step() once per denoise step, not every one offers a callback.
    step_times = []
    orig_step = pipe.scheduler.step

    def timed_step(*a, **kw):
        r = orig_step(*a, **kw)
        step_times.append(time.perf_counter())
        return r
    pipe.scheduler.step = timed_step

    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    level = dict(frames=frames, seconds=round(frames / spec["fps"], 2),
                 width=args.width, height=args.height, steps=args.steps)
    t0 = time.perf_counter()
    try:
        guidance = {} if spec.get("guider") else dict(guidance_scale=args.guidance)
        out = pipe(width=args.width, height=args.height, num_frames=frames,
                   num_inference_steps=args.steps, generator=gen,
                   **guidance, **embeds, **spec["kwargs"])
        t_end = time.perf_counter()
        video = out.frames[0]
        from diffusers.utils import export_to_video
        export_to_video(video, out_path, fps=spec["fps"])
        t_saved = time.perf_counter()
    except torch.OutOfMemoryError as e:
        level.update(status="oom", error=str(e).splitlines()[0][:200],
                     peak_vram_gb=gb(torch.cuda.max_memory_allocated()),
                     peak_reserved_gb=gb(torch.cuda.max_memory_reserved()))
        return level
    except Exception as e:  # noqa: BLE001 -- record and move on
        level.update(status="error", error=f"{type(e).__name__}: {str(e)[:200]}")
        return level
    finally:
        pipe.scheduler.step = orig_step

    # Steps 2..N are timed from callbacks; scale to N so the first step (which
    # shares its start with latent prep) does not skew the per-step figure.
    n = len(step_times)
    denoise = (step_times[-1] - step_times[0]) * n / (n - 1) if n > 1 else None
    decode = t_end - step_times[-1] if n else None
    wall = t_end - t0
    level.update(
        status="ok", wall_s=round(wall, 2),
        prep_s=round(wall - (denoise or 0) - (decode or 0), 2),
        denoise_s=round(denoise, 2) if denoise else None,
        s_per_step=round(denoise / n, 3) if denoise else None,
        decode_s=round(decode, 2) if decode else None,
        export_s=round(t_saved - t_end, 2),
        video_s_per_min=round(level["seconds"] / wall * 60, 2),
        frames_per_s=round(frames / wall, 2),
        peak_vram_gb=gb(torch.cuda.max_memory_allocated()),
        peak_reserved_gb=gb(torch.cuda.max_memory_reserved()),
        file=os.path.relpath(out_path),
    )
    return level


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--label", default=None, help="results.json key; default: the model name")
    ap.add_argument("--frames", default=None, help="comma list; default per model")
    ap.add_argument("--size", default=None, help="WxH; default per model")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offload", action="store_true",
                    help="enable_model_cpu_offload(): weights in host RAM, one component on the GPU "
                         "at a time. Needs more than this host's 30 GB of RAM for Wan2.2-5B.")
    ap.add_argument("--no-vae-tiling", action="store_true",
                    help="decode the whole clip in one VAE pass instead of spatial tiles")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--out", default=None, help="merge into this results.json")
    ap.add_argument("--out-dir", default="video/out")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--dim", action="append", metavar="KEY=VALUE",
                    help="tag the entry with a dimension, e.g. --dim steps=20")
    args = ap.parse_args()

    spec = MODELS[args.model]
    args.label = args.label or args.model
    args.steps = args.steps or spec["steps"]
    args.guidance = spec["guidance"] if args.guidance is None else args.guidance
    if args.size:
        w, _, h = args.size.partition("x")
        args.width, args.height = int(w), int(h)
    else:
        args.width, args.height = spec["width"], spec["height"]
    frames = [int(x) for x in args.frames.split(",")] if args.frames else spec["frames"]
    bad = [f for f in frames if (f - 1) % spec["frame_step"]]
    if bad:
        sys.exit(f"frames must be {spec['frame_step']}k+1 for {args.model}: {bad}")

    import diffusers
    print(f"model    {args.model}  ({spec['repo']})")
    print(f"gpu      {torch.cuda.get_device_name(0)}  torch {torch.__version__}  diffusers {diffusers.__version__}")
    print(f"shape    {args.width}x{args.height} @ {spec['fps']} fps, {args.steps} steps, guidance {args.guidance}"
          f"{', cpu offload' if args.offload else ''}")
    sys.stdout.flush()

    t0 = time.perf_counter()
    embeds = encode_once(spec, PROMPT, args.guidance > 1.0)
    encode_s = time.perf_counter() - t0
    print(f"encoded  {encode_s:.1f}s incl. loading the text encoder, now freed")
    sys.stdout.flush()

    t0 = time.perf_counter()
    pipe = spec["load"](spec["repo"], torch.bfloat16, {n: None for n in spec["encoders"]})
    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    if spec.get("guider"):
        pipe.guider = pipe.guider.__class__.from_config(pipe.guider.config, guidance_scale=args.guidance)
    vae_tiling = not args.no_vae_tiling and hasattr(pipe.vae, "enable_tiling")
    if vae_tiling:
        pipe.vae.enable_tiling()
    load_s = time.perf_counter() - t0
    print(f"loaded   {load_s:.1f}s, {gb(torch.cuda.memory_allocated())} GB on the GPU"
          f"{', VAE tiling' if vae_tiling else ''}")
    sys.stdout.flush()

    out_dir = os.path.join(args.out_dir, args.label.replace(" ", "_").replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    if not args.no_warmup:
        # A tiny clip so CUDA context, kernel selection and the first-call
        # overheads land here rather than in the first measured level.
        f = 1 + spec["frame_step"]
        wu = argparse.Namespace(**vars(args))
        wu.steps = 2
        one_clip(pipe, spec, f, wu, embeds, os.path.join(out_dir, "warmup.mp4"))

    print()
    print(f"{'frames':>7} {'secs':>6} {'wall':>8} {'prep':>6} {'denoise':>8} {'s/step':>7} {'decode':>7} "
          f"{'vid s/min':>9} {'peak GB':>8}  status")
    print("-" * 86)
    levels = []
    for f in frames:
        lv = one_clip(pipe, spec, f, args, embeds, os.path.join(out_dir, f"{f}f_{args.width}x{args.height}.mp4"))
        levels.append(lv)
        if lv["status"] == "ok":
            print(f"{f:>7} {lv['seconds']:>6.2f} {lv['wall_s']:>8.1f} {lv['prep_s']:>6.1f} "
                  f"{lv['denoise_s'] or 0:>8.1f} {lv['s_per_step'] or 0:>7.3f} {lv['decode_s'] or 0:>7.1f} "
                  f"{lv['video_s_per_min']:>9.2f} {lv['peak_vram_gb']:>8.2f}  ok")
        else:
            print(f"{f:>7} {lv['seconds']:>6.2f} {'':>8} {'':>6} {'':>8} {'':>7} {'':>7} {'':>9} "
                  f"{lv.get('peak_vram_gb', 0):>8.2f}  {lv['status']}: {lv.get('error', '')}")
        sys.stdout.flush()
        if lv["status"] == "oom":
            print("stopping: longer clips will not fit either")
            break

    entry = dict(
        label=args.label, model=args.model, repo=spec["repo"],
        generated=dt.date.today().isoformat(),
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, diffusers=diffusers.__version__,
        dtype="bf16", offload=args.offload, vae_tiling=vae_tiling, text_encoder_dropped=True,
        load_s=round(load_s, 1), encode_s=round(encode_s, 1),
        workload=dict(width=args.width, height=args.height, fps=spec["fps"], steps=args.steps,
                      guidance=args.guidance, seed=args.seed, prompt=PROMPT),
        levels=levels,
        dimensions=dict(model=args.model, res=f"{args.width}x{args.height}", steps=str(args.steps),
                        **parse_dims(args.dim)),
    )
    if args.notes:
        entry["notes"] = args.notes
    if args.out:
        merge_into(args.out, entry)
        print(f"\nmerged into {args.out} as '{args.label}'")
    else:
        print()
        print(json.dumps(entry, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
