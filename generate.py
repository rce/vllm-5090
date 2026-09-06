#!/usr/bin/env python3
"""Generate one clip from a prompt, with the same model setups video.py measures.

  ./generate-video ltx-2.5 "a cat on a bicycle"
  ./generate-video ltx-2.5 "a cat on a bicycle" --seconds 8 --seed 3 --out cat.mp4
  ./generate-video wan2.2-5b-turbo "a cat on a bicycle" --size 640x352

Same two-stage load as the sweep (encode the prompt, drop the text encoder,
then load the generator), so it fits the card the same way; the timings it
prints are comparable to docs/results.json. LTX-2.5 clips come with audio.
"""

import argparse
import datetime as dt
import os
import re
import sys
import time

import torch

import video


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(video.MODELS))
    ap.add_argument("prompt")
    ap.add_argument("--seconds", type=float, default=5.0, help="clip length; rounded to what the VAE allows")
    ap.add_argument("--frames", type=int, default=None, help="exact frame count instead of --seconds")
    ap.add_argument("--size", default=None, help="WxH; default per model")
    ap.add_argument("--fps", type=int, default=None, help="LTX-2.5 generates at this rate; others just play at it")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--negative", default=None, help="negative prompt; default per model")
    ap.add_argument("--seed", type=int, default=None, help="default: random")
    ap.add_argument("--no-vae-tiling", action="store_true")
    ap.add_argument("--out", default=None, help="output mp4; default video/gen/<time>_<model>_<prompt>.mp4")
    args = ap.parse_args()

    spec = video.MODELS[args.model]
    args.steps = args.steps or spec["steps"]
    args.fps = args.fps or spec["fps"]
    if "frame_rate" in spec["kwargs"]:
        spec["kwargs"]["frame_rate"] = float(args.fps)
    args.guidance = spec["guidance"] if args.guidance is None else args.guidance
    negative = spec["negative"] if args.negative is None else args.negative
    if args.size:
        w, _, h = args.size.partition("x")
        args.width, args.height = int(w), int(h)
    else:
        args.width, args.height = spec["width"], spec["height"]
    step = spec["frame_step"]
    frames = args.frames or step * max(1, round(args.seconds * args.fps / step)) + 1
    if (frames - 1) % step:
        sys.exit(f"frames must be {step}k+1 for {args.model}, not {frames}")
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    if not args.out:
        slug = re.sub(r"[^a-z0-9]+", "-", args.prompt.lower()).strip("-")[:48]
        args.out = os.path.join("video", "gen", f"{dt.datetime.now():%Y%m%d-%H%M%S}_{args.model}_{slug}.mp4")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"model    {args.model}  ({spec['repo']})")
    print(f"clip     {frames} frames = {frames / args.fps:.2f} s at {args.width}x{args.height} @ {args.fps} fps, "
          f"{args.steps} steps, guidance {args.guidance}, seed {seed}")
    sys.stdout.flush()

    t0 = time.perf_counter()
    embeds = video.encode_once(spec, args.prompt, args.guidance > 1.0, negative)
    print(f"encoded  {time.perf_counter() - t0:.1f}s")
    sys.stdout.flush()

    t0 = time.perf_counter()
    pipe = spec["load"](spec["repo"], torch.bfloat16, {n: None for n in spec["encoders"]})
    pipe.to("cuda")
    if spec.get("guider"):
        pipe.guider = pipe.guider.__class__.from_config(pipe.guider.config, guidance_scale=args.guidance)
    if spec.get("prepare"):
        spec["prepare"](pipe, embeds)
    if args.steps != spec["steps"]:
        spec["kwargs"].pop("sigmas", None)
    if not args.no_vae_tiling and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    print(f"loaded   {time.perf_counter() - t0:.1f}s, {video.gb(torch.cuda.memory_allocated())} GB on the GPU")
    sys.stdout.flush()

    args.seed = seed
    lv = video.one_clip(pipe, spec, frames, args, embeds, args.out)
    if lv["status"] != "ok":
        sys.exit(f"{lv['status']}: {lv.get('error', '')}")
    print(f"done     {lv['wall_s']:.1f}s wall: denoise {lv['denoise_s']:.1f}s ({lv['s_per_step']:.3f} s/step), "
          f"decode {lv['decode_s']:.1f}s, export {lv['export_s']:.1f}s, peak {lv['peak_vram_gb']} GB")
    print(f"wrote    {args.out}")


if __name__ == "__main__":
    main()
