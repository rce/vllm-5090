#!/usr/bin/env python3
"""Generate one clip from a prompt, with the same model setups video.py measures.

  ./generate-video ltx-2.5 "a cat on a bicycle"
  ./generate-video ltx-2.5 "a cat on a bicycle" --seconds 8 --seed 3 --out cat.mp4
  ./generate-video wan2.2-5b-turbo "a cat on a bicycle" --size 640x352

Same two-stage load as the sweep (encode the prompt, drop the text encoder,
then load the generator), so it fits the card the same way; the timings it
prints are comparable to docs/results.json. LTX-2.5 clips come with audio.

Loading is ~25 s per clip on LTX. To pay it once, keep a server running:

  ./generate-video serve ltx-2.5          # holds the generator on the GPU

and the commands above then go to it. Prompt embeddings are cached on disk
(video/cache/), so a prompt already seen costs only the clip itself. A new
prompt still needs the text encoder, which does not fit beside the
generator: the generator is parked in host RAM while the encoder runs, if
there is room for it, and reloaded from disk otherwise.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time

import video

PORT = 8765
CACHE_DIR = os.path.join("video", "cache")


# --- the request -------------------------------------------------------------

def parse_request(args):
    """Turn parsed CLI args into one plain dict describing a clip; no torch here."""
    spec = video.MODELS[args.model]
    fps = args.fps or spec["fps"]
    steps = args.steps or spec["steps"]
    if args.size:
        w, _, h = args.size.partition("x")
        width, height = int(w), int(h)
    else:
        width, height = spec["width"], spec["height"]
    step = spec["frame_step"]
    frames = args.frames or step * max(1, round(args.seconds * fps / step)) + 1
    if (frames - 1) % step:
        sys.exit(f"frames must be {step}k+1 for {args.model}, not {frames}")
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    out = args.out
    if not out:
        slug = re.sub(r"[^a-z0-9]+", "-", args.prompt.lower()).strip("-")[:48]
        out = os.path.join("video", "gen", f"{dt.datetime.now():%Y%m%d-%H%M%S}_{args.model}_{slug}.mp4")
    return dict(model=args.model, prompt=args.prompt, negative=args.negative, frames=frames, fps=fps,
                width=width, height=height, steps=steps,
                guidance=spec["guidance"] if args.guidance is None else args.guidance,
                seed=seed, vae_tiling=not args.no_vae_tiling, out=out)


def describe(req):
    return (f"clip     {req['frames']} frames = {req['frames'] / req['fps']:.2f} s at "
            f"{req['width']}x{req['height']} @ {req['fps']} fps, {req['steps']} steps, "
            f"guidance {req['guidance']}, seed {req['seed']}")


# --- the model side (container only) ------------------------------------------

class Session:
    """One model, kept ready: the generator on the GPU and the prompts it has seen."""

    def __init__(self, model, log=print):
        self.model, self.spec, self.log = model, video.MODELS[model], log
        self.pipe = None
        self.embeds = {}  # cache key -> embeds on the GPU

    # -- prompt embeddings ---------------------------------------------------
    def cache_key(self, req):
        cfg = req["guidance"] > 1.0
        negative = self.spec["negative"] if req["negative"] is None else req["negative"]
        raw = json.dumps([self.model, req["prompt"], negative, cfg])
        return hashlib.sha1(raw.encode()).hexdigest(), negative, cfg

    def embeds_for(self, req):
        import torch
        key, negative, cfg = self.cache_key(req)
        if key in self.embeds:
            return self.embeds[key]
        path = os.path.join(CACHE_DIR, self.model, key + ".pt")
        if os.path.exists(path):
            embeds = torch.load(path, map_location="cuda")
            self.log(f"encoded  cached ({path})")
        else:
            t0 = time.perf_counter()
            parked = self.park()
            try:
                embeds = video.encode_once(self.spec, req["prompt"], cfg, negative)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(to_cpu(embeds), path)
                self.log(f"encoded  {time.perf_counter() - t0:.1f}s")
            finally:
                self.unpark(parked)
        self.embeds[key] = embeds
        return embeds

    # -- the generator -------------------------------------------------------
    def load(self):
        import torch
        spec = self.spec
        t0 = time.perf_counter()
        self.pipe = spec["load"](spec["repo"], torch.bfloat16, {n: None for n in spec["encoders"]})
        self.pipe.to("cuda")
        self.log(f"loaded   {time.perf_counter() - t0:.1f}s, {video.gb(torch.cuda.memory_allocated())} GB on the GPU")

    def park(self):
        """Make room for the text encoder. Returns how to get the generator back."""
        import gc
        import torch
        if self.pipe is None:
            return None
        # The encoder streams from disk through mmap, which is reclaimable, so
        # the parked weights are the only real claim on host RAM. No swap here.
        need = torch.cuda.memory_allocated()
        if mem_available() > need + 3 * 2**30:
            t0 = time.perf_counter()
            self.pipe.to("cpu")
            torch.cuda.empty_cache()
            self.log(f"parked   generator in host RAM, {time.perf_counter() - t0:.1f}s")
            return "cpu"
        self.log("dropped  generator: not enough host RAM to park it, will reload")
        self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        return "reload"

    def unpark(self, how):
        if how == "cpu":
            import ctypes
            t0 = time.perf_counter()
            self.pipe.to("cuda")
            ctypes.CDLL("libc.so.6").malloc_trim(0)  # hand the parked 20 GB back to the host
            self.log(f"restored generator to the GPU, {time.perf_counter() - t0:.1f}s")
        elif how == "reload" or self.pipe is None:
            self.load()

    # -- one clip ------------------------------------------------------------
    def generate(self, req):
        self.log(describe(req))
        embeds = dict(self.embeds_for(req))  # a copy: prepare() consumes entries
        if self.pipe is None:
            self.load()
        pipe, spec = self.pipe, dict(self.spec, kwargs=dict(self.spec["kwargs"]))
        if spec.get("prepare"):  # per clip: on LTX this points pipe.connectors at this prompt
            spec["prepare"](pipe, embeds)
        if "frame_rate" in spec["kwargs"]:
            spec["kwargs"]["frame_rate"] = float(req["fps"])
        if req["steps"] != self.spec["steps"]:
            spec["kwargs"].pop("sigmas", None)
        if spec.get("guider"):
            pipe.guider = pipe.guider.__class__.from_config(pipe.guider.config, guidance_scale=req["guidance"])
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling() if req["vae_tiling"] else pipe.vae.disable_tiling()
        os.makedirs(os.path.dirname(req["out"]) or ".", exist_ok=True)
        args = argparse.Namespace(**req)
        lv = video.one_clip(pipe, spec, req["frames"], args, embeds, req["out"])
        if lv["status"] != "ok":
            self.log(f"{lv['status']}: {lv.get('error', '')}")
            return lv
        self.log(f"done     {lv['wall_s']:.1f}s wall: denoise {lv['denoise_s']:.1f}s ({lv['s_per_step']:.3f} s/step), "
                 f"decode {lv['decode_s']:.1f}s, export {lv['export_s']:.1f}s, peak {lv['peak_vram_gb']} GB")
        self.log(f"wrote    {req['out']}")
        return lv


def to_cpu(x):
    """Tensors to host memory, through the dicts/tuples the encoders return them in."""
    if hasattr(x, "cpu"):
        return x.cpu()
    if isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_cpu(v) for v in x)
    return x


def mem_available():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


# --- server / client ---------------------------------------------------------

def serve(model):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    session = Session(model)
    session.load()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the console for the session's own lines
            pass

        def do_GET(self):
            body = json.dumps(dict(model=model)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Connection", "close")
            self.end_headers()

            def log(line):
                print(line)
                sys.stdout.flush()
                try:
                    self.wfile.write((line + "\n").encode())
                    self.wfile.flush()
                except OSError:  # client gone; the clip still gets made
                    pass
            session.log = log
            try:
                if req["model"] != model:
                    log(f"error: this server holds {model}; restart it with 'serve {req['model']}'")
                    return
                session.generate(req)
            except Exception as e:  # noqa: BLE001 -- report to the client, keep serving
                log(f"error: {type(e).__name__}: {e}")
            finally:
                session.log = print

    print(f"serving  {model} on port {PORT}; ./generate-video {model} \"<prompt>\" now goes here")
    sys.stdout.flush()
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


def server_model():
    """The model a running server holds, or None. Stdlib only: this runs on the host."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=1) as r:
            return json.load(r)["model"]
    except Exception:  # noqa: BLE001
        return None


def send(req):
    import urllib.request
    data = json.dumps(req).encode()
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/", data=data,
                                                      headers={"Content-Type": "application/json"}))
    for line in r:
        sys.stdout.write(line.decode())
        sys.stdout.flush()


# --- entry -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("--serve", action="store_true", help="keep the generator loaded and take requests")
    ap.add_argument("--client", action="store_true", help=argparse.SUPPRESS)  # host side of --serve
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

    if args.model not in video.MODELS:
        sys.exit(f"unknown model {args.model!r}; one of: {', '.join(sorted(video.MODELS))}")
    if args.serve:
        serve(args.model)
        return
    if not args.prompt:
        sys.exit("a prompt is required")
    req = parse_request(args)
    if args.client:
        send(req)
        return

    if video.torch is None:
        sys.exit("no torch here: run this through ./generate-video, or start a server with 'serve'")
    session = Session(args.model)
    print(f"model    {args.model}")
    session.generate(req)


if __name__ == "__main__":
    main()
