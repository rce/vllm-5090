#!/usr/bin/env python3
"""Synthetic concurrency sweep against an OpenAI-compatible vLLM server.

Answers two questions for a given server configuration:

  1. Does it work at all at each concurrency level, or does it fall over?
  2. What does it cost -- time to first token, per-stream decode rate, and
     total system throughput -- as concurrency rises?

Deliberately dependency-free (stdlib only) so it runs anywhere, including
inside the vLLM container. Deterministic by construction: a fixed synthetic
prompt and `ignore_eos` so every request emits exactly --output-tokens tokens,
which makes tok/s comparable across runs rather than a function of how chatty
the model felt.

  ./bench.py --label "qwen3.6-35b-a3b"
  ./bench.py --concurrency 1,2,4,8,16,32 --output-tokens 512
  ./bench.py --out docs/results.json --label baseline    # merge into results
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# A fixed, boring paragraph. Repeated to length so the prompt is deterministic
# and the same every run -- prefix caching is disabled per request below so a
# shared prefix does not quietly turn this into a cache benchmark.
FILLER = (
    "The harbour wall was built from grey stone and it has stood against the "
    "water for two hundred winters without complaint. Fishing boats return in "
    "the late afternoon when the light goes flat and the gulls follow them in. "
)


def build_prompt(target_words: int, salt: int) -> str:
    """Deterministic prompt of roughly target_words words, unique per request."""
    words = FILLER.split()
    out = []
    while len(out) < target_words:
        out.extend(words)
    # The salt goes first so each request has a distinct prefix; identical
    # prefixes across concurrent requests would be served from prefix cache
    # and measure the cache instead of the model.
    return f"Passage {salt}. " + " ".join(out[:target_words]) + \
           "\n\nSummarise the passage above, then continue it in the same voice."


class Result:
    __slots__ = ("ok", "ttft", "wall", "prompt_tokens", "output_tokens", "error")

    def __init__(self, ok, ttft=None, wall=None, prompt_tokens=0, output_tokens=0, error=None):
        self.ok = ok
        self.ttft = ttft
        self.wall = wall
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.error = error


def one_request(base_url, model, prompt, max_tokens, timeout) -> Result:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.8,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Force exactly max_tokens of output so throughput is comparable.
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    prompt_tokens = output_tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return Result(False, error=f"HTTP {resp.status}")
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                    output_tokens = chunk["usage"].get("completion_tokens", 0)
                for ch in chunk.get("choices") or []:
                    delta = ch.get("delta") or {}
                    # Reasoning models emit into `reasoning` before `content`;
                    # either one counts as the stream having started.
                    if ttft is None and (delta.get("content") or delta.get("reasoning")):
                        ttft = time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:200]
        return Result(False, error=f"HTTP {e.code}: {detail}")
    except Exception as e:  # timeouts, connection resets, engine death
        return Result(False, error=f"{type(e).__name__}: {e}")

    wall = time.perf_counter() - t0
    if output_tokens == 0:
        return Result(False, error="no usage reported")
    return Result(True, ttft=ttft, wall=wall,
                  prompt_tokens=prompt_tokens, output_tokens=output_tokens)


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run_level(base_url, model, concurrency, rounds, prompt_words, output_tokens,
              timeout, counter):
    """Run `rounds` batches of `concurrency` simultaneous requests."""
    results, batch_walls = [], []
    for _ in range(rounds):
        prompts = [build_prompt(prompt_words, next(counter)) for _ in range(concurrency)]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            batch = list(pool.map(
                lambda p: one_request(base_url, model, p, output_tokens, timeout),
                prompts))
        batch_walls.append(time.perf_counter() - t0)
        results.extend(batch)
        if not all(r.ok for r in batch):
            break  # a broken level will not get better with more rounds

    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    entry = {
        "concurrency": concurrency,
        "requests": len(results),
        "succeeded": len(ok),
        "failed": len(failed),
        "status": "ok" if ok and not failed else ("partial" if ok else "failed"),
    }
    if failed:
        # One representative error is enough; they are almost always identical.
        entry["error"] = failed[0].error
    if ok:
        ttfts = [r.ttft for r in ok if r.ttft is not None]
        # Per-stream decode rate excludes the prefill wait.
        per_stream = [(r.output_tokens - 1) / (r.wall - r.ttft)
                      for r in ok if r.ttft is not None and r.wall > r.ttft]
        total_out = sum(r.output_tokens for r in ok)
        wall = sum(batch_walls)
        entry.update({
            "prompt_tokens": ok[0].prompt_tokens,
            "output_tokens": ok[0].output_tokens,
            "ttft_p50_s": round(pct(ttfts, 50), 4) if ttfts else None,
            "ttft_p95_s": round(pct(ttfts, 95), 4) if ttfts else None,
            "decode_tok_s_per_stream": round(statistics.median(per_stream), 1) if per_stream else None,
            "total_tok_s": round(total_out / wall, 1) if wall else None,
            "latency_p50_s": round(pct([r.wall for r in ok], 50), 3),
            "latency_p95_s": round(pct([r.wall for r in ok], 95), 3),
        })
    return entry


def server_info(base_url, timeout=10):
    info = {}
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/v1/models")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            models = json.load(r).get("data", [])
        if models:
            info["served_model_name"] = models[0].get("id")
            info["max_model_len"] = models[0].get("max_model_len")
    except Exception as e:
        raise SystemExit(f"bench.py: cannot reach {base_url}: {e}")
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/version")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            info["vllm_version"] = json.load(r).get("version")
    except Exception:
        pass
    return info


def merge_into(path, sweep):
    """Add or replace this sweep in a results.json, keyed by label."""
    try:
        with open(path) as f:
            doc = json.load(f)
    except FileNotFoundError:
        doc = {"schema_version": 1}
    sweeps = doc.setdefault("sweeps", [])
    for i, existing in enumerate(sweeps):
        if existing.get("label") == sweep["label"]:
            sweeps[i] = sweep
            break
    else:
        sweeps.append(sweep)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", help="defaults to whatever /v1/models reports")
    ap.add_argument("--label", help="name for this configuration in the output")
    ap.add_argument("--concurrency", default="1,2,4,8,16",
                    help="comma-separated levels (default: 1,2,4,8,16)")
    ap.add_argument("--rounds", type=int, default=2,
                    help="batches per level (default: 2)")
    ap.add_argument("--prompt-words", type=int, default=700,
                    help="synthetic prompt length in words (default: 700, ~1K tokens)")
    ap.add_argument("--output-tokens", type=int, default=256,
                    help="tokens generated per request, exactly (default: 256)")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", help="merge the sweep into this JSON file")
    ap.add_argument("--notes", help="free-text note stored with the sweep")
    ap.add_argument("--dim", action="append", metavar="KEY=VALUE", default=[],
                    help="dimension tag for filtering, repeatable "
                         "(e.g. --dim seqs=32 --dim vision=off)")
    args = ap.parse_args()

    dimensions = {}
    for d in args.dim:
        if "=" not in d:
            raise SystemExit(f"bench.py: --dim expects KEY=VALUE, got {d!r}")
        k, v = d.split("=", 1)
        dimensions[k.strip()] = v.strip()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    info = server_info(args.base_url)
    model = args.model or info.get("served_model_name")
    if not model:
        raise SystemExit("bench.py: no model given and /v1/models returned none")
    label = args.label or model

    print(f"server   {args.base_url}  vLLM {info.get('vllm_version', '?')}")
    print(f"model    {model}  (max_model_len {info.get('max_model_len', '?')})")
    print(f"workload {args.prompt_words} prompt words -> exactly "
          f"{args.output_tokens} output tokens, {args.rounds} round(s) per level")
    print()

    counter = iter(range(1, 10_000_000))
    print("warmup...", end=" ", flush=True)
    w = one_request(args.base_url, model, build_prompt(args.prompt_words, next(counter)),
                    args.output_tokens, args.timeout)
    print("ok" if w.ok else f"FAILED ({w.error})")
    if not w.ok:
        raise SystemExit("bench.py: warmup failed, not benchmarking a broken server")
    print()

    hdr = f"{'conc':>5}  {'ok':>7}  {'TTFT p50':>9}  {'TTFT p95':>9}  " \
          f"{'tok/s/req':>10}  {'total tok/s':>12}  {'lat p95':>8}"
    print(hdr)
    print("-" * len(hdr))

    entries = []
    for c in levels:
        e = run_level(args.base_url, model, c, args.rounds, args.prompt_words,
                      args.output_tokens, args.timeout, counter)
        entries.append(e)
        okstr = f"{e['succeeded']}/{e['requests']}"
        if e["status"] == "ok":
            print(f"{c:>5}  {okstr:>7}  {e['ttft_p50_s']:>9.3f}  {e['ttft_p95_s']:>9.3f}  "
                  f"{e['decode_tok_s_per_stream']:>10.1f}  {e['total_tok_s']:>12.1f}  "
                  f"{e['latency_p95_s']:>8.2f}")
        else:
            print(f"{c:>5}  {okstr:>7}  {'FAILED':>9}  {e.get('error', '')[:60]}")
            if e["status"] == "failed":
                print(f"{'':>5}  (stopping: concurrency {c} does not work at all)")
                break

    sweep = {
        "label": label,
        "model": model,
        "base_url": args.base_url,
        "generated": time.strftime("%Y-%m-%d"),
        "vllm_version": info.get("vllm_version"),
        "max_model_len": info.get("max_model_len"),
        "workload": {
            "prompt_words": args.prompt_words,
            "output_tokens": args.output_tokens,
            "rounds": args.rounds,
            "ignore_eos": True,
            "thinking": False,
        },
        "levels": entries,
    }
    # Dimensions drive the filters on the results page. Always record the three
    # the sweep knows about itself, then layer the caller's tags on top.
    sweep["dimensions"] = {
        "model": model,
        "prompt": f"{entries[0]['prompt_tokens']} tok" if entries and entries[0].get("prompt_tokens")
                  else f"~{args.prompt_words} words",
        "output": f"{args.output_tokens} tok",
        **dimensions,
    }
    if args.notes:
        sweep["notes"] = args.notes

    if args.out:
        merge_into(args.out, sweep)
        print(f"\nmerged into {args.out} as '{label}'")
    else:
        print()
        json.dump(sweep, sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
