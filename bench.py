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

--single is the other measurement on the results page: one request at a
time on an idle server, the number a single user sees, plus what the server
says about itself (KV pool and concurrency from /metrics, the window from
/v1/models, weights from the HF cache) and what it was started with (the
profile plus overrides). It writes a `runs` entry keyed by --id.

  ./bench.py --single --profile qwen3.6-35b-a3b --id qwen36-moe-default --out docs/results.json
  ./bench.py --single --profile qwen3.6-35b-a3b --set SPEC_DECODE=1 --id qwen36-moe-mtp ...

runs.sh does this for every configuration on the page, starting and
stopping each server in turn.
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))

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
            info["root"] = models[0].get("root")
    except Exception as e:
        raise SystemExit(f"bench.py: cannot reach {base_url}: {e}")
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/version")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            info["vllm_version"] = json.load(r).get("version")
    except Exception:
        pass
    return info


def merge_into(path, entry, key="sweeps", keyed_by="label", method=None):
    """Add or replace this entry in a results.json, keyed by label (or id)."""
    try:
        with open(path) as f:
            doc = json.load(f)
    except FileNotFoundError:
        doc = {"schema_version": 1}
    items = doc.setdefault(key, [])
    for i, existing in enumerate(items):
        if existing.get(keyed_by) == entry[keyed_by]:
            items[i] = entry
            break
    else:
        items.append(entry)
    if method:
        doc.setdefault("method", {}).update(method)
        doc["generated"] = entry.get("generated", doc.get("generated"))
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------- --single

# Long enough that 300 tokens is a cut, not a forced continuation: with
# ignore_eos a model pushed past its natural end writes repetitive text, and
# an MTP draft head predicts repetition too well, which inflates the
# speculative numbers. So --single lets the model stop on its own and asks
# for more than it will get to write.
SINGLE_PROMPT = "Write a detailed essay of about 600 words on the sea: its physical nature, its role in climate, and what it has meant to people."


def read_profile(name, sets):
    """The env the server was started with: Containerfile defaults, then the
    profile file, then KEY=VALUE overrides, the way run.sh layers them."""
    env = {}
    with open(os.path.join(HERE, "Containerfile")) as f:
        in_env = False
        for line in f:
            s = line.strip()
            if s.startswith("ENV "):
                in_env = True
                s = s[4:]
            if not in_env:
                continue
            m = re.match(r"([A-Z_]+)=(.*?)\s*\\?$", s)
            if m:
                env[m.group(1)] = m.group(2)
            if not line.rstrip().endswith("\\"):
                in_env = False
    if name:
        with open(os.path.join(HERE, "profiles", name + ".env")) as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#") and "=" in s:
                    k, v = s.split("=", 1)
                    env[k.strip()] = v.strip()
    for kv in sets:
        if "=" not in kv:
            raise SystemExit(f"bench.py: --set expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def metrics_text(base_url, timeout=10):
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return ""


def cache_info(base_url):
    """KV pool size and the concurrency vLLM derives from it, as the server
    reports them in the vllm:cache_config_info gauge's labels."""
    m = re.search(r"vllm:cache_config_info\{([^}]*)\}", metrics_text(base_url))
    if not m:
        return {}
    labels = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
    out = {}
    if labels.get("kv_cache_size_tokens", "").isdigit():
        out["kv_tokens"] = int(labels["kv_cache_size_tokens"])
    try:
        out["max_concurrency"] = round(float(labels["kv_cache_max_concurrency"]), 2)
    except (KeyError, ValueError):
        pass
    if labels.get("cache_dtype"):
        out["kv_cache_dtype"] = labels["cache_dtype"]
    return out


def spec_counters(base_url):
    text = metrics_text(base_url)
    vals = {}
    for name in ("spec_decode_num_accepted_tokens_total", "spec_decode_num_draft_tokens_total"):
        m = re.search(rf"vllm:{name}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)", text)
        if m:
            vals[name] = float(m.group(1))
    return vals if len(vals) == 2 else None


def weights_gb(checkpoint):
    """Size of the checkpoint's safetensors in the local HF cache, in GB."""
    pat = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                       "hub", "models--" + checkpoint.replace("/", "--"), "snapshots", "*", "**", "*.safetensors")
    files = glob.glob(pat, recursive=True)
    if not files:
        return None
    return round(sum(os.path.getsize(os.path.realpath(f)) for f in files) / 1e9, 1)


def guess_precision(checkpoint):
    c = checkpoint.upper()
    if "NVFP4" in c or "FP4" in c:
        return "NVFP4"
    if "FP8" in c:
        return "FP8 (official)" if c.startswith("QWEN/") or c.startswith("NVIDIA/") else "FP8"
    if "AWQ" in c:
        return "AWQ"
    return "BF16 (official)"


def single_request(base_url, model, max_tokens, timeout):
    """One request, its decode rate excluding the prefill wait."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": SINGLE_PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    out = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                try:
                    chunk = json.loads(line[5:])
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    out = chunk["usage"].get("completion_tokens", 0)
                if ttft is None and any((c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning")
                                        for c in chunk.get("choices") or []):
                    ttft = time.perf_counter() - t0
    except Exception as e:
        return Result(False, error=f"{type(e).__name__}: {e}")
    wall = time.perf_counter() - t0
    if not out or ttft is None or wall <= ttft:
        return Result(False, error="no usable stream")
    if out < max_tokens:
        return Result(False, error=f"model stopped after {out} tokens; the prompt should outlast max_tokens")
    return Result(True, ttft=ttft, wall=wall, output_tokens=out)


def cmd_single(args):
    info = server_info(args.base_url)
    model = args.model or info.get("served_model_name")
    env = read_profile(args.profile, args.set)
    checkpoint = args.checkpoint or env.get("MODEL") or info.get("root") or model
    print(f"server   {args.base_url}  vLLM {info.get('vllm_version', '?')}")
    print(f"model    {model}  ({checkpoint}, max_model_len {info.get('max_model_len', '?')})")
    print(f"workload one request at a time, {args.max_tokens} output tokens of a longer answer, "
          f"{args.samples} samples after a cold one")

    spec_on = env.get("SPEC_DECODE") == "1"
    before = spec_counters(args.base_url) if spec_on else None
    cold = single_request(args.base_url, model, args.max_tokens, args.timeout)
    if not cold.ok:
        raise SystemExit(f"bench.py: cold request failed: {cold.error}")
    rate = lambda r: round((r.output_tokens - 1) / (r.wall - r.ttft), 1)
    print(f"cold     {rate(cold)} tok/s (discarded)")
    samples = []
    for i in range(args.samples):
        r = single_request(args.base_url, model, args.max_tokens, args.timeout)
        if not r.ok:
            raise SystemExit(f"bench.py: sample {i + 1} failed: {r.error}")
        samples.append(rate(r))
        print(f"sample {i + 1} {samples[-1]} tok/s  (TTFT {r.ttft:.3f} s)")
    after = spec_counters(args.base_url) if spec_on else None

    entry = {
        "id": args.id,
        "status": "ok",
        "profile": args.profile,
        "overrides": " ".join(args.set),
        "model": model,
        "checkpoint": checkpoint,
        "shape": args.shape,
        "precision": args.precision or guess_precision(checkpoint),
        "weights_gb": args.weights_gb if args.weights_gb is not None else weights_gb(checkpoint),
        "weights_source": "loaded on the GPU, from vLLM's load report" if args.weights_gb is not None
                          else "checkpoint size on disk",
        "max_model_len": info.get("max_model_len"),
        "vision": env.get("LANGUAGE_MODEL_ONLY", "0") != "1",
        "cuda_graphs": env.get("ENFORCE_EAGER", "0") != "1",
        "spec_decode": spec_on,
        "decode_tok_s": round(statistics.median(samples), 1),
        "decode_cold_sample": rate(cold),
        "decode_samples": samples,
        "generated": time.strftime("%Y-%m-%d"),
        "vllm_version": info.get("vllm_version"),
        **cache_info(args.base_url),
    }
    if before and after and after["spec_decode_num_draft_tokens_total"] > before["spec_decode_num_draft_tokens_total"]:
        acc = after["spec_decode_num_accepted_tokens_total"] - before["spec_decode_num_accepted_tokens_total"]
        drafted = after["spec_decode_num_draft_tokens_total"] - before["spec_decode_num_draft_tokens_total"]
        entry["spec_acceptance"] = round(acc / drafted, 3)
    if args.notes:
        entry["notes"] = args.notes
    if not args.shape:
        del entry["shape"]
    print(f"\ndecode   {entry['decode_tok_s']} tok/s median · KV pool {entry.get('kv_tokens', '?')} tokens "
          f"({entry.get('max_concurrency', '?')}x at {entry['max_model_len']}) · weights {entry['weights_gb']} GB"
          + (f" · MTP acceptance {entry['spec_acceptance']}" if "spec_acceptance" in entry else ""))
    method = {
        "workload": f"single request, {args.max_tokens} output tokens cut from a longer answer (no ignore_eos, "
                    "so speculative decoding sees natural text), temperature 0, thinking disabled, idle server",
        "prompt": SINGLE_PROMPT,
        "warmup": "first request after startup discarded as cold; reported value is the median of the "
                  f"{args.samples} that follow",
        "kv_tokens": "kv_cache_size_tokens from the vllm:cache_config_info gauge on /metrics",
        "acceptance": "vllm:spec_decode_num_accepted_tokens_total / vllm:spec_decode_num_draft_tokens_total, over the samples",
    }
    if args.out:
        merge_into(args.out, entry, key="runs", keyed_by="id", method=method)
        print(f"merged into {args.out} as '{args.id}'")
    else:
        print()
        json.dump(entry, sys.stdout, indent=2)
        print()


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
    s = ap.add_argument_group("--single: one request at a time, a `runs` entry")
    s.add_argument("--single", action="store_true", help="measure single-stream decode and describe the server")
    s.add_argument("--id", help="runs entry id (required with --single)")
    s.add_argument("--profile", help="profile the server was started with (profiles/<name>.env)")
    s.add_argument("--set", action="append", metavar="KEY=VALUE", default=[],
                   help="override the server was started with, repeatable (as given to run.sh)")
    s.add_argument("--checkpoint", help="default: the profile's MODEL")
    s.add_argument("--shape", help="free text, e.g. 'MoE, 35B total / 3B active'")
    s.add_argument("--precision", help="default: guessed from the checkpoint name")
    s.add_argument("--weights-gb", type=float,
                   help="weights on the GPU (runs.sh reads it from the server log); default: checkpoint size on disk")
    s.add_argument("--max-tokens", type=int, default=300)
    s.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    if args.single:
        if not args.id:
            raise SystemExit("bench.py: --single needs --id")
        cmd_single(args)
        return

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
