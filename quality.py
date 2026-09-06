#!/usr/bin/env python3
"""Divergence probe: did a server configuration change the model, or only its speed?

Sibling of bench.py. Where bench.py measures how fast a configuration is, this
measures whether two configurations of the *same model* produce the same
next-token distributions. It needs no datasets, no judge model and no second
copy of the weights -- only the running server's logprobs.

Two steps:

  capture   Run the prompt set against the running server and store, per token,
            the model's top-k next-token probabilities. Two records per prompt:
              gen    what the server generated greedily, with the distribution
                     at every generated position (the decode path);
              score  the same fixed continuation fed back in as prompt text,
                     with the distribution at every position (the prefill path,
                     via vLLM's `prompt_logprobs`).
            With --ref, the continuation scored is taken from an earlier
            capture, so every capture in a family scores the same text.

  compare   Take two captures and report how far apart they are: KL divergence
            per position, top-1 agreement, how the probability of the reference
            token shifted, perplexity, and on the generation side how often the
            greedy output was token-for-token identical and where it first split.

  ./quality.py capture --label qwen36-baseline
  ./quality.py capture --label qwen36-fp8kv --ref quality/captures/qwen36-baseline.json.gz
  ./quality.py compare quality/captures/qwen36-baseline.json.gz \\
                       quality/captures/qwen36-fp8kv.json.gz --out docs/results.json

  jitter    Send the same scoring request repeatedly to one server and report
            how much its own answers move. Run this first: it is the floor
            below which a compare result means nothing.

docs/divergence.md explains the metrics.

Memory note: `prompt_logprobs` materialises full-vocabulary logits for every
prompt position in a prefill chunk (150k vocab x fp32 = 0.6 MB per token), on
top of the KV budget vLLM profiled. A server with little headroom -- the FP8
27B checkpoint leaves ~0.3 GiB -- will OOM mid-capture. Start it with
`--max-num-batched-tokens 256` and capture with `--concurrency 1`.

Speculative-decoding note: with MTP enabled, vLLM 0.28.0 answers
`prompt_logprobs` from the draft head for short prompts (row i predicts token
i+1) and with a partially overwritten buffer for others. The teacher-forced
numbers against such a server are not a model measurement; the generation
side is unaffected. Record those comparisons with `--suspect`.
"""

import argparse
import gzip
import hashlib
import json
import math
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

EPS = 1e-10
SCHEMA = 1


# --------------------------------------------------------------------------- HTTP

def post(base_url, path, body, timeout):
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"HTTP {e.code} on {path}: {detail}") from None


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
        raise SystemExit(f"quality.py: cannot reach {base_url}: {e}")
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/version")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            info["vllm_version"] = json.load(r).get("version")
    except Exception:
        pass
    return info


def tokenize(base_url, model, messages, thinking, timeout):
    """Token ids of the chat-templated prompt, exactly as the server would see it."""
    body = {
        "model": model,
        "messages": messages,
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    return post(base_url, "/tokenize", body, timeout)["tokens"]


def parse_token_key(key):
    # With return_tokens_as_token_ids the server keys logprobs by "token_id:N"
    # instead of decoded text, which is lossy (two ids can decode identically).
    if not key.startswith("token_id:"):
        raise RuntimeError(f"unexpected logprob key {key!r}; server too old?")
    return int(key[9:])


def generate(base_url, model, prompt_ids, max_tokens, top_k, timeout):
    """Greedy continuation with the top-k distribution at every generated step."""
    body = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
        "logprobs": top_k,
        "return_tokens_as_token_ids": True,
        "return_token_ids": True,
    }
    resp = post(base_url, "/v1/completions", body, timeout)
    choice = resp["choices"][0]
    ids = choice["token_ids"]
    top = []
    for step in choice["logprobs"]["top_logprobs"]:
        row = sorted(((parse_token_key(k), lp) for k, lp in step.items()),
                     key=lambda t: -t[1])
        top.append([[i, round(lp, 5)] for i, lp in row])
    if len(top) != len(ids):
        raise RuntimeError(f"logprob rows {len(top)} != tokens {len(ids)}")
    return {"ids": ids, "finish": choice.get("finish_reason"), "top": top}


def score(base_url, model, prompt_ids, cont_ids, top_k, timeout):
    """Teacher-forced pass: distribution at every continuation position.

    The whole text goes in as the prompt; vLLM's prompt_logprobs returns, for
    each position, the actual token's logprob plus the top-k alternatives. A
    request with prompt_logprobs skips the prefix cache entirely, so this is
    always a full recompute of the prompt on the prefill path.
    """
    body = {
        "model": model,
        "prompt": prompt_ids + cont_ids,
        "max_tokens": 0,
        "echo": True,
        "temperature": 0,
        "prompt_logprobs": top_k,
    }
    resp = post(base_url, "/v1/completions", body, timeout)
    plp = resp["choices"][0].get("prompt_logprobs")
    if plp is None:
        raise RuntimeError("server returned no prompt_logprobs")
    p = len(prompt_ids)
    if len(plp) < p + len(cont_ids):
        raise RuntimeError(f"prompt_logprobs has {len(plp)} rows, need {p + len(cont_ids)}")
    top = []
    for j, tok in enumerate(cont_ids):
        entry = plp[p + j]
        row = sorted(((int(k), v["logprob"]) for k, v in entry.items()),
                     key=lambda t: -t[1])
        if tok not in {i for i, _ in row}:
            raise RuntimeError(f"position {p + j}: actual token {tok} missing from prompt_logprobs")
        top.append([[i, round(lp, 5)] for i, lp in row])
    return {"ids": cont_ids, "top": top}


# ------------------------------------------------------------------------ capture

def load_prompts(path, limit=None, cats=None):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            if cats and p.get("cat") not in cats:
                continue
            out.append(p)
    if limit:
        out = out[:limit]
    return out


def messages_of(p):
    msgs = []
    if p.get("system"):
        msgs.append({"role": "system", "content": p["system"]})
    msgs.append({"role": "user", "content": p["user"]})
    return msgs


def load_capture(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("schema_version") != SCHEMA:
        raise SystemExit(f"quality.py: {path}: unsupported schema {doc.get('schema_version')}")
    return doc


def save_capture(path, doc):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))


def capture(args):
    info = server_info(args.base_url)
    model = args.model or info.get("served_model_name")
    if not model:
        raise SystemExit("quality.py: no model given and /v1/models returned none")
    label = args.label or model
    prompts = load_prompts(args.prompts, args.limit,
                           set(args.cats.split(",")) if args.cats else None)

    ref = None
    if args.ref:
        ref = load_capture(args.ref)
        ref_by_id = {p["id"]: p for p in ref["prompts"]}
        missing = [p["id"] for p in prompts if p["id"] not in ref_by_id]
        if missing:
            raise SystemExit(f"quality.py: --ref lacks {len(missing)} prompt(s): {missing[:5]}")
        if ref["settings"]["thinking"] != args.thinking:
            raise SystemExit("quality.py: --ref was captured with a different thinking "
                             "setting; the prompts would not match")

    print(f"server   {args.base_url}  vLLM {info.get('vllm_version', '?')}")
    print(f"model    {model}  (max_model_len {info.get('max_model_len', '?')})")
    print(f"prompts  {len(prompts)} from {args.prompts}"
          + (f", scoring continuations from {args.ref}" if ref else ""))
    print(f"settings greedy, {args.max_tokens} tokens, top-{args.top_k}, "
          f"thinking={'on' if args.thinking else 'off'}, concurrency {args.concurrency}")
    print()

    t_start = time.perf_counter()
    done = {"n": 0}

    def one(p):
        pid = tokenize(args.base_url, model, messages_of(p), args.thinking, args.timeout)
        sha = hashlib.sha1(json.dumps(pid).encode()).hexdigest()[:16]
        if ref is not None:
            r = ref_by_id[p["id"]]
            if r["prompt_sha"] != sha:
                raise RuntimeError(f"{p['id']}: prompt tokens differ from --ref "
                                   f"(different tokenizer or chat template?)")
        gen = generate(args.base_url, model, pid, args.max_tokens, args.top_k, args.timeout)
        cont = ref_by_id[p["id"]]["score"]["ids"] if ref is not None else gen["ids"]
        sc = score(args.base_url, model, pid, cont, args.top_k, args.timeout)
        done["n"] += 1
        print(f"\r  {done['n']}/{len(prompts)}  {time.perf_counter() - t_start:6.0f}s", end="",
              flush=True)
        return {
            "id": p["id"], "cat": p.get("cat", "?"),
            "prompt_tokens": len(pid), "prompt_sha": sha,
            "gen": gen, "score": sc,
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one, prompts))
    elapsed = time.perf_counter() - t_start
    print(f"\r  {len(results)} prompts in {elapsed:.0f}s"
          f"  ({sum(len(r['gen']['ids']) for r in results)} generated tokens,"
          f" {sum(len(r['score']['ids']) for r in results)} scored)")

    doc = {
        "schema_version": SCHEMA,
        "label": label,
        "model": model,
        "base_url": args.base_url,
        "generated": time.strftime("%Y-%m-%d"),
        "vllm_version": info.get("vllm_version"),
        "max_model_len": info.get("max_model_len"),
        "prompt_set": args.prompts,
        "ref_label": ref["label"] if ref else None,
        "settings": {
            "max_tokens": args.max_tokens, "top_k": args.top_k,
            "thinking": args.thinking, "temperature": 0, "seed": 0,
            "concurrency": args.concurrency,
        },
        "elapsed_s": round(elapsed, 1),
        "notes": args.notes,
        "prompts": results,
    }
    out = args.out or f"quality/captures/{label}.json.gz"
    save_capture(out, doc)
    print(f"wrote {out}")


# ------------------------------------------------------------------------ compare

def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def kl_topk(rows_a, rows_b):
    """KL(A || B) between two truncated distributions, as a lower bound.

    Each side knows its own top-k (plus the actual token). Tokens known to both
    are compared directly; everything else is lumped into one 'other' bucket on
    both sides. Merging bins can only lower KL, so the result never overstates
    the true divergence, and the covered mass says how tight it is.
    """
    a = dict(rows_a)
    b = dict(rows_b)
    shared = a.keys() & b.keys()
    kl = 0.0
    mass_a = mass_b = 0.0
    for t in shared:
        pa = math.exp(a[t])
        mass_a += pa
        mass_b += math.exp(b[t])
        kl += pa * (a[t] - b[t])
    # Mass outside the shared set. Float rounding in the server's log-softmax
    # can push a sum a hair over 1, so floor 'other' at what is provably
    # there: the listed-but-unshared tokens.
    only_a = sum(math.exp(a[t]) for t in a.keys() - shared)
    only_b = sum(math.exp(b[t]) for t in b.keys() - shared)
    oa = max(1.0 - mass_a, only_a, EPS)
    ob = max(1.0 - mass_b, only_b, EPS)
    kl += oa * math.log(oa / ob)
    return max(kl, 0.0), mass_a


def argmax(rows):
    return max(rows, key=lambda r: r[1])[0]


def logp_of(rows, tok):
    for t, lp in rows:
        if t == tok:
            return lp
    return None


def summarize(kls, dps, agree, lpa, lpb, mass):
    n = len(kls)
    if n == 0:
        return {"positions": 0}
    out = {
        "positions": n,
        "kld_mean": round(statistics.fmean(kls), 6),
        "kld_p50": round(pct(kls, 50), 6),
        "kld_p90": round(pct(kls, 90), 6),
        "kld_p99": round(pct(kls, 99), 6),
        "kld_p999": round(pct(kls, 99.9), 6),
        "kld_max": round(max(kls), 6),
        "top1_agree": round(agree / n, 5),
        "dp_mean_abs": round(statistics.fmean(abs(d) for d in dps), 6),
        "dp_p1": round(pct(dps, 1), 5),
        "dp_p5": round(pct(dps, 5), 5),
        "dp_p50": round(pct(dps, 50), 5),
        "dp_p95": round(pct(dps, 95), 5),
        "dp_p99": round(pct(dps, 99), 5),
        "support_mass_mean": round(statistics.fmean(mass), 5),
    }
    if lpa:
        out["ppl_a"] = round(math.exp(-statistics.fmean(lpa)), 4)
        out["ppl_b"] = round(math.exp(-statistics.fmean(lpb)), 4)
    return out


def compare_teacher_forced(A, B):
    """Positions where both captures scored the same fixed text."""
    b_by_id = {p["id"]: p for p in B["prompts"]}
    kls, dps, lpa, lpb, mass = [], [], [], [], []
    agree = 0
    per_prompt = []     # mean KLD per prompt, for the bootstrap
    per_cat = {}
    n_prompts = 0
    for pa in A["prompts"]:
        pb = b_by_id.get(pa["id"])
        if pb is None:
            continue
        if pa["prompt_sha"] != pb["prompt_sha"]:
            raise SystemExit(f"quality.py: {pa['id']}: prompts differ between captures "
                             f"(different tokenizer, template or thinking setting)")
        if pa["score"]["ids"] != pb["score"]["ids"]:
            raise SystemExit(f"quality.py: {pa['id']}: captures scored different text. "
                             f"Capture B with --ref A so both score the same continuation.")
        n_prompts += 1
        cat = per_cat.setdefault(pa["cat"], {"kls": [], "agree": 0, "dps": []})
        this = []
        for tok, ra, rb in zip(pa["score"]["ids"], pa["score"]["top"], pb["score"]["top"]):
            kl, m = kl_topk(ra, rb)
            la, lb = logp_of(ra, tok), logp_of(rb, tok)
            dp = math.exp(lb) - math.exp(la)
            same = argmax(ra) == argmax(rb)
            kls.append(kl); dps.append(dp); lpa.append(la); lpb.append(lb); mass.append(m)
            agree += same
            this.append(kl)
            cat["kls"].append(kl); cat["dps"].append(dp); cat["agree"] += same
        if this:
            per_prompt.append(statistics.fmean(this))
    out = summarize(kls, dps, agree, lpa, lpb, mass)
    out["prompts"] = n_prompts
    if len(per_prompt) >= 2:
        rng = random.Random(0)
        means = []
        for _ in range(2000):
            sample = [per_prompt[rng.randrange(len(per_prompt))] for _ in per_prompt]
            means.append(statistics.fmean(sample))
        out["kld_mean_ci95"] = [round(pct(means, 2.5), 6), round(pct(means, 97.5), 6)]
    out["per_category"] = {
        c: {
            "positions": len(v["kls"]),
            "kld_mean": round(statistics.fmean(v["kls"]), 6),
            "top1_agree": round(v["agree"] / len(v["kls"]), 5),
            "dp_mean_abs": round(statistics.fmean(abs(d) for d in v["dps"]), 6),
        } for c, v in sorted(per_cat.items()) if v["kls"]
    }
    return out


def compare_generation(A, B):
    """Greedy outputs from each server, compared up to where they first split."""
    b_by_id = {p["id"]: p for p in B["prompts"]}
    kls, dps, mass, lpa, lpb = [], [], [], [], []
    agree = 0
    identical = 0
    first_div = []
    n = 0
    for pa in A["prompts"]:
        pb = b_by_id.get(pa["id"])
        if pb is None:
            continue
        n += 1
        ga, gb = pa["gen"]["ids"], pb["gen"]["ids"]
        common = min(len(ga), len(gb))
        d = next((i for i in range(common) if ga[i] != gb[i]), None)
        if d is None:
            if len(ga) == len(gb):
                identical += 1
            else:
                d = common
        if d is not None:
            first_div.append(d)
        upto = common if d is None else d
        for i in range(upto):
            ra, rb = pa["gen"]["top"][i], pb["gen"]["top"][i]
            tok = ga[i]
            kl, m = kl_topk(ra, rb)
            la, lb = logp_of(ra, tok), logp_of(rb, tok)
            if la is None or lb is None:
                continue
            kls.append(kl); mass.append(m); lpa.append(la); lpb.append(lb)
            dps.append(math.exp(lb) - math.exp(la))
            agree += argmax(ra) == argmax(rb)
    out = summarize(kls, dps, agree, lpa, lpb, mass)
    out.update({
        "prompts": n,
        "identical": identical,
        "identical_rate": round(identical / n, 4) if n else None,
        "diverged": len(first_div),
        "first_divergence_p10": pct(first_div, 10),
        "first_divergence_p50": pct(first_div, 50),
        "first_divergence_min": min(first_div) if first_div else None,
        "gen_tokens_a": sum(len(p["gen"]["ids"]) for p in A["prompts"]),
        "gen_tokens_b": sum(len(p["gen"]["ids"]) for p in B["prompts"]),
    })
    return out


def merge_into(path, entry, key="divergence"):
    """Add or replace an entry in a results.json list, keyed by label."""
    try:
        with open(path) as f:
            doc = json.load(f)
    except FileNotFoundError:
        doc = {"schema_version": 1}
    items = doc.setdefault(key, [])
    for i, existing in enumerate(items):
        if existing.get("label") == entry["label"]:
            items[i] = entry
            break
    else:
        items.append(entry)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")


def fmt(v, spec):
    return format(v, spec) if isinstance(v, (int, float)) else "-"


def compare(args):
    A = load_capture(args.a)
    B = load_capture(args.b)
    if A["model"] != B["model"]:
        print(f"warning: comparing different served names ({A['model']} vs {B['model']}); "
              f"only meaningful if it is the same model", file=sys.stderr)
    if A["settings"]["top_k"] != B["settings"]["top_k"]:
        print("warning: captures used different top-k; the KLD bound is only as tight "
              "as the smaller one", file=sys.stderr)

    dimensions = {}
    for d in args.dim:
        if "=" not in d:
            raise SystemExit(f"quality.py: --dim expects KEY=VALUE, got {d!r}")
        k, v = d.split("=", 1)
        dimensions[k.strip()] = v.strip()

    tf = compare_teacher_forced(A, B)
    gen = compare_generation(A, B)
    label = args.label or f"{A['label']} vs {B['label']}"

    print(f"{label}")
    print(f"  A: {A['label']}  ({A['model']}, vLLM {A.get('vllm_version')}, {A['generated']})")
    print(f"  B: {B['label']}  ({B['model']}, vLLM {B.get('vllm_version')}, {B['generated']})")
    print(f"  {tf['prompts']} prompts, top-{A['settings']['top_k']}, "
          f"thinking={'on' if A['settings']['thinking'] else 'off'}")
    print()
    print(f"teacher-forced (same text scored by both; prefill path) -- {tf['positions']} positions")
    ci = tf.get("kld_mean_ci95")
    print(f"  KLD(A||B)   mean {fmt(tf.get('kld_mean'), '.2e')}"
          + (f"  [95% CI {ci[0]:.2e} .. {ci[1]:.2e}]" if ci else "")
          + f"   p50 {fmt(tf.get('kld_p50'), '.2e')}  p99 {fmt(tf.get('kld_p99'), '.2e')}"
            f"  p99.9 {fmt(tf.get('kld_p999'), '.2e')}  max {fmt(tf.get('kld_max'), '.2e')}")
    print(f"  top-1 agree {fmt(tf.get('top1_agree'), '.4%')}    support mass "
          f"{fmt(tf.get('support_mass_mean'), '.4f')}")
    print(f"  dp(ref)     mean|dp| {fmt(tf.get('dp_mean_abs'), '.2e')}   p1 {fmt(tf.get('dp_p1'), '+.4f')}"
          f"  p5 {fmt(tf.get('dp_p5'), '+.4f')}  p50 {fmt(tf.get('dp_p50'), '+.4f')}"
          f"  p95 {fmt(tf.get('dp_p95'), '+.4f')}  p99 {fmt(tf.get('dp_p99'), '+.4f')}")
    print(f"  perplexity  A {fmt(tf.get('ppl_a'), '.4f')}   B {fmt(tf.get('ppl_b'), '.4f')}")
    print(f"  {'category':<14} {'positions':>9}  {'KLD mean':>9}  {'top-1':>8}  {'mean|dp|':>9}")
    for c, v in tf["per_category"].items():
        print(f"  {c:<14} {v['positions']:>9}  {v['kld_mean']:>9.2e}  {v['top1_agree']:>8.3%}"
              f"  {v['dp_mean_abs']:>9.2e}")
    print()
    print(f"generation (each server's own greedy output; decode path) -- {gen['prompts']} prompts")
    print(f"  identical   {gen['identical']}/{gen['prompts']}  ({fmt(gen.get('identical_rate'), '.1%')})")
    if gen["diverged"]:
        print(f"  first split p10 {fmt(gen['first_divergence_p10'], '.0f')}  "
              f"p50 {fmt(gen['first_divergence_p50'], '.0f')}  min {gen['first_divergence_min']}"
              f"  (token index, over {gen['diverged']} diverged prompts)")
    print(f"  common-prefix positions {gen['positions']}: KLD mean {fmt(gen.get('kld_mean'), '.2e')}"
          f"  p99 {fmt(gen.get('kld_p99'), '.2e')}  top-1 agree {fmt(gen.get('top1_agree'), '.4%')}")

    entry = {
        "label": label,
        "a": A["label"], "b": B["label"],
        "model": A["model"],
        "generated": time.strftime("%Y-%m-%d"),
        "vllm_version": A.get("vllm_version"),
        "prompt_set": A.get("prompt_set"),
        "settings": A["settings"],
        "teacher_forced": tf,
        "generation": gen,
        "dimensions": {"model": A["model"], **dimensions},
    }
    if args.notes:
        entry["notes"] = args.notes
    if args.suspect:
        # The teacher-forced numbers are kept but marked: the results page
        # leaves a suspect entry out of the chart and says why in the table.
        entry["suspect"] = args.suspect
    if args.out:
        merge_into(args.out, entry)
        print(f"\nmerged into {args.out} as '{label}'")
    elif args.json:
        print()
        json.dump(entry, sys.stdout, indent=2)
        print()


# ------------------------------------------------------------------------- jitter

def jitter(args):
    """Is the server even deterministic? Score the same text repeatedly and see."""
    info = server_info(args.base_url)
    model = args.model or info.get("served_model_name")
    prompts = load_prompts(args.prompts, args.limit)
    print(f"server   {args.base_url}  vLLM {info.get('vllm_version', '?')}   model {model}")
    print(f"{len(prompts)} prompts x {args.repeats} identical scoring requests, one at a time")
    kls, flips, n = [], 0, 0
    worst = (0.0, None)
    for p in prompts:
        pid = tokenize(args.base_url, model, messages_of(p), args.thinking, args.timeout)
        cont = generate(args.base_url, model, pid, args.max_tokens, args.top_k, args.timeout)["ids"]
        runs = [score(args.base_url, model, pid, cont, args.top_k, args.timeout)["top"]
                for _ in range(args.repeats)]
        for j in range(len(cont)):
            for r in runs[1:]:
                kl, _ = kl_topk(runs[0][j], r[j])
                kls.append(kl)
                same = argmax(runs[0][j]) == argmax(r[j])
                flips += not same
                n += 1
                if kl > worst[0]:
                    worst = (kl, f"{p['id']} pos {j}")
        print(f"  {p['id']:<24} {len(cont)} tokens  max KLD so far {worst[0]:.2e}", flush=True)
    print()
    print(f"run-to-run KLD  mean {statistics.fmean(kls):.2e}  p50 {pct(kls, 50):.2e}  "
          f"p99 {pct(kls, 99):.2e}  max {worst[0]:.2e} ({worst[1]})")
    print(f"top-1 flips     {flips}/{n}  ({flips / n:.3%})")
    entry = {
        "label": args.label or model,
        "model": model,
        "vllm_version": info.get("vllm_version"),
        "generated": time.strftime("%Y-%m-%d"),
        "prompts": len(prompts), "repeats": args.repeats, "positions": n,
        "max_tokens": args.max_tokens, "top_k": args.top_k, "thinking": args.thinking,
        "kld_mean": round(statistics.fmean(kls), 6),
        "kld_p50": round(pct(kls, 50), 6),
        "kld_p99": round(pct(kls, 99), 6),
        "kld_max": round(worst[0], 6),
        "top1_flip_rate": round(flips / n, 5),
    }
    if args.notes:
        entry["notes"] = args.notes
    if args.out:
        merge_into(args.out, entry, key="jitter")
        print(f"merged into {args.out} as '{entry['label']}'")
    elif args.json:
        json.dump(entry, sys.stdout)
        print()


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="record top-k distributions from the running server")
    c.add_argument("--base-url", default="http://127.0.0.1:8000")
    c.add_argument("--model", help="defaults to whatever /v1/models reports")
    c.add_argument("--label", help="name for this configuration (default: model name)")
    c.add_argument("--prompts", default="quality/prompts.jsonl")
    c.add_argument("--ref", metavar="CAPTURE",
                   help="score the continuations stored in this capture instead of "
                        "this server's own; required for a teacher-forced comparison")
    c.add_argument("--max-tokens", type=int, default=256,
                   help="greedy continuation length (default: 256)")
    c.add_argument("--top-k", type=int, default=20,
                   help="alternatives recorded per position; the server caps this at "
                        "--max-logprobs, 20 by default")
    c.add_argument("--concurrency", type=int, default=4)
    c.add_argument("--no-thinking", dest="thinking", action="store_false",
                   help="render the chat template with thinking disabled")
    c.add_argument("--cats", help="comma-separated categories to include")
    c.add_argument("--limit", type=int, help="only the first N prompts (smoke test)")
    c.add_argument("--timeout", type=int, default=600)
    c.add_argument("--out", help="capture path (default: quality/captures/<label>.json.gz)")
    c.add_argument("--notes")
    c.set_defaults(func=capture)

    p = sub.add_parser("compare", help="divergence between two captures")
    p.add_argument("a", help="reference capture")
    p.add_argument("b", help="capture to compare against it")
    p.add_argument("--label", help="name for this comparison (default: 'A vs B')")
    p.add_argument("--out", help="merge the comparison into this JSON file")
    p.add_argument("--json", action="store_true", help="print the full entry as JSON")
    p.add_argument("--notes")
    p.add_argument("--dim", action="append", metavar="KEY=VALUE", default=[],
                   help="dimension tag for filtering on the results page, repeatable")
    p.add_argument("--suspect", metavar="REASON",
                   help="mark the teacher-forced result as an instrument artefact, "
                        "not a model difference (kept in the table, left off the chart)")
    p.set_defaults(func=compare)

    j = sub.add_parser("jitter", help="score identical requests repeatedly; run-to-run noise")
    j.add_argument("--base-url", default="http://127.0.0.1:8000")
    j.add_argument("--model")
    j.add_argument("--prompts", default="quality/prompts.jsonl")
    j.add_argument("--limit", type=int, default=8, help="prompts to use (default: 8)")
    j.add_argument("--repeats", type=int, default=4, help="scoring passes per prompt (default: 4)")
    j.add_argument("--max-tokens", type=int, default=128)
    j.add_argument("--top-k", type=int, default=20)
    j.add_argument("--no-thinking", dest="thinking", action="store_false")
    j.add_argument("--timeout", type=int, default=600)
    j.add_argument("--label", help="name for this server configuration")
    j.add_argument("--out", help="merge the result into this JSON file under 'jitter'")
    j.add_argument("--json", action="store_true")
    j.add_argument("--notes")
    j.set_defaults(func=jitter)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
