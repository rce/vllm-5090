#!/usr/bin/env python3
"""Agentic-loop checks against an OpenAI-compatible vLLM server.

Sibling of bench.py, for the workload bench.py does not model: an agent that
re-sends a growing conversation every turn and mostly wants a tool call back.
Two subcommands, both stdlib-only:

  tools   Does tool calling actually work through this server's parsers? Sends
          a handful of fixed requests with a function schema and checks what
          comes back: a structured tool_calls entry (not the model's tool markup
          leaking into content), valid JSON arguments, a sane answer once the
          tool result is fed back, parallel calls, no false calls, and the same
          through the streaming API. A profile with the wrong --tool-call-parser
          or --reasoning-parser fails here and nowhere else.

  loop    Replay a synthetic agent transcript: a shared system prompt, then N
          turns of (assistant tool call, ~1K-token tool result), asking for the
          next step after each. Measures what a loop feels: time to first token
          per turn as the context grows, and how much of that prefix the server
          re-computed. Runs warm (natural, prefix cache allowed) and cold (every
          turn made unique, so the whole context is re-prefilled) so the cache's
          contribution is a number rather than an assumption. Output length is
          fixed with ignore_eos, as in bench.py, so runs are comparable.

  ./agentic.py tools --out docs/results.json --label "qwen3.6-35b-a3b"
  ./agentic.py loop  --out docs/results.json --label "qwen3.6-35b-a3b" --agents 1,4
"""

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------- transport


def post(base_url, path, body, timeout):
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"HTTP {e.code} on {path}: {detail}") from None


def get_text(base_url, path, timeout=10):
    with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def server_info(base_url, timeout=10):
    info = {}
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=timeout) as r:
            models = json.load(r).get("data", [])
        if models:
            info["served_model_name"] = models[0].get("id")
            info["max_model_len"] = models[0].get("max_model_len")
    except Exception as e:
        raise SystemExit(f"agentic.py: cannot reach {base_url}: {e}")
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/version", timeout=timeout) as r:
            info["vllm_version"] = json.load(r).get("version")
    except Exception:
        pass
    return info


def prefix_cache_counters(base_url):
    """(queries, hits) in tokens from /metrics, summed over labels; None if absent."""
    try:
        text = get_text(base_url, "/metrics")
    except Exception:
        return None
    totals = {"queries": 0.0, "hits": 0.0}
    found = False
    for line in text.splitlines():
        m = re.match(r"vllm:prefix_cache_(queries|hits)(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)", line)
        if m:
            totals[m.group(1)] += float(m.group(2))
            found = True
    return (totals["queries"], totals["hits"]) if found else None


def merge_into(path, entry, key):
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


def parse_dims(pairs):
    dims = {}
    for d in pairs:
        if "=" not in d:
            raise SystemExit(f"agentic.py: --dim expects KEY=VALUE, got {d!r}")
        k, v = d.split("=", 1)
        dims[k.strip()] = v.strip()
    return dims


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ---------------------------------------------------------------- tools

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the workspace and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path relative to the workspace root"}},
                "required": ["path"],
            },
        },
    },
]

# Model-side tool markup that must never reach `content` if the parser works.
LEAK_MARKERS = ("<tool_call>", "</tool_call>", "<function=", "<think>", "</think>", "<TOOLCALL>")


def chat(base_url, model, messages, timeout, stream=False, max_tokens=2048):
    body = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": 0,
    }
    if not stream:
        t0 = time.perf_counter()
        resp = post(base_url, "/v1/chat/completions", body, timeout)
        ch = resp["choices"][0]
        msg = ch["message"]
        return {
            "finish": ch.get("finish_reason"),
            "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "tool_calls": msg.get("tool_calls") or [],
            "raw_message": msg,
            "output_tokens": (resp.get("usage") or {}).get("completion_tokens"),
            "latency_s": round(time.perf_counter() - t0, 3),
        }

    # Streaming: reassemble tool calls from deltas the way a client would.
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    content, reasoning, finish, usage = [], [], None, {}
    calls = {}  # index -> {id, name, arguments}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
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
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content"):
                    content.append(d["content"])
                if d.get("reasoning_content") or d.get("reasoning"):
                    reasoning.append(d.get("reasoning_content") or d.get("reasoning"))
                for tc in d.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    tool_calls = [{"id": c["id"], "type": "function",
                   "function": {"name": c["name"], "arguments": c["arguments"]}}
                  for _, c in sorted(calls.items())]
    return {
        "finish": finish,
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "tool_calls": tool_calls,
        "raw_message": None,
        "output_tokens": usage.get("completion_tokens"),
        "latency_s": round(time.perf_counter() - t0, 3),
    }


def parse_args_json(tc):
    try:
        return json.loads(tc["function"]["arguments"] or "{}")
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def leaked(text):
    return [m for m in LEAK_MARKERS if m in text]


def check_tools(base_url, model, timeout):
    """Run the fixed checks; each returns (passed, detail) plus the response stats."""
    checks = []
    reasoning_seen = False

    def record(name, r, passed, detail):
        checks.append({
            "name": name, "pass": bool(passed), "detail": detail,
            "finish": r["finish"], "output_tokens": r["output_tokens"],
            "latency_s": r["latency_s"], "reasoning_tokens_seen": bool(r["reasoning"]),
        })
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<22} {detail}  "
              f"({r['finish']}, {r['output_tokens']} tok, {r['latency_s']} s)")

    def one_call_ok(r, city):
        """Shared assertions for 'exactly one get_weather call for <city>'."""
        if leaked(r["content"]):
            return False, f"tool/think markup leaked into content: {leaked(r['content'])}"
        if r["finish"] == "length":
            return False, "hit max_tokens before calling"
        if len(r["tool_calls"]) != 1:
            return False, f"expected 1 tool call, got {len(r['tool_calls'])} (finish={r['finish']})"
        tc = r["tool_calls"][0]
        if tc["function"]["name"] != "get_weather":
            return False, f"wrong tool: {tc['function']['name']}"
        a = parse_args_json(tc)
        if a is None:
            return False, f"arguments are not JSON: {tc['function']['arguments'][:80]!r}"
        if city.lower() not in str(a.get("city", "")).lower():
            return False, f"city argument {a.get('city')!r} does not mention {city}"
        if r["finish"] != "tool_calls":
            return False, f"call parsed but finish_reason is {r['finish']!r}, not 'tool_calls'"
        return True, f"get_weather({json.dumps(a)})"

    # 1. one call, non-streaming
    msgs = [{"role": "user", "content": "What is the weather in Helsinki right now? Use the tool."}]
    r = chat(base_url, model, msgs, timeout)
    reasoning_seen |= bool(r["reasoning"])
    ok, detail = one_call_ok(r, "Helsinki")
    record("single_call", r, ok, detail)
    first = r

    # 2. feed the result back and expect an answer, not another call
    if ok:
        tc = first["tool_calls"][0]
        assistant = first["raw_message"] if first["raw_message"] else {"role": "assistant", "tool_calls": first["tool_calls"]}
        assistant = {k: v for k, v in assistant.items() if k in ("role", "content", "tool_calls")}
        assistant["role"] = "assistant"
        msgs2 = msgs + [assistant, {
            "role": "tool", "tool_call_id": tc.get("id") or "call_0", "name": "get_weather",
            "content": json.dumps({"temperature_c": 7, "conditions": "light rain", "wind_kph": 23}),
        }]
        r2 = chat(base_url, model, msgs2, timeout)
        reasoning_seen |= bool(r2["reasoning"])
        c = r2["content"]
        if leaked(c):
            record("tool_result_roundtrip", r2, False, f"markup leaked: {leaked(c)}")
        elif r2["tool_calls"]:
            record("tool_result_roundtrip", r2, False, "called a tool again instead of answering")
        elif r2["finish"] != "stop":
            record("tool_result_roundtrip", r2, False, f"finish_reason {r2['finish']!r}")
        elif not ("7" in c and "rain" in c.lower()):
            record("tool_result_roundtrip", r2, False, f"answer does not use the result: {c[:80]!r}")
        else:
            record("tool_result_roundtrip", r2, True, f"answer uses the result ({len(c)} chars)")
    else:
        checks.append({"name": "tool_result_roundtrip", "pass": False, "detail": "skipped: single_call failed",
                       "finish": None, "output_tokens": None, "latency_s": None, "reasoning_tokens_seen": False})
        print("  SKIP  tool_result_roundtrip  single_call failed")

    # 3. parallel calls
    msgs = [{"role": "user", "content": "I need the current weather in both Helsinki and Oslo. "
                                        "Call the weather tool for each city."}]
    r = chat(base_url, model, msgs, timeout)
    reasoning_seen |= bool(r["reasoning"])
    cities = []
    bad = None
    for tc in r["tool_calls"]:
        a = parse_args_json(tc)
        if a is None or tc["function"]["name"] != "get_weather":
            bad = f"bad call {tc['function']['name']}({tc['function']['arguments'][:60]!r})"
        else:
            cities.append(str(a.get("city", "")).lower())
    if leaked(r["content"]):
        record("parallel_calls", r, False, f"markup leaked: {leaked(r['content'])}")
    elif bad:
        record("parallel_calls", r, False, bad)
    elif any("helsinki" in c for c in cities) and any("oslo" in c for c in cities):
        record("parallel_calls", r, True, f"{len(r['tool_calls'])} calls in one turn: {cities}")
    else:
        record("parallel_calls", r, False, f"{len(r['tool_calls'])} call(s): {cities} (finish={r['finish']})")

    # 4. no false call: tools offered, none needed
    msgs = [{"role": "user", "content": "Reply with the single word: hello"}]
    r = chat(base_url, model, msgs, timeout, max_tokens=1024)
    reasoning_seen |= bool(r["reasoning"])
    if r["tool_calls"]:
        record("no_false_call", r, False, f"called {r['tool_calls'][0]['function']['name']} unprompted")
    elif leaked(r["content"]):
        record("no_false_call", r, False, f"markup leaked: {leaked(r['content'])}")
    elif r["finish"] != "stop" or not r["content"].strip():
        record("no_false_call", r, False, f"finish={r['finish']!r}, content={r['content'][:40]!r}")
    else:
        record("no_false_call", r, True, f"plain answer {r['content'].strip()[:30]!r}")

    # 5. the same single call through the streaming API
    msgs = [{"role": "user", "content": "What is the weather in Helsinki right now? Use the tool."}]
    try:
        r = chat(base_url, model, msgs, timeout, stream=True)
        reasoning_seen |= bool(r["reasoning"])
        ok, detail = one_call_ok(r, "Helsinki")
        record("streaming_call", r, ok, detail)
    except Exception as e:
        record("streaming_call", {"finish": None, "output_tokens": None, "latency_s": None, "reasoning": ""},
               False, f"{type(e).__name__}: {e}")

    # 6. read_file with a path argument, checking the argument survives intact
    msgs = [{"role": "user", "content": "Show me the contents of src/utils/date_parse.py."}]
    r = chat(base_url, model, msgs, timeout)
    reasoning_seen |= bool(r["reasoning"])
    if leaked(r["content"]):
        record("path_argument", r, False, f"markup leaked: {leaked(r['content'])}")
    elif len(r["tool_calls"]) != 1 or r["tool_calls"][0]["function"]["name"] != "read_file":
        record("path_argument", r, False, f"expected one read_file call, got "
               f"{[t['function']['name'] for t in r['tool_calls']]} (finish={r['finish']})")
    else:
        a = parse_args_json(r["tool_calls"][0])
        if a is None:
            record("path_argument", r, False, "arguments are not JSON")
        elif a.get("path") != "src/utils/date_parse.py":
            record("path_argument", r, False, f"path mangled: {a.get('path')!r}")
        else:
            record("path_argument", r, True, "read_file(path) exact")

    return checks, reasoning_seen


def cmd_tools(args):
    info = server_info(args.base_url)
    model = args.model or info["served_model_name"]
    label = args.label or model
    print(f"server   {args.base_url}  vLLM {info.get('vllm_version', '?')}")
    print(f"model    {model}")
    print(f"checks   tool calling through the server's parsers, temperature 0, template defaults\n")
    checks, reasoning_seen = check_tools(args.base_url, model, args.timeout)
    passed = sum(1 for c in checks if c["pass"])
    print(f"\n{passed}/{len(checks)} passed; reasoning field seen: {reasoning_seen}")
    entry = {
        "label": label,
        "model": model,
        "generated": time.strftime("%Y-%m-%d"),
        "vllm_version": info.get("vllm_version"),
        "passed": passed,
        "total": len(checks),
        "reasoning_field_seen": reasoning_seen,
        "checks": checks,
        "dimensions": {"model": model, **parse_dims(args.dim)},
    }
    if args.notes:
        entry["notes"] = args.notes
    if args.out:
        merge_into(args.out, entry, "toolcalls")
        print(f"merged into {args.out} as '{label}'")
    else:
        json.dump(entry, sys.stdout, indent=2)
        print()


# ---------------------------------------------------------------- loop

SYSTEM_BASE = (
    "You are a coding agent working inside a software repository. You have tools "
    "for reading files, and you work in small verified steps: read what you need, "
    "state the next concrete action, and call at most one tool per turn. Do not "
    "guess at file contents; read them. Keep explanations short and put the "
    "reasoning for each step in one or two sentences before the tool call. "
    "The repository is a mid-sized Python service with a web layer, a job "
    "queue, a data access layer and a test suite; module names follow the "
    "package layout under src/. When you are finished, summarise what changed. "
)

FILE_FILLER = (
    "def process_batch(items, *, retries=3):\n"
    "    \"\"\"Apply the handler to every item, retrying transient failures.\"\"\"\n"
    "    results = []\n"
    "    for item in items:\n"
    "        for attempt in range(retries):\n"
    "            try:\n"
    "                results.append(handle(item))\n"
    "                break\n"
    "            except TransientError as exc:\n"
    "                log.warning('retry %d for %s: %s', attempt, item.id, exc)\n"
    "        else:\n"
    "            results.append(Failure(item.id))\n"
    "    return results\n\n"
)


def words(text, n):
    w = text.split()
    out = []
    while len(out) < n:
        out.extend(w)
    return " ".join(out[:n])


def build_system(system_words, salt=None):
    text = words(SYSTEM_BASE, system_words)
    # Cold mode salts the very first token so no prefix at all is reusable.
    return (f"Run {salt}. " if salt is not None else "") + text


def tool_output(agent, turn, tool_words):
    header = f"# src/module_{agent}_{turn}.py  (agent {agent}, turn {turn})\n"
    body = FILE_FILLER
    out = header
    while len(out.split()) < tool_words:
        out += body.replace("process_batch", f"process_batch_{turn}_{len(out) % 97}")
    return out


def transcript(agent, upto_turn, system_words, tool_words, cold_salt=None):
    msgs = [{"role": "system", "content": build_system(system_words, cold_salt)},
            {"role": "user", "content": f"Task {agent}: the retry logic in the batch processor drops "
                                        f"the last failure's message. Find where and fix it."}]
    for t in range(1, upto_turn + 1):
        path = f"src/module_{agent}_{t}.py"
        msgs.append({"role": "assistant", "content": f"Reading {path} next.",
                     "tool_calls": [{"id": f"call_{agent}_{t}", "type": "function",
                                     "function": {"name": "read_file", "arguments": json.dumps({"path": path})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{agent}_{t}", "name": "read_file",
                     "content": tool_output(agent, t, tool_words)})
    return msgs


def one_turn(base_url, model, messages, output_tokens, timeout):
    body = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "max_tokens": output_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    prompt_tokens = completion_tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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
                    completion_tokens = chunk["usage"].get("completion_tokens", 0)
                if ttft is None:
                    for ch in chunk.get("choices") or []:
                        d = ch.get("delta") or {}
                        if d.get("content") or d.get("reasoning") or d.get("reasoning_content") or d.get("tool_calls"):
                            ttft = time.perf_counter() - t0
                            break
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    wall = time.perf_counter() - t0
    if ttft is None:
        ttft = wall
    return {"ok": True, "ttft": ttft, "wall": wall,
            "prompt_tokens": prompt_tokens, "output_tokens": completion_tokens}


def run_agent(base_url, model, agent, turns, system_words, tool_words, output_tokens, timeout, cold):
    """One agent's loop: after each appended tool result, ask for the next step."""
    rows = []
    for t in range(1, turns + 1):
        salt = f"{agent}-{t}" if cold else None
        msgs = transcript(agent, t, system_words, tool_words, cold_salt=salt)
        r = one_turn(base_url, model, msgs, output_tokens, timeout)
        r["turn"] = t
        rows.append(r)
        if not r["ok"]:
            break
    return rows


def run_level(base_url, model, agents, mode, args):
    cold = mode == "cold"
    before = prefix_cache_counters(base_url)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=agents) as pool:
        per_agent = list(pool.map(
            lambda a: run_agent(base_url, model, a, args.turns, args.system_words,
                                args.tool_words, args.output_tokens, args.timeout, cold),
            range(1, agents + 1)))
    wall = time.perf_counter() - t0
    after = prefix_cache_counters(base_url)

    per_turn = []
    for t in range(1, args.turns + 1):
        rs = [rows[t - 1] for rows in per_agent if len(rows) >= t and rows[t - 1]["ok"]]
        if not rs:
            break
        per_turn.append({
            "turn": t,
            "context_tokens": rs[0]["prompt_tokens"],
            "ttft_p50_s": round(pct([r["ttft"] for r in rs], 50), 3),
            "ttft_max_s": round(max(r["ttft"] for r in rs), 3),
            "decode_tok_s_per_stream": round(statistics.median(
                (r["output_tokens"] - 1) / (r["wall"] - r["ttft"]) for r in rs if r["wall"] > r["ttft"]), 1),
        })
    errors = [r["error"] for rows in per_agent for r in rows if not r["ok"]]
    turns_ok = sum(1 for rows in per_agent for r in rows if r["ok"])
    entry = {
        "agents": agents,
        "mode": mode,
        "turns_requested": agents * args.turns,
        "turns_ok": turns_ok,
        "status": "ok" if turns_ok == agents * args.turns else ("partial" if turns_ok else "failed"),
        "wall_s": round(wall, 2),
    }
    if errors:
        entry["error"] = errors[0]
    if per_turn:
        ttfts = [r["ttft"] for rows in per_agent for r in rows if r["ok"]]
        entry.update({
            "ttft_first_turn_s": per_turn[0]["ttft_p50_s"],
            "ttft_last_turn_s": per_turn[-1]["ttft_p50_s"],
            "ttft_mean_s": round(statistics.mean(ttfts), 3),
            "ttft_p95_s": round(pct(ttfts, 95), 3),
            "context_tokens_last": per_turn[-1]["context_tokens"],
            "prompt_tokens_total": sum(r["prompt_tokens"] for rows in per_agent for r in rows if r["ok"]),
            "turns_per_minute": round(turns_ok / wall * 60, 1) if wall else None,
            "per_turn": per_turn,
        })
    if before and after and after[0] > before[0]:
        q, h = after[0] - before[0], after[1] - before[1]
        entry["prefix_cache_hit_rate"] = round(h / q, 4)
        entry["prefix_cache_queried_tokens"] = int(q)
    return entry


def cmd_loop(args):
    info = server_info(args.base_url)
    model = args.model or info["served_model_name"]
    label = args.label or model
    agent_levels = [int(x) for x in args.agents.split(",") if x.strip()]
    modes = ["warm", "cold"] if args.mode == "both" else [args.mode]
    print(f"server   {args.base_url}  vLLM {info.get('vllm_version', '?')}")
    print(f"model    {model}  (max_model_len {info.get('max_model_len', '?')})")
    print(f"workload {args.turns} turns, ~{args.tool_words} words per tool result, "
          f"{args.system_words}-word system prompt, exactly {args.output_tokens} output tokens per turn")
    if prefix_cache_counters(args.base_url) is None:
        print("note     /metrics has no prefix-cache counters; hit rate will be missing")
    print()

    # Warm-up: one short turn so compilation/JIT is not charged to turn 1.
    w = one_turn(args.base_url, model, transcript(0, 1, args.system_words, args.tool_words), 8, args.timeout)
    if not w["ok"]:
        raise SystemExit(f"agentic.py: warmup failed: {w['error']}")

    hdr = f"{'agents':>6} {'mode':>5}  {'ok':>7}  {'TTFT t1':>8} {'TTFT last':>9} {'TTFT p95':>9}  " \
          f"{'ctx last':>8}  {'cache hit':>9}  {'turns/min':>9}  {'wall':>7}"
    print(hdr)
    print("-" * len(hdr))
    levels = []
    for agents in agent_levels:
        for mode in modes:
            e = run_level(args.base_url, model, agents, mode, args)
            levels.append(e)
            okstr = f"{e['turns_ok']}/{e['turns_requested']}"
            if e.get("per_turn"):
                hit = e.get("prefix_cache_hit_rate")
                print(f"{agents:>6} {mode:>5}  {okstr:>7}  {e['ttft_first_turn_s']:>8.3f} "
                      f"{e['ttft_last_turn_s']:>9.3f} {e['ttft_p95_s']:>9.3f}  "
                      f"{e['context_tokens_last']:>8}  {(f'{hit*100:.1f}%' if hit is not None else '—'):>9}  "
                      f"{e['turns_per_minute']:>9.1f}  {e['wall_s']:>6.1f}s")
            else:
                print(f"{agents:>6} {mode:>5}  {okstr:>7}  FAILED  {e.get('error', '')[:70]}")
            if e["status"] == "failed":
                break

    entry = {
        "label": label,
        "model": model,
        "generated": time.strftime("%Y-%m-%d"),
        "vllm_version": info.get("vllm_version"),
        "max_model_len": info.get("max_model_len"),
        "workload": {
            "turns": args.turns,
            "tool_words": args.tool_words,
            "system_words": args.system_words,
            "output_tokens": args.output_tokens,
            "ignore_eos": True,
            "thinking": False,
        },
        "levels": levels,
        "dimensions": {"model": model, **parse_dims(args.dim)},
    }
    if args.notes:
        entry["notes"] = args.notes
    if args.out:
        merge_into(args.out, entry, "loops")
        print(f"\nmerged into {args.out} as '{label}'")
    else:
        print()
        json.dump(entry, sys.stdout, indent=2)
        print()


# ---------------------------------------------------------------- cli


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--base-url", default="http://127.0.0.1:8000")
        p.add_argument("--model", help="defaults to whatever /v1/models reports")
        p.add_argument("--label", help="name for this configuration in the output")
        p.add_argument("--timeout", type=int, default=600)
        p.add_argument("--out", help="merge the result into this JSON file")
        p.add_argument("--notes")
        p.add_argument("--dim", action="append", metavar="KEY=VALUE", default=[],
                       help="dimension tag for filtering on the results page, repeatable")

    t = sub.add_parser("tools", help="tool-calling round-trip checks")
    common(t)
    t.set_defaults(func=cmd_tools)

    l = sub.add_parser("loop", help="replay a growing agent transcript")
    common(l)
    l.add_argument("--agents", default="1,4", help="parallel agents, comma-separated levels (default: 1,4)")
    l.add_argument("--mode", choices=["warm", "cold", "both"], default="both",
                   help="warm = prefix cache allowed, cold = every turn unique (default: both)")
    l.add_argument("--turns", type=int, default=16)
    l.add_argument("--tool-words", type=int, default=400, help="words of code per tool result (~1.2K tokens)")
    l.add_argument("--system-words", type=int, default=600)
    l.add_argument("--output-tokens", type=int, default=128, help="generated per turn, exactly")
    l.set_defaults(func=cmd_loop)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
