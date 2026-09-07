#!/usr/bin/env python3
"""Turn Claude Code telemetry into a usage profile the loop benchmark can replay.

Input: one or more CloudWatch Logs Insights exports (CSV, or the JSON that
`aws logs get-query-results` returns) of the query in
agent-notes/2026-09-07-claude-code-usage-profile.md: one row per
`api_request`, `user_prompt` or `tool_result` event, with token counts and
durations but no prompt text, user ids or tool parameters.

Output: docs/usage.json -- the distributions (context per request, output
length, cache share, cadence, requests per human prompt, tool mix) and two
representative sessions as turn lists, which `agentic.py loop --profile`
replays: a *typical* session (median request count) and the *heavy* one
(largest context).

  ./usage.py export.csv --out docs/usage.json
  ./usage.py day1.json day2.json --gap-minutes 30 --source "work account, 2026-09"

Sessions come from the `session` column when the export has one, else from
gaps of more than --gap-minutes between requests. Token counts are the
provider's tokenizer's; the benchmark replays the same numbers on the local
tokenizer, which is close enough for shape and not for exact sizes.
"""

import argparse
import collections
import csv
import datetime as dt
import json
import statistics
import sys

GAP_MINUTES = 30
RESET_RATIO = 0.5   # context dropping below this fraction of the previous turn = compaction / new task


# --- input ------------------------------------------------------------------

def load(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            if p.endswith(".json"):
                doc = json.load(f)
                for res in doc.get("results", []):
                    rows.append({c["field"]: c["value"] for c in res if not c["field"].startswith("@ptr")})
            else:
                rows.extend(csv.DictReader(f))
    out = []
    for r in rows:
        ev = (r.get("event") or "").replace("claude_code.", "")
        if ev not in ("api_request", "user_prompt", "tool_result"):
            continue
        row = {"event": ev, "t": parse_ts(r["ts"]), "session": r.get("session") or None,
               "model": r.get("model") or None, "tool_name": r.get("tool_name") or None,
               "success": (r.get("success") or "").lower() == "true"}
        for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens",
                  "duration_ms", "prompt_length"):
            v = r.get(k)
            row[k] = int(float(v)) if v not in (None, "") else None
        if ev == "api_request":
            row["context"] = row["input_tokens"] + row["cache_read_tokens"] + row["cache_creation_tokens"]
            row["fresh"] = row["input_tokens"] + row["cache_creation_tokens"]
            row["start"] = row["t"] - dt.timedelta(milliseconds=row["duration_ms"] or 0)
        out.append(row)
    out.sort(key=lambda r: r["t"])
    return out


def parse_ts(s):
    s = s.strip().replace("T", " ").replace("Z", "")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s[:26], fmt)
        except ValueError:
            pass
    return dt.datetime.fromtimestamp(float(s) / 1000)   # epoch ms


# --- stats ------------------------------------------------------------------

def pct(values, p):
    s = sorted(values)
    if not s:
        return None
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summary(values, digits=0):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    r = lambda x: round(x, digits) if digits else int(round(x))
    return {"n": len(vals), "min": r(min(vals)), "p10": r(pct(vals, 10)), "p50": r(pct(vals, 50)),
            "p90": r(pct(vals, 90)), "p99": r(pct(vals, 99)), "max": r(max(vals)),
            "mean": round(statistics.mean(vals), 1), "sum": r(sum(vals))}


def log2_hist(values):
    h = collections.Counter(v.bit_length() - 1 for v in values if v and v > 0)
    return [{"from": 2 ** b, "to": 2 ** (b + 1) - 1, "n": h[b]} for b in sorted(h)]


def bin_hist(values, width):
    h = collections.Counter(v // width for v in values if v is not None)
    return [{"from": b * width, "to": (b + 1) * width, "n": h[b]} for b in sorted(h)]


# --- sessions ---------------------------------------------------------------

def split_sessions(api, gap_minutes):
    """Group requests into sessions: by id when the export has one, else by gaps."""
    if all(r["session"] for r in api):
        groups = collections.defaultdict(list)
        for r in api:
            groups[r["session"]].append(r)
        return sorted(groups.values(), key=lambda s: s[0]["t"])
    sessions, cur = [], []
    for r in api:
        if cur and (r["start"] - cur[-1]["t"]).total_seconds() > gap_minutes * 60:
            sessions.append(cur)
            cur = []
        cur.append(r)
    if cur:
        sessions.append(cur)
    return sessions


def turns_of(session):
    """The per-turn shape the benchmark replays."""
    turns = []
    for i, r in enumerate(session):
        prev = session[i - 1] if i else None
        t = {"turn": i + 1, "context": r["context"], "fresh": r["fresh"],
             "cache_read": r["cache_read_tokens"], "output": r["output_tokens"]}
        if prev:
            t["gap_s"] = round((r["start"] - prev["t"]).total_seconds(), 1)
            if r["context"] < prev["context"] * RESET_RATIO:
                t["reset"] = True
            if t["gap_s"] < 0:
                t["parallel"] = True
        turns.append(t)
    return turns


def describe(session, name=None):
    turns = turns_of(session)
    d = {
        "day": session[0]["t"].date().isoformat(),
        "requests": len(session),
        "minutes": round((session[-1]["t"] - session[0]["start"]).total_seconds() / 60, 1),
        "context_first": session[0]["context"],
        "context_max": max(r["context"] for r in session),
        "context_last": session[-1]["context"],
        "output_total": sum(r["output_tokens"] for r in session),
        "resets": sum(1 for t in turns if t.get("reset")),
        "parallel_turns": sum(1 for t in turns if t.get("parallel")),
        "models": dict(collections.Counter(r["model"] for r in session)),
    }
    if name:
        d = {"name": name, **d, "turns": turns}
    return d


# --- main -------------------------------------------------------------------

def build(rows, gap_minutes, source, min_turns):
    api = [r for r in rows if r["event"] == "api_request"]
    prompts = [r for r in rows if r["event"] == "user_prompt"]
    tools = [r for r in rows if r["event"] == "tool_result"]
    if not api:
        sys.exit("usage.py: no api_request rows in the input")
    sessions = split_sessions(api, gap_minutes)

    # cadence inside sessions: gap from one request's end to the next one's start
    gaps = [(b["start"] - a["t"]).total_seconds() for s in sessions for a, b in zip(s, s[1:])]
    in_flight = []
    for s in sessions:
        span = (s[-1]["t"] - s[0]["start"]).total_seconds()
        if span > 0:
            in_flight.append(sum(r["duration_ms"] or 0 for r in s) / 1000 / span)
    deltas = [b["context"] - a["context"] for s in sessions for a, b in zip(s, s[1:])
              if b["context"] >= a["context"] * RESET_RATIO]

    # requests per human prompt: api_request rows between consecutive user_prompt rows
    per_prompt, n, seen = [], 0, False
    for r in rows:
        if r["event"] == "user_prompt":
            if seen:
                per_prompt.append(n)
            n, seen = 0, True
        elif r["event"] == "api_request" and seen:
            n += 1
    if seen:
        per_prompt.append(n)

    tool_rows = []
    for (name, ok), grp in sorted(_group(tools, lambda r: (r["tool_name"], r["success"])).items(),
                                  key=lambda kv: -len(kv[1])):
        ds = [r["duration_ms"] for r in grp if r["duration_ms"] is not None]
        tool_rows.append({"tool": name, "success": ok, "calls": len(grp),
                          "ms_p50": int(pct(ds, 50)) if ds else None,
                          "ms_p90": int(pct(ds, 90)) if ds else None,
                          "ms_max": max(ds) if ds else None})

    # representative sessions: the lower median by request count (a task, not
    # the whole afternoon), and the one that reached the largest context
    candidates = [s for s in sessions if len(s) >= min_turns] or sessions
    by_len = sorted(candidates, key=len)
    typical = by_len[(len(by_len) - 1) // 2]
    heavy = max(candidates, key=lambda s: max(r["context"] for r in s))

    total_ctx = sum(r["context"] for r in api)
    total_cached = sum(r["cache_read_tokens"] for r in api)
    return {
        "schema_version": 1,
        "generated": dt.date.today().isoformat(),
        "source": {
            "description": source,
            "from": rows[0]["t"].date().isoformat(),
            "to": rows[-1]["t"].date().isoformat(),
            "requests": len(api),
            "prompts": len(prompts),
            "tool_calls": len(tools),
            "sessions": len(sessions),
            "session_rule": "session id" if all(r["session"] for r in api) else f"gap > {gap_minutes} min",
            "models": dict(collections.Counter(r["model"] for r in api)),
        },
        "requests": {
            "context": summary([r["context"] for r in api]),
            "fresh_prefill": summary([r["fresh"] for r in api]),
            "cache_read": summary([r["cache_read_tokens"] for r in api]),
            "output": summary([r["output_tokens"] for r in api]),
            "duration_ms": summary([r["duration_ms"] for r in api]),
            "provider_output_tok_s": summary(
                [r["output_tokens"] / (r["duration_ms"] / 1000) for r in api if r["duration_ms"]], 1),
            "cache_share": round(total_cached / total_ctx, 4) if total_ctx else None,
            "cold_requests": sum(1 for r in api if r["cache_read_tokens"] < 1000),
            "output_hist_log2": log2_hist([r["output_tokens"] for r in api]),
            "context_hist_20k": bin_hist([r["context"] for r in api], 20000),
        },
        "sessions": {
            "requests": summary([len(s) for s in sessions]),
            "minutes": summary([(s[-1]["t"] - s[0]["start"]).total_seconds() / 60 for s in sessions], 1),
            "context_max": summary([max(r["context"] for r in s) for s in sessions]),
            "list": [describe(s) for s in sessions],
        },
        "cadence": {
            "gap_s": summary(gaps, 1),
            "gap_hist": _gap_hist(gaps),
            "parallel_pairs": sum(1 for g in gaps if g < 0),
            "in_flight_fraction": summary(in_flight, 2),
        },
        "growth": {
            "context_delta_per_turn": summary(deltas),
            "context_delta_positive": summary([d for d in deltas if d > 0]),
        },
        "prompts": {
            "chars": summary([r["prompt_length"] for r in prompts]),
            "requests_per_prompt": summary(per_prompt),
            "requests_per_prompt_hist": _per_prompt_hist(per_prompt),
        },
        "tools": tool_rows,
        "replay": {
            "typical": describe(typical, "typical"),
            "heavy": describe(heavy, "heavy"),
        },
    }


def _group(rows, key):
    g = collections.defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    return g


def _gap_hist(gaps):
    edges = [("parallel", -1e9, 0), ("<2s", 0, 2), ("<10s", 2, 10), ("<60s", 10, 60), ("<5m", 60, 300), ("5m+", 300, 1e9)]
    return [{"bucket": name, "n": sum(1 for g in gaps if lo <= g < hi)} for name, lo, hi in edges]


def _per_prompt_hist(per):
    edges = [("0", 0, 0), ("1", 1, 1), ("2-5", 2, 5), ("6-20", 6, 20), ("21+", 21, 10 ** 9)]
    return [{"bucket": name, "n": sum(1 for n in per if lo <= n <= hi)} for name, lo, hi in edges]


def show(profile):
    s, rq, c, p = profile["source"], profile["requests"], profile["cadence"], profile["prompts"]
    print(f"source   {s['description']}: {s['from']} .. {s['to']}, {s['requests']} requests, "
          f"{s['prompts']} prompts, {s['tool_calls']} tool calls, {s['sessions']} sessions ({s['session_rule']})")
    row = lambda name, d: print(f"{name:26s} p10 {d['p10']:>9} p50 {d['p50']:>9} p90 {d['p90']:>9} p99 {d['p99']:>9} max {d['max']:>9}")
    row("context tokens", rq["context"])
    row("fresh prefill tokens", rq["fresh_prefill"])
    row("output tokens", rq["output"])
    row("gap between requests, s", c["gap_s"])
    row("requests per prompt", p["requests_per_prompt"])
    print(f"cache share {rq['cache_share']:.3f}, cold requests {rq['cold_requests']}, parallel pairs {c['parallel_pairs']}")
    for name in ("typical", "heavy"):
        r = profile["replay"][name]
        print(f"{name:8s} {r['requests']} turns, {r['minutes']} min, context {r['context_first']} -> max {r['context_max']} "
              f"-> {r['context_last']}, {r['resets']} resets, {r['output_total']} output tokens")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="Logs Insights exports: .csv, or .json from get-query-results")
    ap.add_argument("--out", help="write the profile here (default: print)")
    ap.add_argument("--gap-minutes", type=float, default=GAP_MINUTES,
                    help="a pause longer than this starts a new session when the export has no session id")
    ap.add_argument("--min-turns", type=int, default=5, help="sessions shorter than this are not picked for replay")
    ap.add_argument("--source", default="Claude Code OpenTelemetry logs", help="one line about where the data came from")
    args = ap.parse_args()
    rows = load(args.files)
    profile = build(rows, args.gap_minutes, args.source, args.min_turns)
    show(profile)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(profile, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"\nwrote {args.out}")
    else:
        print()
        json.dump(profile, sys.stdout, indent=1)
        print()


if __name__ == "__main__":
    main()
