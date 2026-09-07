# 2026-09-07 — A usage profile from real Claude Code telemetry

`agentic.py loop` replays a made-up transcript: a 600-word system prompt and
sixteen "read this file" turns of ~1.2K tokens each, 256-token answers. The
shape is plausible but invented. Claude Code exports OpenTelemetry logs
that describe real sessions request by request, and the user has those for
actual work in AWS CloudWatch. This note is the extraction plan: which
CloudWatch Logs Insights queries to run, what comes back, and how it turns
into a replay profile for the benchmark.

Nothing here has been run yet. The numbers arrive when the user runs the
queries; results and the profile derived from them go in a follow-up
section at the bottom.

## What Claude Code emits

Each log record is one event, named in the `event.name` attribute (older
convention; the record body carries the same string). Every event carries
`session.id`, `app.version`, `organization.id`, `user.account_uuid`,
`user.id`, `terminal.type`, and `user.email` when the account has one.

| event | attributes that matter here |
| --- | --- |
| `claude_code.api_request` | `model`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_creation_tokens`, `duration_ms`, `cost_usd`, `speed` on newer versions |
| `claude_code.tool_result` | `tool_name`, `success`, `duration_ms`, `error`; `tool_parameters` for Bash and a few others (the command text: do not export) |
| `claude_code.user_prompt` | `prompt_length`; `prompt` only if prompt logging was enabled (do not export) |
| `claude_code.api_error` | `model`, `error`, `status_code`, `duration_ms`, `attempt` |
| `claude_code.tool_decision` | `tool_name`, `decision`, `source` |

The metrics side (`claude_code.token.usage`, `claude_code.cost.usage`,
`claude_code.session.count`, ...) is aggregated and adds nothing the logs do
not already have per request.

What the logs do **not** contain is the prompt and tool-result text (unless
opted in), so the replay stays shape-faithful: synthetic filler at the
measured sizes, which is what the loop benchmark already does. Token counts
are Claude tokenizer counts; Qwen's tokenizer lands within roughly 10–20 %
of them on code and English, so sizes transfer, exact numbers do not.

## What the profile needs

Per request, the context the server has to prefill is
`input_tokens + cache_read_tokens + cache_creation_tokens`; the part it can
skip with a warm prefix cache is `cache_read_tokens`; what it has to decode
is `output_tokens`. From the per-request rows of a session, ordered by time:

1. **Context growth per turn** — how fast the transcript grows and where it
   plateaus (compaction shows as a drop). Replaces the fixed 1.2K/turn.
2. **Output length per turn** — tool-calling turns are short; the sweeps
   force 256. The distribution, not the mean, is what matters.
3. **Cache share** — `cache_read / context` per request, and how often a
   request is a cold prefill (cache_read ≈ 0 after a gap).
4. **Cadence** — gap between consecutive requests in a session, which is
   tool time plus the human. Long gaps are when a local server evicts.
5. **Concurrency** — requests whose `[ts − duration_ms, ts]` windows overlap
   within a session are subagents running in parallel; across sessions it is
   how many people or terminals hit the server at once.
6. **Requests per human prompt** — the agent-loop length per user turn,
   from `user_prompt` events interleaved with `api_request`.
7. **Tool mix and durations** — which tools, how often, how long they take.
8. **Model mix** — which requests went to a small model (subagents, quick
   turns); on a single 5090 those would all hit the same server.

Everything in that list is computable offline from one raw export of the
three event types. The aggregate queries below are for a quick look in the
console and as cross-checks against the offline numbers.

## Field names

The queries assume the OpenTelemetry collector's CloudWatch Logs exporter,
which writes each record as JSON with `attributes` and `resource` objects.
Logs Insights flattens those, so an attribute called `event.name` becomes
the field `attributes.event.name`. If the records arrived another way (a
Firehose path, or an exporter that flattens attributes to the top level),
the prefix differs; run the discovery query first and substitute
`attributes.` throughout. Nothing else in the queries changes.

## Queries

All Logs Insights. Time range: as much as is retained; the per-session
export should cover at least a few weeks to catch long sessions.

### 0. Discovery: shape and volume

```
fields @timestamp, @message
| sort @timestamp desc
| limit 3
```

```
stats count() as events,
      min(@timestamp) as first,
      max(@timestamp) as last,
      count_distinct(attributes.session.id) as sessions,
      count_distinct(attributes.app.version) as versions
  by attributes.event.name
```

### 1. The raw export (the one that matters)

Every request, prompt and tool call, with the identifying and free-text
attributes left out. `user.email`, `user.id`, `user.account_uuid`,
`organization.id`, `prompt`, `tool_parameters` and `error` are never
selected. `session.id` is kept because everything groups on it; hash it if
the ids should not leave the account (they are random UUIDs, so that is
optional).

```
filter attributes.event.name in ["claude_code.api_request", "claude_code.user_prompt", "claude_code.tool_result"]
| fields @timestamp as ts,
         attributes.event.name as event,
         attributes.session.id as session,
         attributes.app.version as version,
         attributes.terminal.type as terminal,
         attributes.model as model,
         attributes.speed as speed,
         attributes.input_tokens as input_tokens,
         attributes.output_tokens as output_tokens,
         attributes.cache_read_tokens as cache_read_tokens,
         attributes.cache_creation_tokens as cache_creation_tokens,
         attributes.duration_ms as duration_ms,
         attributes.cost_usd as cost_usd,
         attributes.prompt_length as prompt_length,
         attributes.tool_name as tool_name,
         attributes.success as success
| sort ts asc
| limit 10000
```

The console caps a result at 10 000 rows. If the range has more, narrow the
time window and export in pieces, or use the CLI script below, which pages
by day. Export as CSV or JSON; either is fine for the offline step.

### 2. Per-request token shape, by model

```
filter attributes.event.name = "claude_code.api_request"
| fields attributes.input_tokens + attributes.cache_read_tokens + attributes.cache_creation_tokens as context,
         attributes.output_tokens as output,
         attributes.cache_read_tokens as cached
| stats count() as requests,
        pct(context, 10) as ctx_p10, pct(context, 50) as ctx_p50,
        pct(context, 90) as ctx_p90, pct(context, 99) as ctx_p99, max(context) as ctx_max,
        pct(output, 10) as out_p10, pct(output, 50) as out_p50,
        pct(output, 90) as out_p90, pct(output, 99) as out_p99, max(output) as out_max,
        sum(context) as ctx_total, sum(cached) as cached_total, sum(output) as out_total,
        pct(attributes.duration_ms, 50) as ms_p50, pct(attributes.duration_ms, 90) as ms_p90
  by attributes.model
| sort requests desc
```

`cached_total / ctx_total` is the overall cache share.

### 3. Output-length histogram

Powers of two, because the interesting range runs from a 30-token tool call
to a 4 000-token file write.

```
filter attributes.event.name = "claude_code.api_request" and attributes.output_tokens > 0
| fields floor(log(attributes.output_tokens) / log(2)) as log2_bucket
| stats count() as requests,
        min(attributes.output_tokens) as from_tokens,
        max(attributes.output_tokens) as to_tokens
  by log2_bucket
| sort log2_bucket asc
```

### 4. Per-session table

One row per session: turns, length in minutes, where the context ended up.
Insights allows one `stats` per query, so the distribution over sessions is
computed offline from this table (export it too).

```
filter attributes.event.name = "claude_code.api_request"
| fields attributes.session.id as session,
         attributes.input_tokens + attributes.cache_read_tokens + attributes.cache_creation_tokens as context
| stats count() as requests,
        count_distinct(attributes.model) as models,
        min(@timestamp) as start,
        (max(@timestamp) - min(@timestamp)) / 60000 as minutes,
        pct(context, 50) as ctx_p50,
        max(context) as ctx_max,
        sum(context) as ctx_total,
        sum(attributes.cache_read_tokens) as cached_total,
        sum(attributes.output_tokens) as out_total,
        sum(attributes.cost_usd) as cost_usd
  by session
| sort requests desc
| limit 10000
```

### 5. Concurrency: busiest five minutes

```
filter attributes.event.name = "claude_code.api_request"
| stats count() as requests,
        count_distinct(attributes.session.id) as sessions,
        sum(attributes.duration_ms) / 300000 as mean_in_flight
  by bin(5m)
| sort requests desc
| limit 30
```

`mean_in_flight` is total request time over the bin length: the average
number of requests in progress at any instant, which is the concurrency
level a server would see. Overlap inside a session (parallel subagents) is
derived offline from the raw export.

### 6. Tool mix

```
filter attributes.event.name = "claude_code.tool_result"
| stats count() as calls,
        pct(attributes.duration_ms, 50) as ms_p50,
        pct(attributes.duration_ms, 90) as ms_p90,
        max(attributes.duration_ms) as ms_max
  by attributes.tool_name, attributes.success
| sort calls desc
```

### 7. Human prompts

```
filter attributes.event.name = "claude_code.user_prompt"
| stats count() as prompts,
        count_distinct(attributes.session.id) as sessions,
        pct(attributes.prompt_length, 50) as chars_p50,
        pct(attributes.prompt_length, 90) as chars_p90,
        max(attributes.prompt_length) as chars_max
```

Requests per human prompt come from the raw export: count `api_request`
rows between consecutive `user_prompt` rows in a session.

### 8. Errors and rate limits, for completeness

```
filter attributes.event.name = "claude_code.api_error"
| stats count() as errors, pct(attributes.duration_ms, 50) as ms_p50
  by attributes.model, attributes.status_code
| sort errors desc
```

## CLI export, paging by day

For more than 10 000 rows. Needs the AWS CLI with a profile that can run
`logs:StartQuery` and `logs:GetQueryResults` on the group. Writes one JSON
file per day of the raw export query; the offline step reads them all.

```sh
#!/usr/bin/env bash
# usage: cw-export.sh <log-group> <start YYYY-MM-DD> <end YYYY-MM-DD> <out-dir>
set -euo pipefail
group=$1; start=$2; end=$3; out=$4
mkdir -p "$out"
query=$(cat <<'EOF'
filter attributes.event.name in ["claude_code.api_request", "claude_code.user_prompt", "claude_code.tool_result"]
| fields @timestamp as ts, attributes.event.name as event, attributes.session.id as session,
         attributes.app.version as version, attributes.terminal.type as terminal,
         attributes.model as model, attributes.speed as speed,
         attributes.input_tokens as input_tokens, attributes.output_tokens as output_tokens,
         attributes.cache_read_tokens as cache_read_tokens, attributes.cache_creation_tokens as cache_creation_tokens,
         attributes.duration_ms as duration_ms, attributes.cost_usd as cost_usd,
         attributes.prompt_length as prompt_length, attributes.tool_name as tool_name, attributes.success as success
| sort ts asc
| limit 10000
EOF
)
day=$start
while [[ "$day" < "$end" ]]; do
  next=$(date -u -d "$day + 1 day" +%F)
  id=$(aws logs start-query --log-group-name "$group" \
         --start-time "$(date -u -d "$day" +%s)" --end-time "$(date -u -d "$next" +%s)" \
         --query-string "$query" --limit 10000 --query queryId --output text)
  while :; do
    status=$(aws logs get-query-results --query-id "$id" --query status --output text)
    [[ "$status" == Complete ]] && break
    [[ "$status" == Failed || "$status" == Cancelled || "$status" == Timeout ]] && { echo "$day: $status" >&2; break; }
    sleep 2
  done
  aws logs get-query-results --query-id "$id" --output json > "$out/$day.json"
  n=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['results']))" "$out/$day.json")
  echo "$day: $n rows"
  [[ "$n" == 10000 ]] && echo "$day: hit the 10000 cap, split this day" >&2
  day=$next
done
```

## From export to replay

Offline, a script (to be written: `usage.py`, host side, stdlib only)
reads the export and produces `docs/usage.json`: the distributions above,
plus a handful of *representative sessions* as turn lists — per turn, the
context size, the cache-read share and the output length. Picked as the
median session by request count and the p90 session, so the benchmark has
a typical and a heavy case.

`agentic.py loop` then gets `--profile docs/usage.json --session median|heavy`
and replays that shape instead of the fixed transcript: filler tokens to
reach each turn's context, `max_tokens` from that turn's output length,
warm and cold as today. Labels and dimensions stay consistent with the
existing `loops` entries so the four standard configs can be compared
against both the synthetic and the real shape.

## Results: the profile

The user ran the raw export against the work account's logs the same day:
three days (2026-09-04 to 06), 1160 rows, 518 API requests, 108 human
prompts, 534 tool calls. The `session.id` column came back empty (the
attribute is named differently in that pipeline), so sessions were split
on 30-minute gaps: six of them, two long ones (171 and 199 requests, two
and five and a half hours). `usage.py` reduced it to `docs/usage.json`; the
page shows it under "What real sessions look like".

| per request | p10 | p50 | p90 | p99 | max |
| --- | --- | --- | --- | --- | --- |
| context (input + cache read + cache create) | 38 033 | 105 344 | 223 456 | 267 315 | 274 585 |
| fresh prefill (input + cache create) | 398 | 1 154 | 6 611 | 117 360 | 207 111 |
| output | 146 | 520 | 2 235 | 8 751 | 14 207 |
| provider duration, s | 3.7 | 9.3 | 31.3 | 93.9 | 175 |

- **Context is the story.** The median request already carries 105K
  tokens; 71% of requests are over the 64K window every local
  configuration runs at, and 92% are over the dense 27B's 32K. The context
  histogram is flat from 40K to 280K: sessions grow until compaction, which
  shows up as a drop of 50–99K, seven to thirteen times in a long session.
  The synthetic loop's 20K endpoint is where real sessions *start* (one
  fresh session went 905 → 44K in twelve requests).
- **Fresh prefill per turn matches the synthetic guess.** Median 1.15K
  tokens, which is the 1.2K the loop adds per turn. The tail is the
  compactions and session starts: p99 117K, 13 of 518 requests cold.
  Overall 96.1% of context tokens came from the provider's prefix cache.
- **Output is 4× the synthetic 128.** Median 520, p90 2.2K, p99 8.7K, and
  the distribution is wide: 256–1023 is the mode (266 of 518), but 65
  requests emitted 2K–14K (file writes, long answers). The sweeps' fixed
  256 is the p25.
- **Cadence.** Median gap between one request's end and the next one's
  start is 3.2 s (tool time); p90 128 s (the human). 43 of 512 consecutive
  pairs overlap: parallel subagents or background tasks, 8% of turns. Time
  in flight per session is 10–30% of wall time.
- **A human prompt costs 3 model calls at the median, 11 at p90, 26 at
  most.** 19 of 108 prompts needed exactly one; 36 needed six or more.
- **Tools.** Bash is 76% of calls (389 ok, 15 failed), p50 0.4 s, p90 23 s,
  max 2 min. Edit and Write are instant; Read 16 calls. The rest is
  AskUserQuestion, Skill, ToolSearch, one WebSearch.
- **Provider output speed**, as seen from the client (output tokens over
  request duration, so it includes prefill and queueing): median 61 tok/s,
  p90 87. The local MoE decodes at 260–290 tok/s single-stream; the dense
  27B at 25–50.

The replay sessions: *typical* is the 35-request, 18-minute session
(context 919 → 66K, two resets, 20.5K output tokens); *heavy* is the
199-request, 5.5-hour one (65K → 275K → 47K, ten resets, 215K output
tokens). At a 64K window 33 of the typical session's turns fit and 55 of
the heavy one's; at 256K, all but 11 of the heavy session's.

## Results: the replay

First trial on the 64K MoE, typical session, one agent, before the
over-window fix: 24 of 33 turns ran; the 25th needed 63.3K + 2.2K and got
HTTP 400, and the replay stopped. Now such a turn is skipped and counted.
Measured contexts track the targets within 1% after the first turn (the
first is short by 12% because the filler is calibrated from that response).

Warm, the first turn (a 50K cold prefill) took 2.9 s; turns 2–8 at
59–64K took 0.23–0.41 s each, the prefix cache hit 83.7%. Cold, every turn
at 60K took 3.4–4.1 s: 5.7K prefill tokens per second. Decode at 60K of
context ran at 261–266 tok/s against 290 at 2K.

The full chain ran the same day: five configurations × typical and heavy
sessions, the MoE also at its native 262 144 window (`MAX_MODEL_LEN=262144
./run.sh -p qwen3.6-35b-a3b`; the KV pool of 370K tokens holds 1.4 such
sessions). Entries are in `docs/results.json` under `loops` with `shape` =
typical / heavy, and on the page under the loop section's Shape chips.

### Typical session (35 turns, context 0.9K → 66K, two resets), one agent

| configuration | window | turns in | warm TTFT last / mean / p95 | cache hit | cold TTFT last / mean / p95 | decode tok/s | warm turns/min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.6-35b-a3b | 64K | 32 of 35 | 0.28 / 0.77 / 2.9 s | 81% | 3.3 / 3.2 / 4.1 s | 267 | 20.7 |
| qwen3.6-35b-a3b 256K | 256K | 35 of 35 | 0.29 / 0.54 / 2.3 s | 89% | 4.5 / 3.4 / 4.4 s | 266 | 21.7 |
| nemotron-3.5-lightning | 64K | 32 of 35 | 0.18 / 0.72 / 2.8 s | 81% | 3.3 / 3.2 / 4.0 s | 386 | 27.2 |
| qwen3.8-27b text-only, graphs | 32K | 2 of 35 | — | — | — | 174 | — |
| qwen3.8-27b default | 32K | 2 of 35 | — | — | — | 93 | — |

Four agents in parallel on the same session (each its own copy): MoE warm
last-turn TTFT 1.16 s, p95 9.0 s, 37.6 turns/min across the four, decode
152 tok/s per stream; Nemotron 0.17 s, p95 8.7 s, 48.3 turns/min, 181 per
stream. Cold with four agents is 3.3–5.2 s per turn at the end and 55 tok/s
per stream on the MoE: four 60K prefills share the card.

### Heavy session (199 turns, context 0.9K → 275K, ten resets), one agent

| configuration | window | turns in | warm TTFT last / mean / p95 / max | cache hit | cold mean / p95 / max | warm wall | cold wall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.6-35b-a3b 256K | 256K | 188 of 199 | 0.51 / 0.86 / 2.3 / 13.4 s | 94% | 11.2 / 27.1 / 42.2 s | 17.3 min | 49.9 min |
| qwen3.6-35b-a3b | 64K | 55 of 199 | 0.52 / 0.62 / 2.4 / 3.4 s | 80% | 2.4 / 3.5 / 4.1 s | 2.9 min | 4.6 min |
| nemotron-3.5-lightning | 64K | 55 of 199 | 0.34 / 0.56 / 2.3 / 3.3 s | 80% | 2.4 / 3.5 / 4.0 s | 2.2 min | 3.9 min |
| qwen3.8-27b (either) | 32K | 5 of 199 | — | — | — | — | — |

The provider's own API time for these sessions, from the telemetry
(`duration_ms` summed): typical 5.5 min of a 18.5-minute session (30% of
wall), heavy 47 min of 335 (14%). The local MoE at 256K replays the typical
session's model turns in 1.6 min warm and the heavy one in 17 min warm,
against 5.5 and 47 min at the provider, with the same context and output
sizes. That is not a like-for-like model comparison (nothing here scores the
answers), but it says the serving side is not the bottleneck: the local
model would have spent 9% and 5% of those sessions' wall time.

### What the replay says

- **The window is the finding.** At 64K the typical session loses 3 of 35
  turns and the heavy one 144 of 199; at 32K the dense 27B configurations
  serve only the tiny Haiku-class turns (2 and 5). The MoE at 256K serves
  everything but the eleven turns above 262K, and the heavy session runs
  warm at a 94% cache hit, 0.5 s to first token at the end and a 2.3 s p95.
  Its worst warm turn is 13.4 s, turn 85, where the real session's context
  jumped from 37K to 147K with only 388 fresh tokens (a cached transcript
  came back at the provider); in the replay that is a 110K prefill. Cold,
  the session is 11 s per turn on average and 42 s at 260K, which is what
  serving without a prefix cache (or with one that evicts) costs.
- **Warm TTFT is flat in context size; the rebuilds are the spikes.** Every
  configuration's warm curve sits at 0.2–0.5 s across 40–70K of context and
  jumps to 2–3.5 s on the turn after each reset (the typical session resets
  at turns 9 and 16, so turns 10 and 17 rebuild 45K of context cold), and on
  the 64K servers also after every skipped over-window turn (26 and 34),
  because the replay starts a fresh conversation there too. The synthetic
  loop's smoothly rising TTFT never showed this shape.
- **Nemotron leads on decode and warm latency, ties on prefill.** 386 vs
  267 tok/s single-stream at 56K context, 0.18 vs 0.28 s warm TTFT; cold
  turns are the same 3.2 s on both, so prefill throughput (~5.7K tok/s at
  60K) is the same for the two A3B models.
- **Decode holds up with context.** The MoE decodes at 265 tok/s at 60K and
  247 tok/s averaged over the heavy session (contexts to 275K), against 290
  at 2K.
- **Four agents cost 4× on the reset turns and little elsewhere.** Warm
  last-turn TTFT goes 0.28 → 1.16 s (MoE) and 0.18 → 0.17 s (Nemotron); the
  p95 is the reset turns at 8–9 s where four 60K prefills queue.
- **The 64K entries' over-window counts include one runtime skip each**: a
  turn the plan thought would fit, whose measured context plus output came
  to 65 537 tokens on the local tokenizer.

Trial numbers before the chain (24-turn partial run) are superseded by the
table above.
