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

## Results

Not yet run.
