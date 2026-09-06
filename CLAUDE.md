Basic project structure

```
Containerfile            vLLM is run in a container to easily pin versions etc
profiles/                One env file per model configuration, fed to the container by run.sh
bench.py                 Synthetic concurrency sweep against a running server
quality.py               Divergence probe: does a flag change what the model says? (docs/divergence.md)
quality/prompts.jsonl    The fixed prompt set the probe scores; captures land in quality/captures/ (untracked)
agentic.py               Tool-call smoke checks and a replayed agent-loop benchmark (warm vs cold prefix cache)
docs/                    Publicly viewable interactive page for browsing benchmark results
docs/results.json        The actual benchmark results
README.md                Human written document presenting the project and repository
agent-notes/             The location where agents can and should write their notes and plans as markdown files
```

## README.md vs agent-notes/

These have different audiences, and the split matters more than it looks.

`README.md` is **written by a human, for humans**. It says what this project is
for and what it is trying to become. It is meant to be skimmed by someone who
has just landed here. LLM-written prose tends to be dense, evenly-weighted and
exhausting to glance at — it explains everything at the same volume and buries
the point. Keep that out of the README.

`agent-notes/` is where the dense material belongs: findings, reference tables,
flag rationales, dead ends, what was measured and how. Write as much as is
useful there. It is for the deeper dive, and for the next agent picking the work
up.

So: if you have written something long, correct, and reference-shaped, it goes
in `agent-notes/`. Do not migrate it into the README, and do not rewrite the
README to sound like it. When README content genuinely needs to change, prefer
raising it rather than rewriting the human's voice.

## Benchmarks

`bench.py` is the standard measurement, so numbers stay comparable between
configurations. Do not hand-roll a one-off timing script and report from it;
run the sweep and merge into `docs/results.json`. Anything ad-hoc is a finding
for `agent-notes/`, not a benchmark result.

`quality.py` is the equivalent for "is it still the same model". Run
`quality.py jitter` on a configuration before trusting any comparison against
it: the MoE servers are not deterministic run to run, and a comparison at the
noise floor means nothing. Comparison summaries merge into `docs/results.json`
the same way sweeps do; the raw captures do not go in git.

`agentic.py` covers the agent-shaped questions: `tools` checks that structured
tool calls survive the server's parsers, `loop` replays a growing coding-agent
transcript and reports per-turn TTFT with and without prefix caching. Both merge
into `docs/results.json` (`toolcalls`, `loops`).
