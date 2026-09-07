Basic project structure

```
Containerfile            vLLM is run in a container to easily pin versions etc
profiles/                One env file per model configuration, fed to the container by run.sh
bench.py                 Synthetic concurrency sweep against a running server
quality.py               Divergence probe: does a flag change what the model says? (docs/divergence.md)
quality/prompts.jsonl    The fixed prompt set the probe scores; captures land in quality/captures/ (untracked)
agentic.py               Tool-call smoke checks and a replayed agent-loop benchmark (warm vs cold prefix cache)
usage.py                 Reduces Claude Code telemetry exports to docs/usage.json, the real-session shape agentic.py can replay
Containerfile.video      diffusers on top of the vLLM image, for the video generation side
video.sh / video.py      Video generation length/speed sweep: how long a clip costs, how long a clip fits
docs/                    Publicly viewable interactive pages for browsing benchmark results
docs/index.html          The LLM serving page; docs/video.html is the video generation page
docs/results.json        The actual LLM benchmark results
docs/usage.json          Usage profile from real Claude Code sessions (counts only), shown on the page and replayed by agentic.py
docs/video.json          The video generation results, same idea, separate file and page
docs/style.css, charts.js  Shared by both pages: the stylesheet, the tooltip and the line chart
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
into `docs/results.json` (`toolcalls`, `loops`). `loop --profile docs/usage.json`
replays a real session from the usage profile instead of the synthetic
transcript; `usage.py` builds that profile from a Claude Code telemetry export
(the queries are in `agent-notes/2026-09-07-claude-code-usage-profile.md`).
Loop entries carry a `shape` dimension (synthetic, typical, heavy) so the two
kinds stay apart on the page.

`video.py` is the same idea for video generation: one prompt, one seed, clips
of growing length until VRAM runs out, reporting wall time, per-step time,
decode time and peak VRAM per clip. It runs in its own container
(`Containerfile.video`, launched by `video.sh`) and merges into
`docs/video.json` (`videos`), which `docs/video.html` shows; the video side is
kept apart from the LLM results on purpose. The clips land in `video/out/`
(untracked); nothing scores them.
