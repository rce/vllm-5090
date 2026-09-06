# vllm-5090

Running open models locally on a single RTX 5090, and measuring what that
actually gets you.

> **TODO (human):** what this is for, and where it's going.

Two models are set up and benchmarked so far — Qwen3.6-35B-A3B and
Qwen3.8-27B — both NVFP4, both served by vLLM in a pinned container.

## Getting a model running

```sh
podman build -t vllm-5090 .
hf download nvidia/Qwen3.6-35B-A3B-NVFP4
./run.sh -p qwen3.6-35b-a3b
```

That gives an OpenAI-compatible API on `:8000`. `./run.sh` with no arguments
serves Qwen3.8-27B instead.

## Measuring it

```sh
./bench.py --label "my config" --out docs/results.json
```

A synthetic concurrency sweep — 1, 2, 4, 8, 16 simultaneous requests — that
reports time to first token, per-stream decode rate and total throughput, and
records whether the configuration works at each level at all.

## Where things are

| | |
| --- | --- |
| `profiles/` | One env file per model configuration |
| `docs/` | Browsable benchmark results — `docs/results.json` is the source of truth |
| `agent-notes/` | The deep dive: findings, flag rationales, dead ends, what was measured and how |

If you want to know *why* a particular flag is set, or which configurations
failed and how, `agent-notes/operating-the-stack.md` is the place to start.
