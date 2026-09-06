# vllm-5090

Tooling I use for running open models on a single RTX 5090 and various
benchmarks that might or might not be useful in deciding which to use and when.

Two models I am interested in at the moment:

- Qwen3.6-35B-A3B
- Qwen3.8-27B

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
| `docs/` | [Browsable benchmark results](https://rce.github.io/vllm-5090/) — `docs/results.json` is the source of truth |
| `agent-notes/` | The deep dive as written by the infinitely persistent LLMs I use to setup the configuration and benchmarks, flag rationales, dead ends, what was measured and how |
