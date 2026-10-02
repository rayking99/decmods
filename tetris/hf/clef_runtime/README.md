# Clef Flash on MLX

This isolated Apple Silicon environment serves `mlx-community/clef-flash-4bit` through its bundled joint schema head. It uses the pinned 9B Flash release, with a 4-bit affine backbone and BF16 vision encoder/head. The original Cloudflare head was verified byte-for-byte. Before loading, the server verifies all backbone/head shards, tokenizer, configuration and native loader against the archived pinned Hugging Face manifest, and rejects missing, changed or additional inference files. The MLX loader is also checked against its pinned SHA-256 before import.

```sh
uv sync --project tetris/hf/clef_runtime --locked
tetris/hf/clef_runtime/.venv/bin/python tetris/hf/clef_runtime/download.py
tetris/hf/clef_runtime/.venv/bin/python tetris/hf/clef_runtime/server.py
```

The server binds to `127.0.0.1:11444`. `GET /v1/models` returns source and runtime metadata. `POST /v1/systemone` accepts the explicit alias `clef-flash:mlx-4bit`, a text/JSON `state`, and native typed `questions`. Inference is serialized and uses Metal. Oversized input is rejected at 16,384 tokens; no state truncation or model substitution occurs. The HTTP benchmark transport uses text/JSON; native PIL image/video inputs are outside this transport's tested scope.

Capture readiness metadata and run the unchanged game protocols:

```sh
curl --fail http://127.0.0.1:11444/v1/models > tetris/hf/research/clef-mlx-runtime.json
tetris/.venv/bin/python tetris/hf/run_clef_benchmark.py --metadata tetris/hf/research/clef-mlx-runtime.json
tetris/.venv/bin/python arcade/run_models.py --models clef-flash:mlx-4bit
tetris/.venv/bin/python arcade/run_models.py --models clef-flash:mlx-4bit --updated-blockstar
```

The model's native confidence is maximum softmax probability. Choice IDs are sorted internally, so permutations of identical option sets have identical encoded positions, while tournament membership can vary and exact probability ties are resolved in request order. The native loader implements no persistent inference cache. Ordinary `mlx_lm.generate` / `mlx_vlm.generate` do not execute the trained decision head.

Weights and virtual environments are ignored by Git. The lockfile, adapter, frozen inputs, source metadata, raw responses, audits, and published replays are retained. See the [conversion card](https://huggingface.co/mlx-community/clef-flash-4bit) and [pinned source record](../research/clef-flash-mlx-source.json).
