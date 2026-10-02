# Decision Lab

Four games for inspecting local decision models: **Tetris, 2048, Connect Four, and BlockStar**. The browser shows actual recorded model choices, their measured inference time, and reproducible game outcomes. The two new browser games, 2048 and Connect Four, are also playable by a human.

[Open the GitHub Pages site](https://rayking99.github.io/decmods/) · [BlockStar source repository](https://github.com/rayking99/BlockStar) · [Tetris experiment and runtime notes](tetris/README.md)

## Preview the site

```sh
python3 -m http.server 8770 --bind 127.0.0.1 --directory docs
```

Open <http://127.0.0.1:8770/>. This site needs no build step, external JavaScript packages, account, API key, or model download. GitHub Pages serves `docs/` from the `master` branch. Published replays use saved inference; hosted Pages does not run models or connect to a visitor's local model services. Human play and recorded model play are labelled separately.

## New game benchmark

The frozen [suite](arcade/suite.json) contains eight states for each new environment, evaluated in two deterministic option orders. The benchmark runs one further game per model and environment, with seed 42 and caps of 24 player decisions in 2048, 12 in Connect Four, and 16 slides in the small BlockStar replay puzzle. Tetris retains its earlier, separately timed benchmark of 32 boards in two orders and two capped games per model.

- **2048:** ordinary merges and seeded random spawns. Rank immediate merge score, then empty cells, then maximum tile, before spawning. The model does not see or select a future random tile.
- **Connect Four:** rank a four-ply minimax evaluation after the player's proposed move. The deterministic opponent searches its reply plus three additional plies. Derived search scores are supplied in the options, so agreement tests policy following rather than unaided strategic reasoning.
- **BlockStar:** reproduce the original rigid rectangles, board, target, traversal order, every slide distance, and swept collisions. Rank declared ordered-progress metrics. One of eight frozen states is the original 20×18 layout; seven are smaller fixtures. Full-board probe results and timing are reported separately. The capped model replay uses a small, solvable five-block fixture.

Every legal action reaches the native decision model. A shared 20-option group tournament accommodates Julia's native limit and the original BlockStar board's 183 legal slides. Winners advance through further model calls; no application heuristic prunes options or chooses a fallback. Tournament order can influence results and is not equivalent to a single global decision. Every policy tie is accepted.

Principal median and P95 timings use the first option-order pass, excluding warmups, states identical to a warmup, and model loading, and include all model calls for each decision. Repeated inputs are reported separately because native caches affect timing; even a first-order pass can reuse options or prefixes. Inference is serialized on one Apple M3 Max with 128 GB unified memory. Native heads, precisions, and prompt layouts differ. These small samples are descriptive, not universal model rankings.

The tested variants are Nimble 9B, Tev1 4B and 0.8B, Winnow E4B, Julia-1, Laya English and multilingual, Kev 4B on MLX and MPS, CLM 8B, Lev 4B, and Clef Flash 9B on MLX 4-bit. The exact installed checkpoints and runtime metadata are retained in the results. Jev-Omni's upstream loader requires CUDA, so no Mac result or substitute family is claimed.

For the already prepared local runtime assets:

```sh
tetris/.venv/bin/python arcade/run_models.py
python3 arcade/finalize.py
```

The runner resumes completed variants. Downloads and environment setup follow the detailed [Tetris runtime notes](tetris/README.md); ignored model weights and upstream environments are not included in this repository. `--models nimble tev1:0.8b` limits a run to those explicit aliases.

## Clef Flash MLX

The requested [MLX 4-bit conversion](https://huggingface.co/mlx-community/clef-flash-4bit) is pinned to `d9ec324f7992383bdfb7a0b4eed8b4b9d10f81be`. Its 9B backbone uses affine 4-bit quantization with group size 64; the vision encoder and original joint schema head remain BF16. The downloaded Safetensors total 6.194 GB. The head's SHA-256 exactly matches the pinned Cloudflare release. [Source provenance](tetris/hf/research/clef-flash-mlx-source.json) and [native smoke evidence](tetris/hf/research/clef-mlx-smoke.json) capture the checks.

The bundled `clef_mlx.py` runs the backbone and decision head together. Ordinary chat generation does not expose that head. Before loading, our loopback server verifies the backbone/head shards, tokenizer, configuration and loader against the archived pinned manifest. It preserves native choice, score and boolean responses, uses Metal, and sets `truncate=False` with a 16,384-token maximum. Oversized requests are rejected. Native confidence is the maximum softmax probability; no adapter temperature or concentration transform is added. Clef sorts choice IDs before encoding, so reordering an identical option set does not change its encoded positions; tournament group membership can still differ, and native exact-probability ties use request order. The inspected loader has no persistent request or prefix cache.

Clef is measured in a subsequent serialized run on the same hardware and frozen cases, with each environment's original shared tournament protocol. Earlier model traces are retained. Its Tetris principal latency uses the first option-order pass, with repeated-pass timing reported separately. These game tests use text/JSON state; they do not measure image or video quality.

```sh
uv sync --project tetris/hf/clef_runtime --locked
tetris/hf/clef_runtime/.venv/bin/python tetris/hf/clef_runtime/download.py
tetris/hf/clef_runtime/.venv/bin/python tetris/hf/clef_runtime/server.py --port 11444
```

With the native server running, capture `/v1/models` to a metadata JSON, then run `tetris/hf/run_clef_benchmark.py --metadata <path>` for Tetris. `arcade/run_models.py --models clef-flash:mlx-4bit` and its `--updated-blockstar` variant run the other frozen suites. Each recorded response retains the exact revision, native loader/head hashes, quantization and locked dependency versions.

## BlockStar reference comparison

[Provenance and baseline traces](arcade/blockstar-source.json) pin `rayking99/BlockStar` at commit `f40b338af841bcbcbe402770e5e07610b3397c14`. An isolated execution of its original stochastic driver, with a single run per seeded subprocess and output writing disabled, solved the complete original board in all three measured runs: 103, 107, and 140 registered slides, taking 2.913–3.812 seconds. Every registered move was independently replayed to the exact target. This is a three-run source baseline, not a performance distribution or a trained model.

The small five-block replay has a BFS-proven minimum of seven slides; the declared progress rule solves it in nine. Model replay outcomes can be compared with these small-puzzle baselines. Full original-board model trials measure two one-step policy decisions, not a complete solve, so their timing and agreement must not be treated as equivalent to the original driver's full-puzzle outcomes.

No pretrained AgentFormer checkpoint is present in the inspected source. No retraining, native CUDA benchmark, or complete original-puzzle solve by a decision model is claimed. The adapter independently implements movement rules and layout data; upstream source and `results.txt` are not vendored.

## Updated BlockStar integration

The modern source is separately pinned to [`fed370f`](https://github.com/rayking99/BlockStar/commit/fed370f11044b56a8fabd449316cdfe0ca12ad6f). It adds import-safe discovery and replay validation, whole-slide search, order discovery, fixed terrain, connected rigid shapes, and state-conditioned, legally masked imitation training. [Updated provenance](arcade/blockstar-update-source.json) records the source, official layouts, measured native discovery runs, and full solution traces.

The original layout's 183 legal slides remain equivalent. The updated large yard is 28×30 with 24 movable blocks, 48 fixed terrain cells, and 393 initial legal slides. Our independent adapter verifies every initial endpoint against the source referee. Fixed `#` cells cannot move or be crossed.

The supplemental [frozen suite](arcade/updated/suite.json) tests both official initial layouts in two option orders through all twelve runtimes. It uses a declared sorted-label goal-prefix policy; the upstream solver discovers its own order and imposes no mandatory target order. These are four one-step probes per model, with a distinct warmup and separate first/repeated-order timings. They do not measure a complete model solve or provide a latency distribution.

Fresh native discovery found a 37-slide solution for the original layout in 29.890 seconds from 228 rollouts, and a 64-slide large-yard solution in 45.503 seconds from 200 rollouts. Both reach the exact target and are replayable on the site. Discovery time and certificate replay time are reported separately. The original solution reproduces the published champion's slide count with a different legal move ordering; the large-yard sequence also matches its published champion. Neither result proves global optimality.

```sh
tetris/.venv/bin/python arcade/run_models.py --updated-blockstar
python3 arcade/verify_update.py
python3 arcade/build_site_data.py
```

The [complete supplemental evidence](docs/data/blockstar-update-results.json.gz) preserves every native request, answer, probability, option, and resulting board with lossless compression. Earlier Tetris and game measurements retain their original protocols and source pins.

## Validation and evidence

```sh
python3 -m unittest discover -s arcade -v
node --test docs/assets/game-engines.test.mjs
python3 arcade/verify.py
python3 arcade/verify_update.py
cd tetris
uv run python -m unittest -v
uv run python verify_benchmark.py --results benchmarks/huggingface/results.json --baseline benchmarks/results.json
```

The audit recomputes every oracle label and summary, verifies paired inputs and complete tournament coverage, matches selected actions to native responses, and reconstructs saved boards, deterministic opponent replies, and seeded spawns. [Compact public evidence](docs/data/) omits duplicate API payloads and replay RNG state. [Complete new raw traces](docs/data/arcade-results.json.gz) are published with lossless gzip compression to fit GitHub's file-size limit; benchmark and audit scripts read that archive automatically on a fresh checkout. The local working `arcade/results.json` is ignored. The original Tetris traces and audits are retained under `tetris/benchmarks/`.

The site uses relative asset paths for GitHub Pages project URLs, responsive layouts, semantic controls, keyboard play, touch controls, and reduced-motion support.
