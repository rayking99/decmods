# Nimble plays Tetris

A local, visible experiment in making a decision model control a game. Python implements the board and physics; Ollama's **Nimble 9B** selects the placements through `POST /v1/systemone`. The browser displays the real board, the selected landing, actual model probabilities, per-move latency, and the decision history.

```sh
cd /Users/jso/code/documentation/tetris
ollama pull nimble
uv run python app.py --autostart
```

Open [the local game](http://127.0.0.1:8765). Ollama must be running; version 0.35 or newer is required. Use **Pause**, **One piece**, and **Restart** to inspect play. Restart repeats the same seven-bag sequence. There are no Python dependencies; uv manages the Python environment and lockfile.

```sh
uv run python app.py --seed 123 --delay 0.8 --max-pieces 500
uv run python -m unittest -v
uv run python verify_replay.py
```

## What the model decides

The engine enumerates every legal rotation and horizontal position, hard-drops the piece, clears completed rows, and computes the resulting holes, column heights, and roughness. These are observations, with no combined heuristic score or application-selected winner. The supplied policy asks Nimble to avoid holes, clear rows, and keep the stack low and compact. Nimble chooses which landing to execute.

The API allows at most 26 options. Pieces with more placements use two shuffled groups and a final decision between the model-selected group winners. Every landing reaches the model; none is pruned by a heuristic. This grouping can influence the result and is not equivalent to scoring every option in one global choice. When only one legal landing exists, it is explicitly logged as a forced move. Invalid responses stop play; there is no heuristic fallback.

This is **placement-mode Tetris**: the piece can rotate and shift at the top before a straight hard drop. There is no hold, soft drop, wall kick, sliding beneath overhangs, or continuous gravity. The clock waits for inference. It tests board decisions rather than frame-by-frame keyboard control. The score is 100/300/500/800 for one/two/three/four rows, without levels or drop bonuses.

## Evidence and replay

Every run is saved incrementally in `runs/<run-id>.json`. **Save replay** also downloads the current record. It includes model metadata and digest, the seed, every complete API request and raw response, decision latency, the executed move, and the board after each move. Latency includes all model calls for a placement and any cold loading; the display delay is excluded. Final-round probabilities are displayed for tournaments; probabilities from earlier groups remain in the replay. Ollama's `confidence` is presented as preference concentration, not calibrated correctness.

`verify_replay.py` reconstructs the latest run (or a supplied JSON path), checks every option's observations, proves that all legal landings reached the model, compares executed moves with the native API answers, and checks every saved board. Repeated requests can benefit from Ollama's prompt cache; latency depends on hardware and game state.

Nimble was selected because it leads Ollama's published comparison of its local decision models (75.7%, versus Tev1 4B's 73.3% and Tev1 0.8B's 63.5%). This is a publisher classification benchmark, **not evidence of being the best Tetris player**. The actual downloaded digest is stored in each run, because model tags may change.

Sources checked 1 October 2026: [Nimble model and evaluation](https://ollama.com/library/nimble:latest), [decision API](https://docs.ollama.com/api/systemone), [Ollama decision-model announcement](https://ollama.com/blog/ollama-now-supports-jev-style-decision-models).

## Completed local validation

On 1 October 2026, this demo ran with real Nimble inference on an Apple M3 Max with 128 GB memory and Ollama 0.35.0. All 14 physics and adapter tests passed. Browser verification covered visible live placements and row clears, pause, exactly one single-step placement, restart, resume, and absence of browser errors. The live game remains available while the server is running.

The captured [replay](evidence/replay.json) and its [verification result](evidence/verification.json) preserve an actual model-controlled run. The verifier reconstructed every captured board and matched every executed placement to the raw model answer. [The screenshot](evidence/live-preview.jpg) shows the live viewer. This proves the integration works; it is not a comparison against other Tetris policies or a long-run performance benchmark.

## Comparing all Ollama decision models

`benchmark.py` compares Nimble 9B, Tev1 4B, and Tev1 0.8B through the same native decision API. Install all three models, then run:

```sh
ollama pull tev1:4b
ollama pull tev1:0.8b
uv run python benchmark.py
```

The benchmark freezes 32 boards: 12 from observed model play, 12 from independent reference-policy trajectories, and 8 controlled challenges. Each board is tested with two deterministic option orders, giving 64 paired trials per model. All models receive the same compact observations and the same explicit priority rule. Above 26 placements, all models use the same group tournament. The benchmark also runs two games per model with seeds 42 and 123 and a 40-piece cap per game.

**Policy agreement** accepts every placement tied for the best one-step priority tuple: minimum holes, maximum lines cleared, minimum peak height, minimum total height, then minimum roughness. This measures following the supplied decision policy; it is not an oracle for optimal long-term Tetris. The benchmark records Pareto dominance separately. Game lines cleared and survival are separate outcome measures.

Timings report the entire placement decision, including all model calls, after an excluded cold-start warmup. Inference is serialized, the live demo is paused, and there is no animation delay. Prompt-cache effects remain possible. One opening position is probed with the original verbose demo format; this is not a full game validation of that format. The compact benchmark is shared across models to fit Tev1's smaller context. The publisher's 13-dataset classification scores are shown separately from the Tetris results.

The frozen suite, raw model traces, digests, summaries, and complete game trajectories are saved in `benchmarks/`. Resume an interrupted run with `uv run python benchmark.py --resume`. To prepare a new board suite, use `--new-suite`; the saved hash identifies the exact inputs used.

To show the live comparison dashboard, run these in separate terminals:

```sh
uv run python benchmark_summary.py
uv run python -m http.server 8766 --bind 127.0.0.1 --directory benchmarks
```

Open [the comparison dashboard](http://127.0.0.1:8766/). It polls the small summary rather than the complete raw traces.

### Measured results, 1 October 2026

| Model | Median placement decision | P95 | Policy agreement | Lines, seed 42 | Lines, seed 123 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nimble 9B | 2.021 s | 4.786 s | 46.9% | 8 / 40 pieces | 12 / 40 pieces |
| Tev1 4B | 1.505 s | 3.705 s | 39.1% | 6 / 40 pieces | 7 / 40 pieces |
| Tev1 0.8B | 0.315 s | 0.893 s | 18.8% | 0 / 29 pieces, game over | 1 / 35 pieces, game over |

All 192 fixed-board trials returned valid responses. Nimble had the highest supplied-policy agreement and the most lines cleared in these capped games. Tev1 0.8B was about 6.4 times faster by median placement latency but topped out in both games. Tev1 4B was about 1.34 times faster than Nimble, with lower agreement and fewer cleared lines. These are descriptive results from a small local experiment with this shared compact prompt, not universal model rankings.

`uv run python verify_benchmark.py` audited the paired inputs, recomputed policy labels and summaries, checked model-selected actions against native responses, and reconstructed all 224 game moves. The successful audit is saved in [benchmarks/verification.json](benchmarks/verification.json). The full raw result includes the exact model digests and every request and response. The original-format opening-board probe was accepted by all three models.

## Hugging Face and native Apple GPU runtimes

The extended [comparison and recorded games](http://127.0.0.1:8766/huggingface/) use the **unchanged frozen suite, compact observations, priority rule, option permutations, tournament, game seeds, and 40-piece caps**. The earlier Ollama results are retained as a separately timed matched baseline. The shared API inputs are identical; each author's native token layout and precision differ. Loading and downloads are excluded from placement latency.

The [captured trending list](hf/research/trending-top100.json), saved at 2026-10-01 04:57:44 UTC, contains all 100 models, not just search matches. The requested models ranked Laya 2, CLM 5, Julia-1 11, Jev-Omni 22, Lev 48, and Kev-4B 90. Winnow-E4B was outside this snapshot. These ranks are time-dependent. Cards and model commit metadata are saved in `hf/research/`; [native code commits](hf/research/native-sources.json) identify the exact runtimes used.

**Ollaya is a useful shared decision API**, rather than an ordinary text-generation wrapper: [its API](https://ollaya.dev/docs/api) supports TypeSafe's `/v1/systemone` schema. Its registry includes Winnow, Laya, Kev and CLM, but not all of the requested families. This experiment uses Ollaya 0.8.0 for Winnow's author-published Q8 GGUF on **Metal**. The other evaluated models use their native Python heads with Hugging Face weights. [The local gateway](hf/gateway.py) exposes one endpoint, `http://127.0.0.1:11440/v1/systemone`, and records the backend model/endpoint in every proxied response. It never substitutes a different family.

| Family | Runtime and limits assessed |
| --- | --- |
| [Winnow-E4B](https://huggingface.co/EldanRing/Winnow-E4B) | Author merged Q8 GGUF, Ollaya/llama.cpp Metal. Downloaded weights pinned and SHA256-verified by Ollaya. |
| [Laya](https://huggingface.co/convaiinnovations/laya) | Native Safetensors SDK, PyTorch MPS, English and multilingual checkpoints. Extended `max_len=8192`, `head_max_len=2048`; options exceeding 48 tokens or any dropped input are rejected. English was trained with a 512-token default; multilingual explicitly supports extended context. The current English runtime clamps an out-of-range shipped calibration temperature; accuracy is argmax, and confidence is not evaluated. |
| [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b) | Exact main checkpoint `139fdd94`, pinned Qwen3.5-4B-Base `1001bb4d`, trained LoRA and pointer head. Native BF16 PyTorch MPS and BF16 MLX backends, both with the author's serving prefix cache. The MPS path uses reference DeltaNet/convolution implementations; these are slow, and do not move inference to the CPU. No substitution of the older Qwen3 release. |
| [CLM v0.1 8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) | Qwen3-8B Safetensors on MPS with a last-token/L2 pooling adapter, plus the **original trained state/action projection heads and native schema/scoring**. BF16 encoder, FP32 heads; 2048-token limit rejected rather than truncated. Encoder embeddings are cached, so fresh/repeated timing is reported separately. This is a Python embedding backend rather than the author's CUDA/vLLM server. |
| [Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) | Native FP32 model and strict encoding, in its own Transformers 5.0 environment. Its original batch packer transfers only CUDA batches; the adapter adds MPS placement of those unchanged batches. Native 2–20 choices, 48 tokens per option, 8k context and a 2048-token head budget. CPU/MPS probability parity is checked on four real inputs. |
| [Lev](https://huggingface.co/interfaze-ai/lev) | Exact published LoRA, matching instruct base, calibration and trained head, through its native engine. The author loader targets CUDA/CPU; the adapter places the unchanged model/head on MPS. BF16, no CPU fallback; native option-order averaging retained. |
| [Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni) | The published Python loader explicitly rejects devices other than CUDA. Its source/configuration were inspected; its large checkpoint was not downloaded or substituted with a generic GGUF chat model. No Mac timing or accuracy is claimed. |

`hf/pyproject.toml` and `hf/uv.lock` contain the shared Python environment. Julia has a separate lock in `hf/julia_runtime/`; Kev uses its own upstream uv lock and `serve` extra, preserving its PyTorch 2.8/Transformers 5.17 requirements. The other Python model adapters use PyTorch 2.14.1. Model/server metadata records actual devices and versions. We verified a real MPS tensor operation before loading models.

The local runtime assets are in ignored `hf/vendor/`, `hf/models/`, `hf/cache/`, and `hf/runtime/` folders. Source repositories and checkpoints are pinned in the evidence; a fresh checkout needs those assets downloaded again. Do not replace the native decision head with a base transformer's default language/classification head.

For an already prepared workspace, these commands run the gateway and an individual Python model. Use separate terminals; inference should remain serialized during timing:

```sh
uv run python hf/gateway.py
uv --project hf run python hf/model_server.py --kind laya --device mps --model laya:en-mps --port 11436
uv --project hf/julia_runtime run python hf/model_server.py --kind julia --device mps --model julia-1:mps --port 11438
uv run python hf/benchmark_hf.py --model laya:en-mps --label 'Laya English · MPS' \
  --endpoint http://127.0.0.1:11440 --metadata hf/research/laya-runtime.json
```

The benchmark resumes completed trials/games and keeps full requests, raw answers, probabilities, model metadata, and board trajectories in [benchmarks/huggingface/results.json](benchmarks/huggingface/results.json). The report uses the small summary. Audit it together with the original three-model baseline:

```sh
uv run python verify_benchmark.py --results benchmarks/huggingface/results.json --baseline benchmarks/results.json
```

The additional live viewer uses the same compact policy, all legal landings, actual model inference, and model metadata:

```sh
uv run python app.py --model winnow:e4b --endpoint http://127.0.0.1:11440 --port 8767 \
  --benchmark-policy --provider 'Ollaya · Metal' --metadata hf/research/winnow-e4b-runtime.json --autostart
```

Open [live Winnow Tetris](http://127.0.0.1:8767/). The comparison page's replay controls play **saved inference traces**, and are explicitly separate from the live viewer. Physics/adapter tests and both replay verifiers remain applicable.

### Measured results on this Mac

Each row has 64 valid fixed-board trials, with no inference errors. Game lines are totals from seeds 42 and 123, capped at 40 pieces per game; weak models can top out earlier. Accuracy is agreement with the supplied one-step rule, not optimal Tetris play. The table reports warm placement latency, including tournament calls.

| Model / runtime | Median placement | Policy agreement | Game lines / pieces |
| --- | ---: | ---: | ---: |
| Lev 4B / PyTorch MPS BF16 | 6.77 s | 75.0% | 26 / 80 |
| Winnow E4B / Ollaya Metal Q8 | 1.33 s | 62.5% | 26 / 80 |
| Kev 4B / native MLX BF16 | 1.60 s | 46.9% | 7 / 80 |
| Nimble 9B / Ollama Q8 baseline | 2.02 s | 46.9% | 20 / 80 |
| Kev 4B / PyTorch MPS BF16 | 4.65 s | 45.3% | 5 / 79 |
| Tev 4B / Ollama Q8 baseline | 1.50 s | 39.1% | 13 / 80 |
| Tev 0.8B / Ollama Q8 baseline | 315 ms | 18.8% | 1 / 64 |
| CLM v0.1 8B / PyTorch MPS | 2.13 s first order | 15.6% | 0 / 46 |
| Julia-1 / PyTorch MPS FP32 | 48 ms | 7.8% | 0 / 52 |
| Laya multilingual / PyTorch MPS FP32 | 113 ms | 4.7% | 0 / 52 |
| Laya English / PyTorch MPS FP32 | 145 ms | 3.1% | 0 / 49 |

Lev best followed the rule; Winnow was the practical speed/quality leader in this small test, clearing the same 26 lines with a median placement **5.1 times faster**. Kev's native MLX backend was 2.9 times faster than its PyTorch MPS backend on the same checkpoint. Small numerical backend differences changed some choices and game trajectories.

CLM's encoder caches complete state/option embeddings. Its first option-order pass took a median 2.1265 s; the repeated pass took 6.6 ms, with cached embeddings and fresh native-head scoring. The mixed overall median of 240.3 ms is retained in the raw summary, but is unsuitable as a fresh-input comparison. Even the first pass reuses some options and tournament finalists. [Cache timing evidence](benchmarks/huggingface/cache-analysis.json) records this distinction.

The combined [audit](benchmarks/huggingface/verification.json), including the subsequent Clef run, passed: **12 timed variants, 768 valid fixed-board decisions, and 822 reconstructed game moves**. It verifies identical paired inputs, complete legal-option coverage, native selected choices, probability distributions, evaluation labels, summaries, and board outcomes. Five singleton moves were originally mislabeled by the HF runner; only those source labels were corrected after reconstructing their boards, preserving the correction evidence in [source-label-correction.json](benchmarks/huggingface/source-label-correction.json). No inference or measured result changed.

[Julia CPU/MPS checks](hf/research/julia-mps-parity.json) agreed on all four sampled choices, with maximum probability difference 0.00001393. This validates the MPS batch-placement adapter on those inputs; it is not a full numerical-equivalence claim. Jev-Omni has no result because its official loader explicitly requires CUDA.

The additional live Winnow run reached **54 pieces, 19 lines, score 2200, zero holes**, including a visible three-line clear. Pause and one-piece stepping were verified, with no browser warnings/errors. Its [saved live replay](evidence/winnow-live-replay.json) passed a separate [reconstruction audit](evidence/winnow-live-verification.json). [Visual proof](evidence/hf-live-preview.png) shows the board, native preferences, counters and move history; this run is separate from the timed benchmark.

## Clef Flash MLX 4-bit supplement — 2 October 2026

The requested [mlx-community/clef-flash-4bit](https://huggingface.co/mlx-community/clef-flash-4bit) is a distinct quantized 9B variant, pinned to `d9ec324f7992383bdfb7a0b4eed8b4b9d10f81be`. It runs the bundled `clef_mlx.py` backbone and joint decision head in an [isolated locked MLX environment](hf/clef_runtime/README.md). The BF16 head matches the pinned Cloudflare release byte-for-byte. The server preserves the native response and rejects input longer than 16,384 tokens; it does not use chat generation, truncate state, or substitute a PyTorch model. [Provenance](hf/research/clef-flash-mlx-source.json) and [real typed-decision smoke](hf/research/clef-mlx-smoke.json) record the checks.

Clef used the same 32 frozen boards, two option orders, original 26-choice tournament, and two 40-piece seeded games as the earlier variants. All 64 trials were valid; policy agreement was **43.8%**, with **65.6%** nondominated placements. First-order full-placement median was **2.3584 seconds** and P95 **5.3470 seconds**, across 32 states. The matched repeated-order median was **2.3859 seconds**. Loading and the distinct recorded warmup were excluded; no frozen state matched the warmup. The native encoder sorts choice IDs, so identical option sets do not retain the request's ordering; tournament group membership can still change, and the native decoder resolves exact probability ties in request order. The inspected loader has no persistent inference cache.

Seed 42 cleared **6 lines in 40 pieces**, and seed 123 cleared **3 in 40**; both reached the cap. Their final hole counts were 23 and 41. These are capped illustrations, not a tournament distribution. Earlier source traces and summaries were preserved; the table above documents the earlier eleven-variant run. Clef is available alongside them on the [Decision Lab site](https://rayking99.github.io/decmods/).
