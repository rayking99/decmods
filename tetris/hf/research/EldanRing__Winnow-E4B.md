---
base_model: google/gemma-4-E4B-it
base_model_relation: finetune
license: apache-2.0
library_name: gguf
pipeline_tag: image-text-to-text
tags:
  - winnow
  - gemma4
  - typed-decisions
  - local-inference
  - gguf
---

# Winnow-E4B

**A compact Gemma 4 fine-tune for typed decisions that also supports ordinary chat and image input.**

Winnow-E4B takes a shared state and a set of questions, then assigns probabilities to the answers each question supplies. The decision path reads candidate-answer logits without generating an explanation. It is designed for tasks such as routing a request, choosing an action, checking a condition, or rating urgency when the possible answers are known. The same merged model weights remain available for regular chat; image input uses the matching vision projector.

Winnow-E4B was created by EldanRing from [Gemma 4 E4B IT](https://huggingface.co/google/gemma-4-E4B-it). The downloadable GGUFs have the decision fine-tune merged into the language model. They require no separate adapter or base-model download.

[Run the model](docs/QUICKSTART.md) · [Inference code](https://github.com/EldanRing/winnow-inference) · [API reference](https://github.com/EldanRing/winnow-inference/blob/main/docs/API.md) · [Evaluation details](docs/BENCHMARKS.md)

## How decisions work

The [Winnow inference server](https://github.com/EldanRing/winnow-inference) exposes `/v1/systemone`. A request provides one state and any combination of these question types:

| Type | What the caller supplies | What the server returns |
|---|---|---|
| `noul` | A yes/no question | Probability of `true` |
| `choice` | Named options with optional descriptions | Selected option, option probabilities, confidence |
| `score` | An ordered list of levels | Level probabilities, expected score, confidence |

The server shares the state prefill across questions, reuses a matching cached prefix when possible, and evaluates only the verified answer-token candidates for each question. Questions can be processed in parallel branches or waves without loading another copy of the model. Ordinary `/v1/chat/completions` uses the normal vocabulary head and chat template, including streaming and image messages when the projector is loaded.

For example, one request can ask both whether a customer wants a refund and which team should handle the ticket:

```json
{
  "model": "Winnow-E4B",
  "state": {"ticket": "I was charged twice and need a refund."},
  "questions": {
    "refund": {
      "type": "noul",
      "instructions": "Does the customer request a refund?"
    },
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this ticket?",
      "criteria": {
        "billing": "Payments and refunds",
        "technical": "Software defects"
      }
    }
  },
  "winnow": {"temperature": 1.2574172017327816}
}
```

The [quickstart](docs/QUICKSTART.md#ask-typed-questions) shows a complete command. Candidate probabilities are conditional on the options supplied in a request. Confidence describes how concentrated the distribution is; it is not a guarantee that an answer is correct.

## Model and training

This is a **rank-32, alpha-64 LoRA fine-tune** of the instruction-tuned Gemma 4 E4B model. A selected decision adapter was continued on targeted hard cases, with step 438 chosen for export. The update trained language tensors; the vision and audio modules remained frozen. The adapter was merged in FP32 before conversion to Q8_0 and BF16 GGUF. Chat still has its full generation head, while the decision server projects only the requested answer rows.

The private training set combines synthetic contrastive decisions, verified labels, teacher distributions where appropriate, semantic tasks, and targeted hard cases. Training, development, calibration, and reserved tests were separated. **The training data and adapter checkpoint are not released.** Public evaluation suites were known before the targeted continuation was designed and informed task selection; the results below should be read as a comparison of deployed models, not a fresh unseen-task estimate.

## Decision results

These figures evaluate the **downloadable Q8 GGUF** against Winnow-12B Q8 on the same questions. They report top-choice accuracy, except the local typed panel, which reports agreement with synthetic teacher labels.

| Evaluation | Winnow-E4B Q8 | Winnow-12B Q8 |
|---|---:|---:|
| JevBench public subset, 231 questions | **80.52%** (186/231) | 85.71% (198/231) |
| Kev v9 clean, 1,046 questions | **72.66%** | 81.45% |
| Kev v9 additional, 390 questions | **55.90%** | 68.97% |
| Local typed decisions, 2,000 decisions | **72.30%** | 70.10% |

Winnow-12B leads on the broader public suites. E4B leads by 2.2 points on this local typed panel, which uses synthetic teacher targets rather than independently verified truth for every item. The JevBench figure is **public-subset accuracy**, not the official composite leaderboard score. Methods, uncertainty intervals, calibration measures, and evaluation boundaries are in the [benchmark report](docs/BENCHMARKS.md).

![Winnow-E4B and Winnow-12B Q8 decision scores on three evaluation panels](assets/e4b-quality.png)

*Decision quality on the same questions. The local typed panel measures agreement with synthetic teacher labels.*

In a separate matched 8K text decision workload, E4B Q8 delivered **147.8 decisions/s** at 64 questions per request, versus **67.2 decisions/s** for Winnow-12B Q8. Those are warm request medians on an RTX 5070 Ti, not a guarantee for other hardware or workloads. The [timing table](docs/BENCHMARKS.md#same-gpu-latency-and-memory) includes smaller batches and memory use.

![Winnow-E4B and Winnow-12B Q8 warm decision throughput, memory, and single-decision latency](assets/e4b-speed-memory.png)

*Matched text-only workload on an RTX 5070 Ti with Q8 weights and KV cache. Timings are medians of ten warm requests.*

## Probabilities and calibration

The server normalizes candidate logits over the options in each question. The optional `winnow.temperature` setting scales those logits without changing which option wins. For **Q8 text decisions**, `1.2574172017327816` was fitted on 778 separate calibration questions. For **BF16**, the separately fitted value is `1.3331553765162731`; do not reuse Q8's temperature for BF16.

These temperatures were measured for the evaluated text-decision profile. Calibration can change with task, prompt length, image input, and deployment settings. Check probabilities on held-out examples from the intended application before treating them as confidence in real-world correctness.

## Downloads and deployment

| File | Role | Size |
|---|---|---:|
| [Winnow-E4B-Q8_0.gguf](gguf/Winnow-E4B-Q8_0.gguf) | Recommended, fully measured language model | 8.01 GB / 7.46 GiB |
| [Winnow-E4B-BF16.gguf](gguf/Winnow-E4B-BF16.gguf) | Optional higher-precision language model | 15.05 GB / 14.02 GiB |
| [mmproj-Winnow-E4B.gguf](gguf/mmproj-Winnow-E4B.gguf) | Matching F16 projector for image input | 990 MB / 0.922 GiB |

Download Q8 for the measured decision, chat, and vision setup; add the projector for images. BF16 passed 8K text-decision serving but was not tested with images or populated 64K context. On reserved text tests it showed no consistent quality advantage over Q8. [Verify files](SHA256SUMS) after download.

| Q8 configuration | Observed peak device memory | Suggested GPU capacity |
|---|---:|---:|
| 8K text-only decision workload | 8.56 GiB | 10 GB+ VRAM |
| 64K vision and chat operational probes | 10.72 GiB across the probes | 12 GB+ VRAM |

Capacity suggestions come from one RTX 5070 Ti using Q8 KV cache, four decision branches, one chat slot, and exclusive memory scheduling. They have not been tested on every GPU in those tiers. The Q8 model and projector answered six questions over a 65,023-position state that included a synthetic image; repeated cached requests took 129–131 ms after an 11.59 s cold request. A separate 62,431-position multimodal chat prompt generated 512 tokens at 71.2 output tokens/s. The [benchmark report](docs/BENCHMARKS.md#64k-context-and-vision-smoke-test) describes these operational probes and their limits.

![Winnow-E4B and 12B, side by side: Q8 GPU memory, chat generation, and E4B cached 64K decision timing](assets/e4b-profiles.png)

*Q8 operational profiles on an RTX 5070 Ti. The 8K text workload was matched; the 64K vision and chat probes used different images and question counts. The 10 GB+ and 12 GB+ capacity suggestions are inferred from measured memory, not tests on those cards.*

## Scope and limitations

- The decision path is intended for supplied answer options. Open-ended responses use the regular chat endpoint.
- General chat quality, real-image vision accuracy, and broad long-context reasoning have not been established by these probes. The image test used a synthetic status panel. Vision and audio modules were not fine-tuned for this release.
- The public suites were known during targeted data design. The local typed panel measures agreement with synthetic teacher labels. Neither result establishes accuracy for every downstream task.
- Probability calibration was fitted on text decisions. It needs validation for a new domain or multimodal workload.
- The 64K and vision measurements apply to **Q8 with the matching projector**. BF16 was evaluated only in the text profile.

## Credits and license

Winnow-E4B is an independent fine-tune by EldanRing of Google DeepMind's [Gemma 4 E4B IT](https://huggingface.co/google/gemma-4-E4B-it), released under [Apache 2.0](https://ai.google.dev/gemma/docs/gemma_4_license). See [LICENSE](LICENSE) and [NOTICE](NOTICE). The separate inference code builds on [llama.cpp](https://github.com/ggml-org/llama.cpp) and preserves its MIT license. Jev-style describes the typed-decision interface; Winnow is not affiliated with or endorsed by TypeSafe, Google, or llama.cpp.
