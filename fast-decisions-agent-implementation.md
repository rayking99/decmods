# Fast decision models and Python implementation

Research checked on 1 October 2026. This brief instructs an implementation agent to build a small decision layer for data ingestion and data engineering. It covers existing models, local and hosted inference, a Python contract, evaluation, and an incremental implementation plan. Recommendations and effort estimates are engineering judgments; reported benchmark results remain attributed to their publishers.

Implement a service that turns supplied evidence into a bounded answer: a known document type, an approved parser, a catalog column ID, a duplicate candidate ID, or a decision to defer. Reuse an inference runtime. Build a thin adapter and application contract rather than a new model engine. Start with Ollama Nimble locally and Luna Structured Outputs online; evaluate Julia when CPU operation and a small download matter. Treat OpenAI Decisions as a future adapter until a public contract and account access are verified.

## Scope and implementation boundaries

Implement semantic classification and routing. Keep exact operations in ordinary code: parsing dates and amounts, checking schema constraints, applying database permissions, calculating totals, matching hashes, and executing approved transformations.

The model proposes a typed value. The application validates that value against the current catalog and workflow. A separate policy determines whether to act, defer, or request more evidence. Do not let model outputs create SQL, choose arbitrary URLs, introduce new tool names, or execute ingestion writes directly.

This document and its Python examples are reference designs. No model weights were downloaded, no paid inference was performed, and no latency or accuracy was measured on this machine. Syntax and document checks are reported separately from model acceptance.

## What System One means

The recent System One category packages fast semantic judgments as state plus typed questions. TypeSafe popularizes the name by analogy with fast intuitive thinking. Its three primitives are a choice among supplied answers, a probability for a yes/no proposition called `noul`, and a score against ordered criteria. This terminology describes a product category and interface; it does not identify one universal architecture. [TypeSafe concepts](https://docs.typesafe.ai/concepts/system-one)

For implementation, distinguish three things:

| Layer | What it does | What the implementation agent must own |
| --- | --- | --- |
| Model | Assigns scores or generates a constrained answer | Model selection, version and evaluation |
| Runtime | Tokenizes inputs, executes inference and constructs outputs | Correct loader, residency and hardware configuration |
| Application contract | Defines accepted decisions and their consequences | Validation, policy, provenance and error handling |

A conventional language model with an enum JSON schema can satisfy the application contract. A native decision model may provide probabilities directly and avoid long decoding. Their output semantics differ, so keep probability availability explicit in the adapter interface.

## How decision inference is implemented

### Encoder and decision head

An encoder reads the state, question and candidate descriptions. A trained head scores candidate representations. A softmax over the permitted candidates yields a distribution; an argmax selects the choice. Julia is built on multilingual mmBERT-small with decision components. Laya describes a bidirectional encoder with option markers and a learned decision head. Both expose candidate sets at inference time. Use their published runtimes because token construction and head loading are part of the checkpoint contract. [Julia card](https://huggingface.co/SupersonicLabs/Julia-1), [Laya card](https://huggingface.co/convaiinnovations/laya)

### Language model used as a scorer

An existing language model can be adapted to score answer letters or answer slots. Restrict the readout to valid options, then normalize their scores. Nimble uses a Qwen3.5-9B adaptation and a scoring runtime; its latest published checkpoint has no separately fitted temperature. Tev1 retains the ordinary next-token language-model head and is explicitly described as a Jev-inspired experiment. Consequently, “decision model” does not automatically imply a non-autoregressive architecture. [Nimble weights](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B), [Tev1 card](https://huggingface.co/togethercomputer/Tev1-4B-experimental)

Mapika Decider reads option-letter scores at answer slots. Kev uses a LoRA adapter and pointer head. These need their own inference code; a chat endpoint that merely loads the base model does not recreate their decision behavior. Do not assume any GGUF conversion preserves a custom head. [Decider card](https://huggingface.co/Mapika/decider-4b), [Kev card](https://huggingface.co/jaredpalmer/kev-4b)

### General model with Structured Outputs

Give Luna an explicit task and constrain the response using a strict JSON schema with enum values. The result is generated JSON validated against the schema. This is immediately useful for the same routing workflow, but it does not supply a calibrated candidate distribution. A generated `confidence: 0.95` is another model-generated field. Do not normalize it into a native probability. OpenAI documents Structured Outputs and Python parsing helpers; refusals and incomplete responses still need handling. [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

### Embedding and classical approaches

JevEmbed is a separate implementation route: fine-tuned embeddings with decision-specific prompts and scoring. Its released Qwen embedding variant uses a 1,024-token configuration with truncation and fixed scoring parameters. It is worth testing when embedding infrastructure is already available. [JevEmbed card](https://huggingface.co/HIT-TMG/JevEmbed-Qwen3-Embedding-0.6B)

Keep a classical baseline in the evaluation. For a stable label taxonomy, a supervised classifier, rules, or nearest-neighbor retrieval may beat a general decision model. One published banking adaptation reports validation macro F1 of 0.740 for fine-tuned Laya versus 0.860 for TF-IDF plus logistic regression on the same split. That is a task-specific result, not a universal ranking. [Banking experiment](https://huggingface.co/dimitrisdais/laya-banking77-typed-decisions)

## Local inference options

These recommendations are starting points for evaluation, not a benchmark ranking.

| Option | Deployment and fit | Implementation recommendation |
| --- | --- | --- |
| Ollama Nimble | Local decision endpoint; relatively large 9B model | First local baseline on a capable Mac or workstation |
| Julia-1 | Small multilingual encoder; CPU inference | First small CPU candidate for clear, short decisions |
| GLiNER2.5-Decide | Encoder classifier with a native classification API | Test for document type, intent, tags and short classification |
| Laya | Encoder family with an SDK and HTTP server | Use for domain adaptation; evaluate base zero-shot behavior carefully |
| Mapika Decider | Open Qwen-based family with its own server | Test when a GPU and more semantic capacity are available |
| Kev | Open adapter and pointer head with its own server | Alternative GPU candidate; verify the chosen revision and runtime |
| Existing local LLM | JSON schema through a local generation server | Reuse when extraction or explanations are also needed |

### Ollama Nimble

Ollama v0.35.0 or later supplies `POST /v1/systemone`. Use `ollama pull nimble`, then call HTTP or the TypeSafe SDK. The Ollama model page explicitly says decision inference is not yet in the Ollama CLI or Python/JavaScript libraries; pulling a model and running decision inference are separate operations. Its published Ollama evaluation reports Nimble 75.7%, Tev1 4B 73.3%, and Tev1 0.8B 63.5% across 13 datasets. These are publisher measurements. [Ollama Nimble](https://ollama.com/library/nimble)

The endpoint accepts up to 64 questions, with 2–26 options or score levels, and a 64 KiB request limit. It is local and text-only; images, streaming, tools and generation controls are unsupported. Each question is scored separately against the full rendered input. Token usage includes repeated input across questions. Use `keep_alive` to control residency. [Ollama API](https://docs.ollama.com/api/systemone)

Do not promise that adding questions is free or that the state is processed only once. Batching behavior belongs to the runtime. Measure one-question and many-question calls separately.

### Julia-1

Julia has 144.3M parameters, Apache-2.0 artifacts and approximately 550.5 MiB of FP32 weights. Native calls accept 2–20 candidates and support an 8,192-token combined budget; historical accuracy evaluation used 1,024 tokens, while the 8K result is a smoke test. The repository includes a resident Python loader and named-question `predict` interface, plus a separate ONNX/WebGPU release. It is not a generic Transformers classification pipeline. [Julia card](https://huggingface.co/SupersonicLabs/Julia-1)

The publisher reports 73.15% across 2,000 typed decisions, 94/100 AG News cases, 86/100 Emotion cases, and 64/100 on a Banking77 pilot with a shortlist. The Jev values are supplied references rather than a fresh comparison run. Large candidate hierarchies can discard the correct label, and their probabilities describe only final candidates. [Julia evaluation artifact](https://huggingface.co/SupersonicLabs/Julia-1/blob/main/metrics/accuracy-20260924.json)

Load the published Python package from a downloaded snapshot. Import `load_model` from `julia`, keep the engine resident and call `predict(state=..., questions=...)`. Preserve strict encoding. Use the attached `examples/julia_service.py` only as a small HTTP wrapper around that runtime. Pin the model revision and test Boolean descriptions, which have changed in the published implementation.

### GLiNER and Laya

GLiNER2.5-Decide exposes `AutoExtractor.from_pretrained` and `classify_text`, with single-label and multi-label tasks. Its card describes a 340M English classifier on DeBERTa-v3-large; the Hub's total tensor count includes additional parameters, so do not confuse that architecture label with memory requirements. A multilingual variant is separately published. Adapt its native classification result rather than assuming Jev wire compatibility. [GLiNER card](https://huggingface.co/fastino/GLiNER2.5-Decide)

Laya offers `pip install laya` and a `laya[serve]` extra. Its current card warns that base checkpoints are near chance on zero-shot typed-decisions; its stronger result comes from a checkpoint fine-tuned on that benchmark's training split. Runtime defaults and truncation need checking. Treat it as a base to specialize instead of accepting the opening calibration claims as a deployment guarantee. [Laya card](https://huggingface.co/convaiinnovations/laya)

### Larger open decision models

Mapika's published family includes 0.8B, 2B, 4B and 35B-A3B models and a vision variant. The 4B runtime needs PyTorch, Transformers and optional acceleration kernels; its card lists 8.4 GB of BF16 weights and per-type calibration temperatures dependent on package version. Use its existing server before implementing one. [Decider card](https://huggingface.co/Mapika/decider-4b)

Kev's current 4B card warns that Qwen3.5 DeltaNet lacks an MPS kernel path in its PyTorch setup, making that version slow on a Mac; it points to an older Qwen3 revision for lower latency. This is a runtime-specific warning, not proof that every Qwen-based decision model is slow on Apple hardware. [Kev card](https://huggingface.co/jaredpalmer/kev-4b)

Do not load Julia, Nimble and larger GPU models into one environment by default. Their loaders, dependency versions and memory footprints differ. Separate inference processes let the application call one stable contract.

## Online inference options

### Luna Structured Outputs

Use `gpt-6-luna` with the Responses API and a strict enum schema as the initial hosted implementation. The official model page supports Structured Outputs and `reasoning.effort="none"`; try that setting first for bounded routing, then compare a higher effort on difficult cases. Published text rates are $0.10 per million input tokens and $0.50 per million output tokens. Account access, processing mode and regional settings still affect deployment. [Luna documentation](https://developers.openai.com/api/docs/models/gpt-6-luna)

The attached `examples/luna_structured.py` demonstrates a decision over known parser IDs. It deliberately returns no probability. Keep instructions outside untrusted state, put an explicit `review` choice in the schema, and validate that the returned ID is still allowed when the application executes the plan.

A strict schema makes the output shape predictable. It does not make the judgment deterministic or correct. Use retries for transient transport failures; do not retry a refused or ambiguous decision until it becomes a preferred answer. Handle refusal, incomplete generation and parsing errors as explicit unsuccessful outcomes.

For budgeting, 1,000 input tokens and 20 output tokens at the quoted standard rates cost about $0.00011 per call, or $110 per million calls, before retries, caching or other charges. This is an illustrative calculation, not a Decisions API price.

### Jev

TypeSafe's current model documentation lists Jev 1.13.0 at $0.042 per million input tokens with output free, text-only input, a 64K total request budget and a 32K state-plus-longest-question budget. It describes reading shared state once and evaluating questions in parallel. Pin the version when thresholds depend on it. The HTTP API allows up to 255 choice options and up to ten score levels. [Model limits and price](https://docs.typesafe.ai/models), [API reference](https://docs.typesafe.ai/api)

This is the direct hosted candidate when distributions over options are required. Verify actual endpoint access, billing and your task's selective error rate before selecting it. Its public architecture and training narrative do not provide open weights with which to reproduce its full implementation.

### Hosted Decider 1 from meraGPT

meraGPT documents a hosted `sd-1` service using the System One schema, up to ten choice labels and a 4,096-token request budget. Its launch post quotes $0.03 per million input tokens, output free, and 76.8% agreement on typed-decisions versus 72.7% for Jev in its run. Labels in that benchmark are teacher-ensemble references, and the post reports one evaluation run. [meraGPT launch](https://meragpt.com/blog/introducing-decider-1)

Keep this hosted product distinct from Mapika's open Decider family. A shared product name is not evidence that the weights, runtime or evaluation are the same.

### OpenAI Decisions API

OpenAI's 29 September announcement confirms a Luna-based Decisions API for finite predefined answers, accepting text or images. It was in limited preview, with a broader release planned in the coming days. The announcement does not establish a public request schema, price or SDK method. [Official announcement](https://openai.com/index/devday-2026-recap/)

A live public-source check on 1 October found no decision-named resource paths, client members, API index entries or changelog references in the checked SDKs:

| SDK | Main version read | Main commit inspected |
| --- | --- | --- |
| Python | 3.22.1 | `157ac4c49a34891d5b66d2950b5082d1b3268315` |
| JavaScript/TypeScript | 7.25.0 | `71d24120c4cc4e5897a767f16d25f2804fa4ad7c` |

This establishes absence of a public dedicated resource in those snapshots; it does not establish absence of private preview access. Do not invent `client.decisions.create(...)` or copy an unofficial payload. [Python snapshot](https://github.com/openai/openai-python/tree/157ac4c49a34891d5b66d2950b5082d1b3268315), [Node snapshot](https://github.com/openai/openai-node/tree/71d24120c4cc4e5897a767f16d25f2804fa4ad7c)

The originally linked Hugging Face article is a community article. Its example is explicitly illustrative, and it sends readers to third-party launch coverage. Use it for context rather than endpoint implementation. In particular, do not treat its approximately 150 ms claim as a measured result for your deployment. [Original article](https://huggingface.co/blog/sora-2/what-is-openai-decisions-api-a-practical-guide)

## Recommended Python application API

Build an internal `DecisionClient.evaluate(request)` interface. Add HTTP only when another process or language needs it. A single Python ingestion application can call a resident local model directly. No extra server is required.

Use an application-owned endpoint such as `POST /v1/decisions` if a service is needed. Keep provider-native `/v1/systemone` behind the adapter. Your endpoint is a proposed application API, not an OpenAI endpoint or an industry standard.

### Request design

Keep evidence separate from the versioned question definition. For production, resolve `decision_spec` server-side from a registry so callers cannot silently change label meanings.

```json
{
  "request_id": "ingest-1842",
  "decision_spec": "parser_route_v1",
  "input_revision": "sha256:SOURCE_CONTENT_HASH",
  "state": {
    "filename": "supplier-export.csv",
    "header": ["Inv No", "Supplier", "Amount", "Due Date"],
    "sample_rows": [
      ["A42", "Example Supplier", "125.00", "2026-09-30"]
    ]
  }
}
```

The registry resolves that specification to known candidate IDs and descriptions:

```json
{
  "route": {
    "type": "choice",
    "instructions": "Select the supported parser indicated by the header and sample rows. Select review if evidence is insufficient or incompatible.",
    "criteria": {
      "invoice_csv_v1": "Invoice rows with invoice identifier, supplier, amount and due date",
      "inventory_csv_v1": "Stock rows with product identifier, quantity and location",
      "review": "Neither parser is supported by the evidence, or the input is ambiguous"
    }
  }
}
```

Do not submit a whole data lake or a whole 100,000-row file for a route decision. Use headers, representative samples, schema profiles, source identity and the small policy fragment relevant to the question. Include edge cases as well as typical rows. If sampling cannot establish a whole-file property, return “unknown” or run a full deterministic scan.

### Response design

Separate the selected candidate, uncertainty and the application's disposition. Preserve the provider's raw response in a controlled audit store rather than flattening away differences.

```json
{
  "request_id": "ingest-1842",
  "decision_spec": "parser_route_v1",
  "input_revision": "sha256:SOURCE_CONTENT_HASH",
  "backend": {"provider": "ollama", "model": "nimble", "revision": "PINNED_DIGEST"},
  "answers": {
    "route": {
      "value": "invoice_csv_v1",
      "probabilities": null,
      "uncertainty": {"kind": "unavailable", "value": null}
    }
  },
  "disposition": "review",
  "policy_version": "ingestion_acceptance_v1",
  "reason_code": "threshold_not_validated"
}
```

This example intentionally omits scores and defers. Fill probabilities only from an actual native distribution. Use uncertainty kinds such as `entropy_concentration`, `max_probability`, `calibrated_probability` or `unavailable`, with documented meanings. A model's response is not proof that a calibration version exists.

Luna's enum answer can pass an application policy based on independently established error rates without claiming per-request calibrated uncertainty. If its proposed outcome is `review`, preserve that outcome. Never convert a single generated choice into a one-hot probability distribution.

### Core implementation modules

Create these components in the target application:

```text
decisions/
  contracts.py       Request and result types and validation
  registry.py        Versioned question and candidate definitions
  client.py          evaluate interface and backend selection
  adapters/
    ollama.py        HTTP to an existing System One server
    julia.py         Published resident engine
    luna.py          Responses with strict enum schemas
    typesafe.py      Hosted native decisions
  policy.py          Acceptance and escalation rules
  audit.py           Provenance and immutable results
  evaluate.py        Held-out replay and comparison reports
```

Prefer Pydantic for the application contract and FastAPI for an optional service. Use `httpx.AsyncClient` for remote providers. Put synchronous local inference in a worker thread or dedicated process, with bounded concurrency. Load weights once at process startup and warm them before readiness. Avoid one model copy per web worker.

The adapter must expose capabilities: supported primitive types, input modalities, token limits, maximum options, probability semantics and model revision. Validate the registry against those capabilities before accepting traffic. Select compatible backends explicitly; never silently trim candidates, truncate input or send private data to a hosted fallback.

For failures, return a typed error and no actionable decision. Distinguish invalid input, missing model, overflow, timeout, refusal and runtime failure. Apply bounded retries to transient remote failures; use a bounded queue for local inference. Health checks should distinguish process liveness from model readiness.

### Do we build our own API

| Need | Action | Estimated effort after prerequisites |
| --- | --- | --- |
| Call Ollama locally from Python | Use HTTP or TypeSafe SDK | 1–2 hours for a first integration |
| Add Luna enum decisions | Use OpenAI Responses and a strict schema | Half a day for an adapter and failure handling |
| Call Julia within one Python app | Install the published runtime and retain one engine | Half a day to a day, depending on dependency setup |
| Share a local model across applications | Wrap the engine with FastAPI | 1–2 days for a prototype and concurrency handling |
| Establish a reliable ingestion workflow | Add labeled replay, policy, provenance and deployment checks | Several days to weeks depending on domain and error cost |
| Train a new general decision model | Develop data, training and calibration infrastructure | A separate research effort; unnecessary for the first release |

These are implementation estimates, not measured project timings. The HTTP wrapper is easy. Choosing questions, proving error rates, and maintaining behavior across changing inputs are the substantial work.

## Probability and confidence handling

For a native choice, candidate scores may be normalized as:

```text
p_i = exp(z_i / T) / sum_j exp(z_j / T)
choice = argmax_i p_i
score = sum_i i * p_i
```

Apply the runtime's published temperature once. Keep raw scores and calibration provenance when available. Do not refit temperatures on the test set or carry temperatures across checkpoint revisions.

Ollama defines confidence as `1 - H(p) / ln(N)`: concentration relative to a uniform distribution. It is not calibrated correctness. Scores are probability-weighted zero-based indices, not normalized probabilities. A middle score can represent certainty about the middle category or uncertainty split between extremes; retain the distribution. [Ollama specification](https://docs.ollama.com/api/systemone)

TypeSafe's own limitations describe disagreements between equivalent Choice/Noul formulations, non-complementary answers to separately phrased Boolean questions, date comparison problems, irrelevant-context degradation and susceptibility to adversarial state. Do not assume consistency identities across independent answers. [Jev limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13)

An early independent security evaluation found that aggregate calibration can hide confident failures in particular attack groups, adapted models did not consistently improve over their base models, and strict error budgets left limited automatic allowance. These findings concern the evaluated configurations and tasks. They support measuring actual automation outcomes rather than accepting a “calibrated” product label. [Security evaluation preprint](https://arxiv.org/html/2609.33401v2)

Sys1Cal-v1 tests exact probability semantics on controlled synthetic problems. It finds that Jev Noul and Score align better with underlying probabilities than Choice, and equivalent representations can change answers. The authors explicitly limit generalization beyond the synthetic benchmark. [Probability semantics preprint](https://arxiv.org/abs/2609.35342)

## Data engineering workflows to implement first

| Workflow | Model input | Bounded output | Deterministic follow-up |
| --- | --- | --- | --- |
| Source classification | Header, sample rows and source metadata | A known source category or review | Check supported schema and provenance |
| Parser routing | Known formats and a small payload sample | Parser ID from a registry | Execute only that installed parser |
| Column mapping | A source column and candidate catalog fields | Catalog field ID or no match | Validate type, units and one-to-one constraints |
| Entity resolution | One record and a retrieved candidate set | Candidate ID or new/ambiguous | Check identifiers and candidate freshness |
| Data quality triage | A failed check and relevant context | Ignore, investigate or quarantine recommendation | Enforce fixed severity and retention policy |
| Change classification | A schema diff | Additive, breaking or unknown | Compute exact compatibility rules |
| Extraction validation | A proposed extraction and its source span | Supported, contradicted or insufficient evidence | Verify span coordinates and literal source presence |
| Model routing | Task category and constraints | Local model, hosted model or review | Apply privacy, cost and latency policy |

Start with parser routing because its output is small and downstream validation can catch many errors. Add column mapping next. Enforce global mapping constraints in code; independently selecting a field for each column does not ensure a valid whole-schema mapping.

For entity resolution, retrieve candidates before making the decision. Evaluate retrieval recall separately: a decision model cannot select the correct entity if it never enters the candidate set. Record candidate IDs and revisions with the answer. A shortlist distribution is conditional on that shortlist, not a global identity probability.

## Evaluation and acceptance instructions

Implement a replay dataset from representative real inputs. Split by supplier, source, document family or time so near duplicates cannot leak between development and test. Keep a separate development/calibration set and untouched final test set. Begin with hundreds of examples for discovery, then increase sample size to support the required error bound; a small successful sample cannot establish rare-error reliability.

Compare rules, a classical classifier where feasible, Julia, Nimble, and Luna under the same application question definition. Allow necessary native prompt adaptations, but record them. Measure:

- Per-class precision and recall, confusion matrices and abstention rate.
- Invalid-output and overflow rates, including unsupported rows.
- Automation coverage versus observed error at each acceptance threshold.
- Brier/log loss and reliability plots where native probabilities exist.
- Cold startup, warm p50/p95 latency, memory and throughput at target concurrency.
- Cost per source record and per accepted decision, including retries and fallback.
- Robustness to paraphrases, option reordering, missing facts and conflicting evidence.

Test overflow without silent truncation. Test incompatible candidate IDs, JSON-shaped instructions inside state, descriptions that invert Boolean meanings, independent answers that conflict, and equal scores. Evaluate the quantized serving artifact, not just its original FP32/BF16 weights; a preserved top label does not establish preserved calibration.

Pick thresholds against an explicit error budget. Shadow the ingestion pipeline before allowing actions. Record rejected or deferred examples so the next improvement targets real gaps. Deterministic checks remain required after acceptance.

## Implementation sequence for the agent

1. Implement the registry for `parser_route_v1`, including `review`, source revision and allowed parser IDs. Write deterministic schema checks first.
2. Add the Ollama adapter with request validation and raw result capture. Prove it against a real installed Nimble version; document hardware and model digest.
3. Add Luna strict-schema output. Keep probabilities unavailable. Prove refusal and incomplete-response handling before accepting results.
4. Run a shared replay suite and report accuracy, coverage, latency and cost. Use a baseline that ignores the input to expose class imbalance.
5. Add Julia as the CPU alternative in a separate environment. Use strict encoding and pinned weights. Re-run the same suite.
6. Implement policy and audit storage, then run in shadow mode. Promote only the selected, verified workflow to automatic parser routing.
7. Add column mapping and candidate retrieval after the first workflow meets its acceptance criteria.
8. When OpenAI publishes Decisions documentation, implement a separate adapter from that exact schema. Compare it with Luna generation before switching production behavior.

Do not train a new model, add every listed backend, or build an agent that autonomously changes ingestion code in the first implementation. Deliver one useful bounded workflow with evidence that it works.

## Future direction for fast decisions

The following is a forecast and design judgment, not a verified vendor roadmap.

Fast decisions are likely to become reusable nodes inside data pipelines: semantic branches, catalog matching, exception triage, retrieval filtering and escalation. Their value grows when a pipeline must repeat a small ambiguous judgment at high volume. The application can keep its state machine and use the model only where language or messy records defeat straightforward rules.

Expect convergence at the application contract while architectures remain varied. Encoders can make compact CPU deployments practical; language-model scorers can retain more background knowledge; general models can provide structured extraction and explanations; hosted native APIs can optimize throughput. The durable interface is evidence plus a versioned question and allowed outcomes, with explicit failure and uncertainty semantics.

The strongest long-term asset is a decision registry paired with a domain evaluation corpus. Model weights can change. A catalog of intended decisions, source evidence, approved outcomes and measured failure cases keeps the system understandable and replaceable. Record whether an uncertainty value came from normalized logits, a fitted calibration transform or generated text.

Use generative models during development to propose candidate mappings, draft question definitions and help label difficult examples. Keep approved runtime question definitions fixed and reviewable. Stable, high-volume decisions can later be distilled or fine-tuned into smaller models when labeled outcomes justify that investment. Generated training labels require independent checks.

Do not optimize solely for the smallest inference latency. A routing decision that starts the wrong expensive pipeline can erase millions of cheap decisions' savings. Optimize end-to-end cost per correct accepted record, with review and recovery included. Evidence quality, candidate recall and exact validation often matter more than the model brand.

## Further discovery sources

The publisher-maintained Hugging Face Decision Index tracks trained models, scoring wrappers, diffusion approaches and prior art. Its current manifest includes Intern-Decision, Lev, MoJev, NanoJev, Lumma-Fev, JPT, Decision 1.0 variants, and other independent reproductions. Use it to locate original artifacts; do not read a changing leaderboard as a deployment recommendation. [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)

The most directly useful original sources are the model cards and runtime documentation linked above. Preserve their revisions in an implementation report. The attached examples exercise the published interface shapes but are not substitute benchmarks.
