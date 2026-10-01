# Decision inference reference examples

Read [the implementation brief](../fast-decisions-agent-implementation.md) first. These examples illustrate separate backends; they are not a finished ingestion application or a single shared dependency environment.

## Ollama HTTP

Use Ollama 0.35 or later and download Nimble with `ollama pull nimble`. Start Ollama, then run:

```sh
python3 ollama_decision.py
```

The client uses the standard library. Native HTTP errors and timeouts propagate to the caller. Production code should translate them into the application's typed errors.

## Luna Structured Outputs

In a separate Python environment, install the official `openai` package and configure `OPENAI_API_KEY`. Running the following sends the example state to OpenAI and incurs API usage:

```sh
python3 luna_structured.py
```

The example uses the regular Responses API. It does not call the limited-preview Decisions API. The result includes no probability estimate.

## Julia HTTP wrapper

Download `SupersonicLabs/Julia-1` at a pinned revision with actual model weights. Install that snapshot's Python package using its `pyproject.toml`, then install `fastapi` and `uvicorn` in that environment. The published runtime imports as `julia`.

```sh
export JULIA_MODEL_DIR=/absolute/path/to/pinned/Julia-1
uvicorn julia_service:app --host 127.0.0.1 --port 8008 --workers 1
```

The wrapper demonstrates resident CPU inference and serialized execution. Its request type supports choices only, limits questions to 16 by application design, and has no production request-size middleware, admission limit, timeout recovery, authentication, warmup or audit storage. Add those controls before deploying beyond a local prototype. Julia's native response is preserved; this is not a promise of full TypeSafe response compatibility.

## Verification status

Python syntax checks and offline interface checks can validate example structure. Model loading, paid inference, hardware latency, backend error behavior and application accuracy require a separate acceptance pass with the real dependencies and pinned model artifacts.
