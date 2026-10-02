---
license: apache-2.0
library_name: mlx
base_model: Cloudflare/clef-flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- clef
- cloudflare
- systemone
- structured-output
- classification
- multimodal
- custom-code
---

# mlx-community/clef-flash-4bit

[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) converted to MLX (4-bit) for Apple Silicon.

Clef turns a state (text, JSON, images, or video) plus a schema of typed questions into a
probability for every allowed option, in a single forward pass. **It is not a chat model** —
`mlx_vlm.generate`, `mlx_lm.generate`, and LM Studio will load the backbone but produce
meaningless text. Use the bundled `clef_mlx.py` loader, which runs the backbone and the
joint schema head.

## Usage

```bash
pip install mlx-vlm huggingface_hub   # no torch needed
```

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("mlx-community/clef-flash-4bit")
sys.path.insert(0, path)
import clef_mlx

model = clef_mlx.load(path)
response = model.systemone({
    "model": "clef-flash",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle the message?",
            "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
        },
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
})
print(response["answers"])
```

Images (PIL) and videos (frame arrays) go in `images` / `videos`, as in the original:

```python
from PIL import Image
model.predict({
    "state": {"task": "Review the attached receipt."},
    "images": [Image.open("receipt.jpg")],
    "questions": {"legible": {"type": "noul", "instructions": "Is the receipt total legible?"}},
})
```

See the [original model card](https://huggingface.co/Cloudflare/clef-flash) for the input format, question types, and benchmarks.

## Conversion

- Backbone: `mlx_vlm.convert -q --q-bits 4 --q-group-size 64` (vision tower kept in bf16).
- Joint schema head: `joint_head.safetensors` copied unchanged (bf16) and run by `clef_mlx.py`.
- `processor_config.json` is the original from Cloudflare/clef-flash; prompt/token layout matches the reference
  `joint_schema_model.py` exactly (images and video).

## Parity vs. official PyTorch implementation (bf16)

| Inputs | Top answer agrees | Max abs Δprob |
|---|---|---|
| Text (4 records, 10 questions) | 10/10 | 0.029 |
| Images + video (5 records, 9 questions) | 9/9 | 0.119 |

Measured on an M5 Max (128 GB). Small spot-check, not a full benchmark run.

## Quality check: Decision Index (sampled)

| | Decision Index (sample) | Same top answer as bf16 | Mean max abs Δp | Median latency |
|---|---|---|---|---|
| **This model (4-bit)** | **54.65** | 96.4% (15,915 answers) | 0.040 | 311 ms |
| MLX bf16, same rows | 55.63 | — | — | 339 ms |
| Cloudflare published (full suite) | 57.07 | | | |

4-bit costs about **1 index point** vs bf16 on identical rows. Most of the remaining gap to the published score is
already present in bf16 (sample noise and harness differences), not quantization. Agreement is lowest on
low-chance many-option tasks (POP909, GPQA, CLINC150).

Method: [Decision Index](https://github.com/apolinario/decision-index) 0.2.1 kit (suite rebuilt byte-identical), stratified
2,000-request sample across all 44 benchmarks (`suite sample --n 2000`), engine = `clef_mlx.py` with
`max_length=16384` and **no truncation** (over-length requests are refused and count as wrong; 15 of 2,000,
mostly BRIGHT). The index is computed from each benchmark's native metric on the sampled rows with the kit's
chance correction and weights; HLE and iSarcasmEval are set to 0 to match how Cloudflare's published run is
scored. With ~40 rows per benchmark, per-benchmark numbers are noisy (±10+ pts) — only the index is meaningful.

## License

Apache-2.0, following [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash).
