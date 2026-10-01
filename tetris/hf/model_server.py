"""Small TypeSafe-wire adapter over native decision runtimes; never substitute heads."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parent


class LayaRuntime:
    def __init__(self, device, checkpoint="laya"):
        from laya import Agent
        directory = ROOT / "models" / checkpoint
        if checkpoint == "laya-multilingual":
            directory = directory / "multilingual"
        self.engine = Agent(str(directory), device=device)
        self.metadata = {"device": str(self.engine.device), "dtype": str(next(self.engine.model.parameters()).dtype),
                         "backend": "native Laya / PyTorch", "max_len": 8192, "head_max_len": 2048,
                         "note": "Extended lossless context; English checkpoint trained with a 512-token default. "
                                 "Native runtime clamps out-of-range shipped choice calibration temperatures."}

    def answer(self, state, questions):
        from laya.common import render_options, serialize_state
        agent = self.engine
        internal = {k: agent._to_internal(v) for k, v in questions.items()}
        for q in internal.values():
            # Native layout caps each option at 48 tokens. Refuse instead of silently cutting metrics.
            lengths = [len(agent.tok(" " + text, add_special_tokens=False).input_ids)
                       for text in render_options(q)]
            if max(lengths, default=0) > 48:
                raise ValueError("An option exceeds the native 48-token contract")
            head = agent.tok(f"{q['t']} question: {q['ins']}", add_special_tokens=False).input_ids
            if len(head) + sum(n + 1 for n in lengths) > 2048:
                raise ValueError("Question/options exceed the lossless head budget")
        result = agent.predict(state, questions, max_len=8192, head_max_len=2048)
        if result["usage"].get("truncated") or result["usage"].get("options"):
            raise ValueError("Native Laya input was truncated or options collapsed")
        if str(agent.device) != self.metadata["device"]:
            raise RuntimeError("Native Laya changed device during inference")
        return result


class MultilingualLayaRuntime(LayaRuntime):
    def __init__(self, device):
        super().__init__(device, "laya-multilingual")
        self.metadata["note"] = "Native multilingual checkpoint supports extended 8192-token context. Lossless inputs."


class CLMRuntime:
    def __init__(self, device):
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
        sys.path.insert(0, str(ROOT / "vendor/clm/src"))
        from clm import Engine
        from clm.embedder import Embedder
        base = ROOT / "models/clm-base"
        tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
        backbone = AutoModel.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa",
                                             local_files_only=True).to(device).eval()

        class HFEmbedder(Embedder):
            def __init__(self):
                super().__init__(cache_size=200_000, batch=8)

            @torch.inference_mode()
            def _fetch(self, texts):
                rows = [tok(t, add_special_tokens=False).input_ids for t in texts]
                if any(len(r) > 2048 for r in rows):
                    raise ValueError("CLM encoder input exceeds its 2048-token contract")
                # Left padding makes the final token the true last token in every row, as vLLM LAST pooling.
                width = max(map(len, rows))
                ids = torch.tensor([[tok.pad_token_id]*(width-len(r)) + r for r in rows], device=device)
                mask = torch.tensor([[0]*(width-len(r)) + [1]*len(r) for r in rows], device=device)
                hidden = backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
                vectors = F.normalize(hidden[:, -1].float(), dim=-1).cpu().numpy()
                return list(vectors), sum(map(len, rows))

        self.engine = Engine(embedder=HFEmbedder(), checkpoint=str(ROOT / "models/clm-heads/CLM_v0.1-8B.pt"),
                             device=device, action_cache=0)
        self.backbone = backbone
        self.metadata = {"device": str(next(backbone.parameters()).device), "dtype": "bfloat16 encoder / float32 native heads",
                         "backend": "HF Qwen3 last-token embedding adapter + native CLM heads/schema",
                         "encoder_cache": True, "projection_cache": False, "max_len": 2048,
                         "note": "Cached repeated states/options included in overall median; fresh first-order timing reported separately."}

    def answer(self, state, questions):
        return self.engine.answer(state, questions)


class LevRuntime:
    def __init__(self, device):
        sys.path.insert(0, str(ROOT / "vendor/lev/packages/lev/src"))
        import lev
        self.engine = lev.load(str(ROOT / "models/lev"))
        # The author's loader selects CUDA or CPU. Only device placement is extended here.
        self.engine.model.to(device)
        if self.engine.mode_b_head is not None:
            self.engine.mode_b_head.to(device)
        self.engine._device = next(self.engine.model.parameters()).device
        self.metadata = {"device": str(self.engine._device), "dtype": "bfloat16",
                         "backend": "native Lev decision engine / PyTorch with MPS placement adapter",
                         "note": "Author loader targets CUDA/CPU; adapter moves unchanged model/head to MPS. No CPU fallback."}

    def answer(self, state, questions):
        result = self.engine.system_one(state, questions).model_dump(mode="json")
        if next(self.engine.model.parameters()).device.type != "mps":
            raise RuntimeError("Lev device changed during inference")
        return result


class JuliaRuntime:
    def __init__(self, device):
        sys.path.insert(0, str(ROOT / "models/julia"))
        from julia import load_model
        self.engine = load_model(str(ROOT / "models/julia"), device=device, backend="torch",
                                 strict_encoding=True, max_length=8192, head_length=2048)
        # Native FastEngine transfers packed batches only for CUDA. Extend device placement,
        # retaining its original lossless encoding, trained head, and forward computation.
        original_pack = self.engine._pack
        self.engine._pack = lambda encoded: {k:v.to(self.engine.device) for k,v in original_pack(encoded).items()}
        self.metadata = {"device": str(self.engine.device), "dtype": "float32",
                         "backend": "native Julia / PyTorch with MPS batch placement adapter",
                         "max_len": 8192, "head_max_len": 2048}

    def answer(self, state, questions):
        rows = [{"state": state, "question": q["instructions"], "type": "choice",
                 "options": list(q["criteria"].values())} for q in questions.values()]
        if any(q["type"] != "choice" for q in questions.values()):
            raise ValueError("This Julia benchmark adapter exposes choice questions")
        encoding = self.engine.encoding_info(rows)
        result = self.engine.predict(state=state, questions=questions)
        # Julia publishes maximum probability. Add explicitly identified concentration for this client.
        for answer in result["answers"].values():
            probs = answer["probabilities"]
            n = len(probs)
            answer["confidence"] = (n * max(probs.values()) - 1) / (n - 1)
            answer["confidence_source"] = "adapter normalized maximum probability; not native Julia confidence"
        result["usage"] = {"input_tokens": sum(row["tokens"] for row in encoding), "output_tokens": 0,
                           "truncated": False, "encoding": encoding}
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    runtimes = {"laya": LayaRuntime, "laya-multilingual": MultilingualLayaRuntime, "julia": JuliaRuntime,
                "clm": CLMRuntime, "lev": LevRuntime}
    parser.add_argument("--kind", choices=tuple(runtimes), required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=11436)
    args = parser.parse_args()
    import torch
    from fastapi import FastAPI, HTTPException
    import uvicorn
    torch.set_num_threads(4)
    runtime = runtimes[args.kind](args.device)
    runtime.metadata.update({"model": args.model, "torch": torch.__version__,
                             "transformers": importlib.metadata.version("transformers")})
    print(json.dumps(runtime.metadata), flush=True)
    app, lock = FastAPI(), threading.Lock()

    @app.get("/v1/models")
    def models():
        return {"models": [runtime.metadata]}

    @app.post("/v1/systemone")
    def systemone(body: dict):
        if body.get("model") != args.model:
            raise HTTPException(404, "Unknown model")
        try:
            with lock:
                result = runtime.answer(body["state"], body["questions"])
            result["native_model"] = result.get("model")
            result["model"] = args.model
            result["runtime"] = runtime.metadata
            return result
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
