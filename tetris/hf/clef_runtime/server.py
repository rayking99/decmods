"""Serve the pinned native Clef MLX decision head, without a chat-model wrapper."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import threading

REPO = "mlx-community/clef-flash-4bit"
REVISION = "d9ec324f7992383bdfb7a0b4eed8b4b9d10f81be"
LOADER_SHA256 = "2a8560f0668ea00e8d1c557b3861ec9b8ee9a1398f85de9e2d1745a636c79f50"
ALIAS = "clef-flash:mlx-4bit"
HF_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RECORD = HF_ROOT / "research/mlx-community__clef-flash-4bit.json"


def verify_checkpoint(checkpoint):
    """Check every inference file against the archived, pinned HF manifest."""
    manifest = json.loads(SOURCE_RECORD.read_text())
    if manifest["id"] != REPO or manifest["sha"] != REVISION:
        raise RuntimeError("Checkpoint manifest differs from the pinned release")
    suffixes = {".json", ".safetensors", ".jinja", ".py"}
    files = {item["rfilename"]: item for item in manifest["siblings"]
             if "/" not in item["rfilename"] and Path(item["rfilename"]).suffix in suffixes}
    actual = {path.name for path in checkpoint.iterdir()
              if path.is_file() and path.suffix in suffixes}
    if actual != set(files):
        raise RuntimeError("Checkpoint inference files differ from the pinned manifest")
    hashes = {}
    for name, item in files.items():
        path = checkpoint / name
        if path.stat().st_size != item["size"]:
            raise RuntimeError(f"Pinned checkpoint size mismatch: {name}")
        digest = hashlib.sha256()
        blob = hashlib.sha1(f"blob {item['size']}\0".encode()) if "lfs" not in item else None
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
                if blob is not None:
                    blob.update(chunk)
        expected = item["lfs"]["sha256"] if blob is None else item["blobId"]
        observed = digest.hexdigest() if blob is None else blob.hexdigest()
        if observed != expected:
            raise RuntimeError(f"Pinned checkpoint content mismatch: {name}")
        hashes[name] = digest.hexdigest()
    return hashes


class ClefRuntime:
    def __init__(self, checkpoint):
        import mlx.core as mx
        if not mx.metal.is_available():
            raise RuntimeError("Clef MLX requires Metal; no CPU substitution")
        mx.set_default_device(mx.gpu)
        probe = mx.ones((2, 2)) @ mx.ones((2, 2))
        mx.eval(probe)
        if probe.tolist() != [[2.0, 2.0], [2.0, 2.0]]:
            raise RuntimeError("Metal device probe failed")
        checkpoint = Path(checkpoint).resolve()
        hashes = verify_checkpoint(checkpoint)
        loader = checkpoint / "clef_mlx.py"
        if hashes["clef_mlx.py"] != LOADER_SHA256:
            raise RuntimeError("Native Clef loader differs from the pinned source")
        config = json.loads((checkpoint / "config.json").read_text())
        quantization = config["quantization"]
        if quantization != {"group_size": 64, "bits": 4, "mode": "affine"}:
            raise RuntimeError("This runtime requires the requested 4-bit affine checkpoint")
        spec = importlib.util.spec_from_file_location("decision_lab_clef_native", loader)
        native = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = native
        spec.loader.exec_module(native)
        self.engine = native.load(checkpoint, backend="vlm", head_dtype=mx.bfloat16)
        self.mx = mx
        self.metadata = {
            "model": ALIAS, "repo": REPO, "checkpoint": REVISION,
            "native_source": f"https://huggingface.co/{REPO}/blob/{REVISION}/clef_mlx.py",
            "loader_sha256": LOADER_SHA256,
            "joint_head_sha256": hashes["joint_head.safetensors"],
            "backend": "native Clef joint schema head / MLX VLM", "device": "metal",
            "dtype": "4-bit affine backbone / bfloat16 vision and unchanged joint head",
            "quantization": quantization, "max_length": 16384, "truncation": False,
            "cache": "No request embedding or prefix cache in native Clef loader",
            "option_order": "Native encoder sorts choice IDs; exact probability ties use request order, and tournament groups can vary",
            "confidence": "Native maximum softmax probability, rounded to four decimals",
            "versions": {name: importlib.metadata.version(name) for name in
                         ("mlx", "mlx-vlm", "transformers", "huggingface-hub", "fastapi", "uvicorn")},
        }

    def answer(self, request):
        if self.mx.default_device() != self.mx.gpu:
            raise RuntimeError("Clef MLX device changed")
        # Preserve both the native typed response and complete state/schema.
        result = self.engine.systemone(request, max_length=16384, truncate=False)
        result["usage"]["truncated"] = False
        result["runtime"] = self.metadata
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=HF_ROOT / "models/clef-flash-4bit")
    parser.add_argument("--port", type=int, default=11444)
    args = parser.parse_args()
    from fastapi import FastAPI, HTTPException
    import uvicorn
    runtime = ClefRuntime(args.checkpoint)
    print(json.dumps(runtime.metadata), flush=True)
    app, lock = FastAPI(), threading.Lock()

    @app.get("/v1/models")
    def models():
        return {"models": [runtime.metadata]}

    @app.post("/v1/systemone")
    def systemone(body: dict):
        if body.get("model") != ALIAS:
            raise HTTPException(404, "Unknown model")
        try:
            with lock:
                return runtime.answer(body)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
