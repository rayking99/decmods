"""Publish a small dashboard snapshot without repeatedly serving raw traces."""
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent / "benchmarks"


def public_summary(results):
    public = {k: v for k, v in results.items() if k != "models"}
    public["models"] = []
    for model in results["models"]:
        item = {k:model[k] for k in ("name","status","summary","cold_start_ms") if k in model}
        if "original_format_probe" in model:
            item["original_format_probe"] = {k:v for k,v in model["original_format_probe"].items() if k != "trace"}
        public["models"].append(item)
    return public


if __name__ == "__main__":
    last = None
    while True:
        path = ROOT / "results.json"
        modified = path.stat().st_mtime_ns if path.exists() else None
        if modified is not None and modified != last:
            results = json.loads(path.read_text())
            temporary = ROOT / "summary.tmp"
            temporary.write_text(json.dumps(public_summary(results),indent=2))
            temporary.replace(ROOT / "summary.json")
            last = modified
            if results["status"] == "complete":
                break
        time.sleep(2)
