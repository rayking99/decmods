"""Launch installed pinned runtimes one at a time, preserving native heads/devices."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
HF = ROOT.parent / "tetris/hf"
BASE_PYTHON = ROOT.parent / "tetris/.venv/bin/python"
ENV = {**os.environ, "HF_HOME": str(HF / "cache"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}

SPECS = [
    ("nimble", "Nimble 9B · Ollama Q8", "ollama", 11434),
    ("tev1:4b", "Tev1 4B · Ollama Q8", "ollama", 11434),
    ("tev1:0.8b", "Tev1 0.8B · Ollama Q8", "ollama", 11434),
    ("winnow:e4b", "Winnow E4B · Metal Q8", "winnow", 11435),
    ("julia-1:mps", "Julia-1 · MPS FP32", "julia", 11438),
    ("laya:en-mps", "Laya English · MPS FP32", "laya", 11436),
    ("laya:multilingual-mps", "Laya multilingual · MPS FP32", "laya-multilingual", 11441),
    ("kev:4b-mlx", "Kev 4B · MLX BF16", "kev-mlx", 11439),
    ("kev:4b-mps", "Kev 4B · MPS BF16", "kev-mps", 11437),
    ("clm:8b-mps", "CLM 8B · MPS BF16", "clm", 11443),
    ("lev:4b-mps", "Lev 4B · MPS BF16", "lev", 11442),
    ("clef-flash:mlx-4bit", "Clef Flash 9B · MLX 4-bit", "clef-mlx", 11444),
]


def get(endpoint, body=None):
    request = Request(endpoint, data=json.dumps(body).encode() if body is not None else None,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def wait_ready(port, process=None, ollama=False):
    for _ in range(300):
        if process and process.poll() is not None:
            raise RuntimeError(f"Runtime exited {process.returncode}; inspect arcade/logs")
        try:
            return get(f"http://127.0.0.1:{port}/" + ("api/tags" if ollama else "v1/models"))
        except OSError:
            time.sleep(1)
    raise RuntimeError("Native runtime startup timed out")


def launch(kind, model, port):
    logfile = open(ROOT / "logs" / f"{kind}-server.log", "w")
    environment = dict(ENV)
    cwd = ROOT
    if kind == "winnow":
        environment.update(OLLAYA_HOST=f"127.0.0.1:{port}", OLLAYA_MODELS=str(HF / "models/ollaya"))
        command = [str(HF / "runtime/bin/ollaya"), "serve"]
    elif kind == "clef-mlx":
        command = [str(HF / "clef_runtime/.venv/bin/python"), str(HF / "clef_runtime/server.py"),
                   "--port", str(port)]
    elif kind.startswith("kev"):
        environment.update(KEV_BACKEND="mlx" if kind == "kev-mlx" else "torch", KEV_DTYPE="bf16", KEV_PREFIX_CACHE="4")
        cwd = HF / "vendor/kev"
        command = [str(cwd / ".venv/bin/python"), "-m", "kev.serve", "--run", str(HF / "models/kev"), "--port", str(port)]
    else:
        python = HF / ("julia_runtime/.venv/bin/python" if kind == "julia" else ".venv/bin/python")
        command = [str(python), str(HF / "model_server.py"), "--kind", kind,
                   "--device", "mps", "--model", model, "--port", str(port)]
    return subprocess.Popen(command, cwd=cwd, env=environment, stdout=logfile, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", help="Run only explicitly named aliases; default all installed variants")
    parser.add_argument("--updated-blockstar", action="store_true", help="Benchmark the separately pinned modern BlockStar scenarios")
    args = parser.parse_args()
    result_root = ROOT / "updated" if args.updated_blockstar else ROOT
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "runtime").mkdir(exist_ok=True)
    failures = []
    for model, label, kind, port in SPECS:
        if args.models and model not in args.models:
            continue
        result_file = result_root / "results.json"
        if args.updated_blockstar:
            from benchmark_update import load_results
        else:
            from benchmark import load_results
        previous = load_results() or {}
        if any(m["name"] == model and m["status"] == "complete" for m in previous.get("models", [])):
            print("SKIP complete", model, flush=True)
            continue
        process = None
        endpoint = f"http://127.0.0.1:{port}"
        try:
            print("LOADING", model, datetime.now(timezone.utc).isoformat(), flush=True)
            if kind != "ollama":
                try:
                    metadata = get(endpoint + "/v1/models")
                except OSError:
                    process = launch(kind, "kev-latest" if kind.startswith("kev") else model, port)
            metadata = wait_ready(port, process, kind == "ollama")
            if kind == "ollama":
                wire_name = model if ":" in model else model + ":latest"
                card = next(m for m in metadata["models"] if m["name"] == wire_name)
                metadata = {"models": [card], "backend": "Ollama native decision API", "version": get(endpoint + "/api/version")}
            else:
                archived_path = HF / "research" / {
                    "winnow": "winnow-e4b-runtime.json", "julia": "julia-1-mps-runtime.json",
                    "laya": "laya-runtime.json", "laya-multilingual": "laya-multilingual-mps-runtime.json",
                    "kev-mlx": "kev-4b-mlx-runtime.json", "kev-mps": "kev-mps-runtime.json",
                    "clm": "clm-8b-mps-runtime.json", "lev": "lev-4b-mps-runtime.json",
                    "clef-mlx": "clef-mlx-runtime.json"}[kind]
                archived = json.loads(archived_path.read_text())
                metadata = {**{k:v for k,v in archived.items() if k not in ("models", "loader_probe")}, **metadata,
                            "metadata_captured": datetime.now(timezone.utc).isoformat()}
                if kind == "clef-mlx":
                    card = next(m for m in metadata["models"] if m["model"] == model)
                    metadata = {**card, **metadata}
            if kind == "winnow":
                from benchmark import Decision
                from games import Game2048
                _, probe = Decision(model, endpoint, Game2048.policy).choose(Game2048(98765))
                actual = get(endpoint + "/api/ps")
                if not any(m.get("device") == "metal" and m.get("name") == model for m in actual.get("models", [])):
                    raise RuntimeError("Winnow did not report Metal acceleration")
                metadata["running_models"] = actual
                metadata["device_probe"] = probe
            # The Kev server publishes its own canonical wire alias; no family substitution.
            wire = "kev-latest" if kind.startswith("kev") else model
            meta_path = ROOT / "runtime" / (model.replace(":", "-") + ".json")
            meta_path.write_text(json.dumps(metadata, indent=2))
            command = [str(BASE_PYTHON), str(ROOT / ("benchmark_update.py" if args.updated_blockstar else "benchmark.py")), "--model", wire,
                       "--label", label, "--endpoint", endpoint, "--metadata", str(meta_path)]
            if kind.startswith("kev"):
                # Distinguish timing variants in saved records while the transport retains the native alias.
                command[command.index("--model") + 1] = model
                command += ["--wire-model", wire]
            subprocess.run(command, check=True, env=ENV)
        except Exception as exc:
            failure = {"name": model, "reason": str(exc)}
            failures.append(failure)
            print("UNAVAILABLE", json.dumps(failure), flush=True)
        finally:
            if kind == "ollama":
                try:
                    # Release only our benchmarked model, including failed trials.
                    get(endpoint + "/api/generate", {"model": model, "keep_alive": 0})
                except Exception as exc:
                    print("Model cleanup:", model, str(exc), flush=True)
            if process:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    if (result_root / "results.json").exists():
        from benchmark import atomic
        data = json.loads((result_root / "results.json").read_text())
        data["status"] = "complete" if not failures else "complete_with_unavailable"
        data["active"] = None
        data["unavailable"] += failures
        data["updated"] = datetime.now(timezone.utc).isoformat()
        atomic(result_root / "results.json", data)
        summary = json.loads((result_root / "summary.json").read_text())
        summary.update(status=data["status"], active=None, unavailable=data["unavailable"], updated=data["updated"])
        atomic(result_root / "summary.json", summary)
    print("SERIAL BENCHMARK FINISHED", flush=True)


if __name__ == "__main__":
    main()
