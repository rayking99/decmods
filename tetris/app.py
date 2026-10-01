"""Local UI and auditable game runner; run with uv run python app.py."""
import argparse
import copy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.request import urlopen

from decision import OllamaDecision
from game import Game

ROOT = Path(__file__).resolve().parent


class Session:
    def __init__(self, model, endpoint, seed, delay, max_pieces, compact=False, provider="Ollama", metadata=None):
        self.lock = threading.RLock()
        self.wake = threading.Condition(self.lock)
        if compact:
            from benchmark import BenchmarkDecision
            self.client = BenchmarkDecision(model, endpoint)
        else:
            self.client = OllamaDecision(model, endpoint)
        self.delay, self.max_pieces = delay, max_pieces
        self.model_info = {"name": model, "provider": provider, "policy": "benchmark" if compact else "survival"}
        if metadata:
            self.model_info["runtime"] = json.loads(Path(metadata).read_text())
            loaded = self.model_info["runtime"].get("models", [])
            native = next((m for m in loaded if m.get("name",m.get("model")) == model),None)
            if native and native.get("digest"):
                self.model_info["digest"] = native["digest"]
        if not metadata and provider == "Ollama":
            try:
                with urlopen(endpoint.rstrip("/") + "/api/tags", timeout=5) as response:
                    for info in json.load(response).get("models", []):
                        if info["name"] in (model, model + ":latest"):
                            self.model_info.update(info)
            except Exception as exc:
                self.model_info["metadata_error"] = str(exc)
        self.epoch = 0
        self.reset(seed)
        threading.Thread(target=self.worker, daemon=True).start()

    def reset(self, seed):
        with self.lock:
            self.epoch += 1
            self.game = Game(seed)
            self.running, self.steps, self.phase = False, 0, "ready"
            self.error, self.pending, self.latest = None, None, None
            self.history, self.latencies = [], []
            self.started = datetime.now(timezone.utc).isoformat()
            self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            self.wake.notify_all()

    def public(self):
        with self.lock:
            return copy.deepcopy({**self.game.snapshot(), "running": self.running,
                "phase": self.phase, "error": self.error, "pending": self.pending,
                "latest": self.latest, "history": self.history[-10:], "epoch": self.epoch,
                "model": self.model_info, "mean_latency_ms": round(sum(self.latencies) / len(self.latencies), 1)
                if self.latencies else None, "run_id": self.run_id,
                "max_pieces": self.max_pieces, "delay": self.delay})

    def replay(self):
        with self.lock:
            return copy.deepcopy({"version": 1, "model": self.model_info, "started": self.started,
                "run_id": self.run_id, "mode": "placement / rotate-shift-hard-drop; no hold or wall kicks",
                "seed": self.game.seed, "final": self.game.snapshot(), "error": self.error,
                "decisions": self.history})

    def save(self):
        replay = self.replay()
        folder = ROOT / "runs"
        folder.mkdir(exist_ok=True)
        target = folder / f"{replay['run_id']}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(replay, indent=2))
        temporary.replace(target)

    def command(self, action, data):
        with self.lock:
            if action == "reset":
                self.reset(int(data.get("seed", self.game.seed)))
            elif action == "pause":
                self.running, self.steps = False, 0
                if self.phase not in ("thinking", "placing", "game_over", "error", "complete"):
                    self.phase = "paused"
            elif action in ("start", "step"):
                if self.phase in ("game_over", "complete"):
                    raise ValueError("Start a new game first")
                self.error = None
                if action == "start":
                    self.running = True
                else:
                    self.running, self.steps = False, self.steps + 1
                self.wake.notify_all()
            else:
                raise ValueError("Unknown action")

    def worker(self):
        while True:
            with self.lock:
                self.wake.wait_for(lambda: self.running or self.steps > 0)
                if self.game.pieces >= self.max_pieces:
                    self.running, self.steps, self.phase = False, 0, "complete"
                    continue
                if not self.game.candidates():
                    self.running, self.steps, self.phase = False, 0, "game_over"
                    continue
                epoch, game = self.epoch, copy.deepcopy(self.game)
                self.phase, self.error = "thinking", None
            try:
                move, decision = self.client.choose(game)
                if decision["source"] != "forced" and self.model_info["provider"] != "Ollama":
                    decision["source"] = "native-decision-api"
                with self.lock:
                    if epoch != self.epoch:
                        continue
                    self.pending, self.latest, self.phase = move.public(), decision, "placing"
                    deadline = time.monotonic() + self.delay
                    while epoch == self.epoch and time.monotonic() < deadline:
                        self.wake.wait(timeout=max(0, deadline - time.monotonic()))
                    # A pause waits at the exact move boundary; reset invalidates old inference.
                    self.wake.wait_for(lambda: epoch != self.epoch or self.running or self.steps > 0)
                    if epoch != self.epoch:
                        continue
                    self.game.apply(move)
                    self.latencies.append(decision["latency_ms"])
                    entry = {"turn": self.game.pieces, "move": move.public(),
                             "decision": decision, "after": copy.deepcopy(self.game.snapshot())}
                    self.history.append(entry)
                    self.pending = None
                    self.steps = max(0, self.steps - 1)
                    self.phase = "running" if self.running else "paused"
                    if not self.game.candidates():
                        self.running, self.steps, self.phase = False, 0, "game_over"
                    elif self.game.pieces >= self.max_pieces:
                        self.running, self.steps, self.phase = False, 0, "complete"
                    print(f"turn={self.game.pieces} piece={move.piece} move={move.id} "
                          f"lines={self.game.lines} holes={move.metrics['holes']} "
                          f"latency={decision['latency_ms']}ms source={decision['source']}", flush=True)
                self.save()
            except Exception as exc:
                with self.lock:
                    if epoch != self.epoch:
                        continue
                    self.error, self.phase, self.running, self.steps = str(exc), "error", False, 0
                print(f"Decision stopped: {exc}", flush=True)
                self.save()


def make_handler(session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, content, mime="application/json"):
            encoded = json.dumps(content).encode() if mime == "application/json" else content
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/state":
                self.send(200, session.public())
            elif path == "/api/replay":
                self.send(200, session.replay())
            elif path in ("/", "/index.html", "/app.js", "/style.css"):
                name = "index.html" if path == "/" else path[1:]
                mime = {"html": "text/html; charset=utf-8", "js": "text/javascript", "css": "text/css"}[name.split(".")[-1]]
                self.send(200, (ROOT / "web" / name).read_bytes(), mime)
            else:
                self.send(404, {"error": "Not found"})

        def do_POST(self):
            # Reject cross-origin browser writes to this loopback-only demo.
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                self.send(403, {"error": "Origin mismatch"})
                return
            if not self.path.startswith("/api/"):
                self.send(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 <= length <= 2048:
                    raise ValueError("Request too large")
                data = json.loads(self.rfile.read(length) or b"{}")
                session.command(self.path.removeprefix("/api/"), data)
                self.send(200, session.public())
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.send(400, {"error": str(exc)})
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nimble")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=.8)
    parser.add_argument("--max-pieces", type=int, default=500)
    parser.add_argument("--autostart", action="store_true")
    parser.add_argument("--benchmark-policy", action="store_true")
    parser.add_argument("--provider", default="Ollama")
    parser.add_argument("--metadata")
    args = parser.parse_args()
    session = Session(args.model, args.endpoint, args.seed, args.delay, args.max_pieces,
                      args.benchmark_policy,args.provider,args.metadata)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session))
    print(f"Decision Tetris: http://127.0.0.1:{args.port} · model={args.model} · seed={args.seed}", flush=True)
    if args.autostart:
        session.command("start", {})
    server.serve_forever()


if __name__ == "__main__":
    main()
