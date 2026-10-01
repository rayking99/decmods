"""Serialized, paired Tetris benchmark for every current Ollama decision model."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from urllib.request import Request, urlopen

from decision import OllamaDecision
from game import Game, HEIGHT, WIDTH, features

ROOT = Path(__file__).resolve().parent
MODELS = ("nimble", "tev1:4b", "tev1:0.8b")
BENCH_POLICY = (
    "Choose the legal Tetris landing by this exact priority rule: "
    "first minimize holes; among ties maximize lines cleared; among ties minimize peak height; "
    "then minimize total height; then minimize roughness. Any remaining tie is acceptable. "
    "Compare option outcomes AFTER row clearing. This rule is the evaluation target."
)


def compact_description(move):
    f = move.metrics
    return (f"rotation={move.rotation}, column={move.x+1}, lines={move.cleared}, "
            f"holes={f['holes']}, peak={f['max_height']}, "
            f"total_height={f['aggregate_height']}, roughness={f['bumpiness']}")


def compact_state(state):
    return {"piece": state["current_piece"], "next": state["next_piece"],
            "board_top_to_bottom": state["board_top_to_bottom"],
            "definitions": "holes=empty cells below blocks; peak=max column height; "
                           "total_height=sum of column heights; roughness=sum of adjacent height differences. "
                           "All options are legal hard drops. Metrics are after line clears."}


class BenchmarkDecision(OllamaDecision):
    def __init__(self, model, endpoint, timeout=120):
        super().__init__(model, endpoint, timeout, BENCH_POLICY, compact_description)

    def request(self, state, options):
        return super().request(compact_state(state), options)


def reference_key(move):
    f = move.metrics
    return (f["holes"], -move.cleared, f["max_height"], f["aggregate_height"], f["bumpiness"])


def acceptable_moves(moves):
    best = min(reference_key(m) for m in moves)
    return [m.id for m in moves if reference_key(m) == best]


def dominated(selected, candidates):
    target = reference_key(selected)
    return any(all(a <= b for a, b in zip(reference_key(other), target))
               and any(a < b for a, b in zip(reference_key(other), target)) for other in candidates)


def restore(snapshot):
    game = Game(snapshot["seed"])
    game.board = copy.deepcopy(snapshot["board"])
    game.queue = [snapshot["current"]] + snapshot["next"]
    game.pieces, game.lines, game.score = snapshot["pieces"], snapshot["lines"], snapshot["score"]
    return game


def make_suite():
    cases = []
    # Actual observed model play: choose the longest saved run, at fixed intervals.
    replays = [(p, json.loads(p.read_text())) for p in (ROOT / "runs").glob("*.json")]
    if replays:
        path, replay = max(replays, key=lambda item: len(item[1]["decisions"]))
        snapshots = [Game(replay["seed"]).snapshot()] + [e["after"] for e in replay["decisions"]]
        eligible = [s for s in snapshots if len(restore(s).candidates()) > 1]
        for index in range(min(12, len(eligible))):
            location = round(index * (len(eligible) - 1) / 11) if len(eligible) >= 12 else index
            cases.append({"id": f"observed-{index:02}", "source": str(path.relative_to(ROOT)),
                          "snapshot": eligible[location]})
    # Independently generated positions from three deterministic reference-policy games.
    for seed in (11, 29, 73):
        game = Game(seed)
        for turn in range(29):
            moves = game.candidates()
            if len(moves) < 2:
                break
            if turn in (7, 14, 21, 28):
                cases.append({"id": f"reference-{seed}-{turn}", "source": "reference-policy trajectory",
                              "snapshot": copy.deepcopy(game.snapshot())})
            game.apply(min(moves, key=reference_key))
    # Controlled challenge boards, including a four-line clear and buried-hole risks.
    profiles = ([4]*9+[0], [0,2,2,4,4,1,1,3,3,0], [6,5,4,3,2,1,0,1,2,3],
                [8,8,8,0,8,8,8,8,8,8])
    for i, heights in enumerate(profiles):
        for piece in ("I", "T"):
            game = Game(300 + i)
            game.queue[0] = piece
            for x, height in enumerate(heights):
                for y in range(HEIGHT-height, HEIGHT):
                    game.board[y][x] = "J"
            if i == 1:
                game.board[19][3] = 0
            cases.append({"id": f"challenge-{i}-{piece}", "source": "controlled skyline / hole fixture",
                          "snapshot": game.snapshot()})
    for case in cases:
        moves = restore(case["snapshot"]).candidates()
        case["accepted"] = acceptable_moves(moves)
        case["candidate_count"] = len(moves)
    assert len(cases) == 32
    return cases


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    p = (len(ordered)-1)*q
    lo, hi = math.floor(p), math.ceil(p)
    return ordered[lo]*(hi-p) + ordered[hi]*(p-lo) if lo != hi else ordered[lo]


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def summarize(model):
    trials = model["trials"]
    successes = [t for t in trials if not t.get("error")]
    latencies = [t["decision"]["latency_ms"] for t in successes]
    calls = [c["latency_ms"] for t in successes for c in t["decision"]["calls"]]
    input_tokens = [c["response"]["usage"]["input_tokens"] for t in successes for c in t["decision"]["calls"]]
    games = model["games"]
    n = len(trials)
    return {"trials": n, "valid": len(successes), "errors": n-len(successes),
            "agreement_pct": round(100*sum(t.get("correct", False) for t in trials)/n, 1) if n else None,
            "nondominated_pct": round(100*sum(not t.get("dominated", True) for t in trials)/n, 1) if n else None,
            "mean_placement_ms": round(statistics.mean(latencies),1) if latencies else None,
            "median_placement_ms": round(statistics.median(latencies),1) if latencies else None,
            "p95_placement_ms": round(quantile(latencies,.95),1) if latencies else None,
            "median_call_ms": round(statistics.median(calls),1) if calls else None,
            "api_calls": len(calls), "max_input_tokens": max(input_tokens) if input_tokens else None,
            "games": [{k:g[k] for k in ("seed","pieces","lines","score","status","holes","max_height")}
                      for g in games],
            "total_lines": sum(g["lines"] for g in games),
            "total_pieces": sum(g["pieces"] for g in games)}


class Runner:
    def __init__(self, args):
        self.args = args
        self.output = ROOT / "benchmarks"
        self.output.mkdir(exist_ok=True)
        suite_path = self.output / "suite.json"
        if args.new_suite or not suite_path.exists():
            atomic_json(suite_path, make_suite())
        self.suite = json.loads(suite_path.read_text())
        self.result_path = self.output / "results.json"
        if args.resume and self.result_path.exists():
            self.results = json.loads(self.result_path.read_text())
            assert self.results["suite_sha256"] == hashlib.sha256(suite_path.read_bytes()).hexdigest()
            assert self.results["orders"] == args.orders
            assert self.results["game_seeds"] == args.game_seeds
            assert self.results["game_piece_limit"] == args.game_pieces
            assert self.results["scope"] == args.models
        else:
            self.results = {"version": 1, "started": datetime.now(timezone.utc).isoformat(),
                "hardware": "Apple M3 Max, 128 GB unified memory", "ollama": "0.35.0",
                "scope": list(args.models), "suite_cases": len(self.suite), "orders": args.orders,
                "game_seeds": args.game_seeds, "game_piece_limit": args.game_pieces,
                "suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
                "policy": BENCH_POLICY, "description_encoding": "compact-v1",
                "method": "Same 32 frozen boards and deterministic option-order permutations per model. "
                          "Agreement is with a supplied one-step lexicographic policy, accepting all ties; "
                          "not optimal Tetris accuracy. Serialized inference, loaded-model trials after warmup. "
                          "Identical seeds for separate games; outcomes may diverge. No animation delays. "
                          "Prompt-cache effects are possible. Group tournament above 26 choices.",
                "status": "running", "active": None, "models": []}

    def http(self, path, payload=None):
        req = Request(self.args.endpoint+path, data=json.dumps(payload).encode() if payload is not None else None,
                      headers={"Content-Type":"application/json"})
        with urlopen(req, timeout=120) as response:
            return json.load(response)

    def publish(self):
        for model in self.results["models"]:
            model["summary"] = summarize(model)
        self.results["updated"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.result_path, self.results)

    def run(self):
        # Pause the demo and wait until its in-flight decision reaches the move boundary.
        try:
            self.http_url = self.args.demo_url
            req = Request(self.args.demo_url+"/api/pause", data=b"{}", headers={"Content-Type":"application/json"})
            with urlopen(req, timeout=10) as response:
                json.load(response)
            while True:
                with urlopen(self.args.demo_url+"/api/state", timeout=10) as response:
                    state = json.load(response)
                if state["phase"] != "thinking":
                    break
                time.sleep(.5)
        except OSError as exc:
            self.results["demo_pause_note"] = str(exc)
        tags = self.http("/api/tags")["models"]
        for name in self.args.models:
            model = next((m for m in self.results["models"] if m["name"] == name), None)
            if model is None:
                metadata = next(t for t in tags if t["name"] in (name,name+":latest"))
                model = {"name":name, "metadata":metadata, "trials":[], "games":[], "status":"running"}
                self.results["models"].append(model)
            if model["status"] == "complete":
                continue
            client = BenchmarkDecision(name, self.args.endpoint)
            self.results["active"] = f"{name}: unload / warmup"
            self.publish()
            for known in MODELS:
                self.http("/api/generate", {"model":known, "keep_alive":0})
            warm = Game(98765)
            warm.queue[0] = "O"
            start = time.perf_counter()
            move, trace = client.choose(warm)
            model["cold_start_ms"] = round((time.perf_counter()-start)*1000,1)
            model["warmup"] = trace
            # Compatibility probe using the original verbose live-game input; excluded from timings.
            if "original_format_probe" not in model:
                original = Game(42)
                original.queue[0] = "T"
                start = time.perf_counter()
                try:
                    _, native = OllamaDecision(name,self.args.endpoint).choose(original)
                    model["original_format_probe"] = {"compatible":True,"trace":native}
                except Exception as exc:
                    model["original_format_probe"] = {"compatible":False,"error":str(exc),
                                                        "latency_ms":round((time.perf_counter()-start)*1000,1)}
            complete = {(t["case"],t["order"]) for t in model["trials"]}
            for order in range(self.args.orders):
                for index, case in enumerate(self.suite):
                    if (case["id"],order) in complete:
                        continue
                    game = restore(case["snapshot"])
                    # Same shuffle for all models, changed between replicates.
                    game.seed = 9000 + index * 101 + order * 100000
                    self.results["active"] = f"{name}: board {index+1}/{len(self.suite)}, order {order+1}/{self.args.orders}"
                    record = {"case":case["id"],"order":order,"accepted":case["accepted"]}
                    try:
                        move, decision = client.choose(game)
                        record.update({"choice":move.id,"correct":move.id in case["accepted"],
                                       "dominated":dominated(move,game.candidates()),"move":move.public(),
                                       "decision":decision})
                    except Exception as exc:
                        record.update({"error":str(exc),"correct":False})
                    model["trials"].append(record)
                    self.publish()
                    print(f"{name} case={case['id']} order={order} "
                          f"correct={record['correct']} latency={record.get('decision',{}).get('latency_ms')} "
                          f"error={record.get('error')}",flush=True)
            completed_seeds = {g["seed"] for g in model["games"] if g["status"] != "running"}
            for seed in self.args.game_seeds:
                if seed in completed_seeds:
                    continue
                game = Game(seed)
                record = next((g for g in model["games"] if g["seed"] == seed), None)
                if record is None:
                    record = {"seed":seed,"moves":[],"status":"running","pieces":0,"lines":0,
                              "score":0,"holes":0,"max_height":0}
                    model["games"].append(record)
                else:
                    for turn in record["moves"]:
                        game.apply(next(m for m in game.candidates() if m.id == turn["move"]["id"]))
                while game.pieces < self.args.game_pieces and game.candidates():
                    self.results["active"] = f"{name}: game seed {seed}, piece {game.pieces+1}/{self.args.game_pieces}"
                    try:
                        move, decision = client.choose(game)
                        game.apply(move)
                        record["moves"].append({"move":move.public(),"decision":decision,"after":copy.deepcopy(game.snapshot())})
                        record.update({"pieces":game.pieces,"lines":game.lines,"score":game.score,
                                       "holes":features(game.board)["holes"],"max_height":features(game.board)["max_height"]})
                    except Exception as exc:
                        record["status"],record["error"] = "error",str(exc)
                        break
                    self.publish()
                    print(f"{name} game={seed} piece={game.pieces} lines={game.lines} "
                          f"latency={decision['latency_ms']}",flush=True)
                if record["status"] != "error":
                    record["status"] = "piece_limit" if game.pieces >= self.args.game_pieces else "game_over"
                self.publish()
            model["status"] = "complete"
            self.publish()
        self.results["status"],self.results["active"] = "complete",None
        self.publish()
        print(json.dumps([{ "model":m["name"], **m["summary"]} for m in self.results["models"]],indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models",nargs="+",default=list(MODELS))
    parser.add_argument("--endpoint",default="http://127.0.0.1:11434")
    parser.add_argument("--demo-url",default="http://127.0.0.1:8765")
    parser.add_argument("--orders",type=int,default=2)
    parser.add_argument("--game-seeds",nargs="+",type=int,default=[42,123])
    parser.add_argument("--game-pieces",type=int,default=40)
    parser.add_argument("--new-suite",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    Runner(args).run()
