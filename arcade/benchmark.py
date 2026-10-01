"""Auditable, serialized native decision-model benchmark across three games."""
import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import gzip
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "tetris"))
from decision import OllamaDecision


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    temporary.replace(path)


def public(action):
    return asdict(action) if is_dataclass(action) else action.public()


def load_results():
    path = ROOT / "results.json"
    if path.exists():
        return json.loads(path.read_text())
    archived = ROOT.parent / "docs/data/arcade-results.json.gz"
    if archived.exists():
        with gzip.open(archived, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return None


class WireAction:
    def __init__(self, action):
        self.action, self.id = action, action.id

    def public(self):
        return public(self.action)


class Decision(OllamaDecision):
    """All legal actions reach a model; tournaments use a shared 20-choice limit."""
    def __init__(self, model, endpoint, policy, wire_model=None):
        self.saved_model = model
        super().__init__(wire_model or model, endpoint, timeout=300, policy=policy,
                         describe_option=lambda action: action.action.text)

    def choose(self, game, order=0):
        candidates = game.candidates()
        if not candidates:
            raise ValueError("No legal actions")
        if len(candidates) == 1:
            move = candidates[0]
            return move, {"source": "forced", "choice": move.id, "latency_ms": 0,
                          "candidate_count": 1, "calls": [], "all_options": [public(move)]}
        seed = int(hashlib.sha256(json.dumps(game.snapshot(), sort_keys=True).encode()).hexdigest()[:16], 16)
        ordered = [WireAction(c) for c in candidates]
        random.Random(seed + order).shuffle(ordered)
        calls = []
        observation = game.observation()
        # Recursive, lossless coverage for BlockStar's many legal slide distances.
        while len(ordered) > 20:
            winners = []
            for begin in range(0, len(ordered), 20):
                group = ordered[begin:begin + 20]
                if len(group) == 1:
                    winners.append(group[0])
                    continue
                call = self.request(observation, group)
                calls.append(call)
                winners.append(next(a for a in group if a.id == call["answer"]["choice"]))
            ordered = winners
        call = self.request(observation, ordered)
        calls.append(call)
        answer = call["answer"]
        move = next(a for a in candidates if a.id == answer["choice"])
        return move, {"source": "native-decision-api", "model": self.saved_model, "choice": move.id,
                      "latency_ms": round(sum(c["latency_ms"] for c in calls), 1),
                      "candidate_count": len(candidates), "calls": calls,
                      "probabilities": answer["probabilities"],
                      "all_options": [public(a) for a in candidates]}


def registry():
    from games import Game2048, ConnectFour
    from blockstar import BlockStar
    return {"2048": Game2048, "connect4": ConnectFour, "blockstar": BlockStar}


def prepare_suite():
    from games import build_suite as game_suite
    from blockstar import build_suite as block_suite
    path = ROOT / "suite.json"
    if not path.exists():
        cases = game_suite() + block_suite()
        for case in cases:
            case["game"] = {"Connect Four": "connect4"}.get(case["game"], case["game"])
            case["accepted"] = sorted(case["accepted"])
        atomic(path, cases)
    return json.loads(path.read_text()), hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(model):
    games = {}
    for name in registry():
        trials = [t for t in model["trials"] if t["game"] == name]
        valid = [t for t in trials if "decision" in t]
        warm_calls = model.get("warmups", {}).get(name, {}).get("calls", [])
        warm_state = warm_calls[0]["request"]["state"] if warm_calls else None
        # Exclude the warmup fixture's timed repeat from the principal first-order metric.
        fresh = [t["decision"]["latency_ms"] for t in valid if t["order"] == 0
                 and t["decision"]["source"] != "forced"
                 and (not warm_calls or t["decision"]["calls"][0]["request"]["state"] != warm_state)]
        repeated = [t["decision"]["latency_ms"] for t in valid if t["order"] == 1
                    and t["decision"]["source"] != "forced"]
        outcomes = [{k: v for k, v in record.items() if k not in ("moves", "initial")}
                    for record in model["games"] if record["game"] == name]
        sorted_latency = sorted(fresh)
        games[name] = {"trials": len(trials), "valid": len(valid),
                       "errors": len(trials) - len(valid),
                       "agreement_pct": round(100 * sum(t["correct"] for t in valid) / len(trials), 1) if trials else None,
                       "median_ms": round(statistics.median(fresh), 1) if fresh else None,
                       "p95_ms": sorted_latency[max(0, math.ceil(.95 * len(sorted_latency)) - 1)] if fresh else None,
                       "repeated_median_ms": round(statistics.median(repeated), 1) if repeated else None,
                       "timing_samples": len(fresh), "outcomes": outcomes,
                       "timing_note": "Principal median/P95 use first-order trials excluding states identical to warmup. Repeated-order median is separate. Native caches can still reuse options and prefixes.",
                       "policy": registry()[name].policy}
        if name == "blockstar":
            original = [t for t in trials if t["case"] == "original-20x18"]
            original_valid = [t for t in original if "decision" in t]
            original_first = [t["decision"]["latency_ms"] for t in original_valid if t["order"] == 0]
            games[name].update(original_trials=len(original), original_valid=len(original_valid),
                original_agreement_pct=round(100 * sum(t["correct"] for t in original_valid) / len(original),1) if original else None,
                original_median_ms=round(statistics.median(original_first),1) if original_first else None,
                original_note="One original 20×18 board in two orders; timing uses the first order. Remaining seven boards and capped replay are smaller fixtures. No full original puzzle solve by these models is claimed.")
    return games


def run(args):
    suite, suite_hash = prepare_suite()
    path = ROOT / "results.json"
    saved_results = load_results()
    if saved_results is not None:
        results = saved_results
        if results["suite_sha256"] != suite_hash:
            raise ValueError("Frozen suite changed; preserve old results before starting a new suite")
    else:
        results = {"version": 1, "started": datetime.now(timezone.utc).isoformat(),
                   "hardware": "Apple M3 Max, 128 GB unified memory", "suite_sha256": suite_hash,
                   "orders": 2, "game_seeds": [42], "limits": {"2048": 24, "connect4": 12, "blockstar": 16},
                   "method": "Eight frozen states per game, two deterministic option orders. Every legal action reaches the native decision model. Shared 20-choice group tournament when needed, accepting all policy ties. No heuristic fallback. Serialized inference; warmup excluded. Median/P95 use first-order trials; repeated cache timings reported separately. One capped game per model/game, same initial states and seeded randomness; trajectories can diverge. Rules and observations provide derived one-step/search metrics, so agreement measures supplied-policy following, not unaided planning or globally optimal play.",
                   "models": [], "unavailable": [{"name": "Jev-Omni", "reason": "Official native loader requires CUDA; no Mac result or family substitution."}]}
    model = next((m for m in results["models"] if m["name"] == args.model), None)
    if model and model["status"] == "complete":
        print("Already complete:", args.model, flush=True)
        return
    if not model:
        model = {"name": args.model, "label": args.label,
                 "metadata": json.loads(Path(args.metadata).read_text()),
                 "endpoint": args.endpoint, "trials": [], "games": [], "warmups": {}, "status": "running"}
        results["models"].append(model)
    results["status"] = "running"

    def publish(active=None):
        results["active"] = active
        results["updated"] = datetime.now(timezone.utc).isoformat()
        model["summary"] = summarize(model)
        atomic(path, results)
        atomic(ROOT / "summary.json", {**{k:v for k,v in results.items() if k != "models"},
            "models": [{"name":m["name"], "label":m["label"], "status":m["status"],
                        "metadata":m["metadata"], "games":m["summary"]} for m in results["models"]]})

    completed = {(t["case"], t["order"]) for t in model["trials"]}
    for name, cls in registry().items():
        client = Decision(args.model, args.endpoint, cls.policy, args.wire_model)
        cases = [c for c in suite if c["game"] == name]
        if name not in model["warmups"]:
            publish(f"{args.label}: {name} warmup")
            # Use a small suite case so an original BlockStar warmup does not dominate setup.
            warm = cls.restore(cases[-1]["snapshot"])
            _, trace = client.choose(warm)
            model["warmups"][name] = trace
        # Complete each order before the next: the cache distinction remains explicit.
        for order in range(2):
            for case in cases:
                if (case["id"], order) in completed:
                    continue
                game = cls.restore(case["snapshot"])
                record = {"game": name, "case": case["id"], "order": order, "accepted": case["accepted"]}
                publish(f"{args.label}: {name} {case['id']} order {order + 1}/2")
                try:
                    action, trace = client.choose(game, order)
                    record.update(choice=action.id, correct=action.id in case["accepted"],
                                  action=public(action), decision=trace)
                except Exception as exc:
                    record.update(error=str(exc), correct=False)
                model["trials"].append(record)
                print(f"{args.model} {name} {case['id']} order={order} correct={record['correct']} ms={record.get('decision',{}).get('latency_ms')} error={record.get('error')}", flush=True)
                publish()
        record = next((g for g in model["games"] if g["game"] == name), None)
        if record and record["status"] != "running":
            continue
        if record:
            game = cls.restore(record["moves"][-1]["after"] if record["moves"] else record["initial"])
        else:
            if name == "blockstar":
                from blockstar import make_replay
                game = make_replay(42)
            else:
                game = cls(42)
            record = {"game": name, "seed": 42, "initial": game.snapshot(), "moves": [], "status": "running"}
            model["games"].append(record)
        while not game.done and len(record["moves"]) < results["limits"][name]:
            publish(f"{args.label}: {name} game turn {len(record['moves']) + 1}")
            try:
                action, trace = client.choose(game)
                game.apply(action)
                record["moves"].append({"action": public(action), "decision": trace, "after": game.snapshot()})
            except Exception as exc:
                record.update(status="error", error=str(exc))
                break
            publish()
        final = game.snapshot()
        record.update({k:v for k,v in final.items() if k not in ("game", "board", "rng", "rng_state", "target", "pieces", "order", "last_move", "seed")})
        record["turn"] = game.turn
        if record.get("error"):
            record["status"] = "error"
        elif game.done:
            record["status"] = final.get("status", "finished")
        else:
            record["status"] = "turn_limit"
        print(f"{args.model} {name} game {record['status']} turns={game.turn}", flush=True)
        publish()
    model["status"] = "complete"
    publish()
    print("MODEL COMPLETE", args.model, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--wire-model", help="Explicit native transport alias, saved alongside the benchmark variant")
    run(parser.parse_args())
