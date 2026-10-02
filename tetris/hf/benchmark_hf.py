"""Run native HF decision APIs against the unchanged frozen Ollama Tetris protocol."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmark import (BenchmarkDecision, atomic_json, dominated, features, restore, summarize)
from game import Game


def run(args):
    output = ROOT / "benchmarks/huggingface"
    output.mkdir(exist_ok=True)
    suite_path = ROOT / "benchmarks/suite.json"
    suite = json.loads(suite_path.read_text())
    assert len(suite) == len({case["id"] for case in suite}) == 32
    result_path = output / "results.json"
    if result_path.exists():
        results = json.loads(result_path.read_text())
    else:
        baseline = json.loads((ROOT / "benchmarks/results.json").read_text())
        results = {k:v for k,v in baseline.items() if k not in ("models", "active", "updated", "status", "scope")}
        results.update({"version":2, "started":datetime.now(timezone.utc).isoformat(),
                        "scope":[], "models":[], "baseline_path":"../results.json",
                        "method":baseline["method"] + " Native model prompt layouts and precisions differ; "
                                 "same complete state, instructions, options and order at the API boundary. "
                                 "Lossless input required. Calibration/confidence is not compared."})
    assert results["suite_sha256"] == hashlib.sha256(suite_path.read_bytes()).hexdigest()
    # This runner implements the original frozen protocol, including its 26-
    # choice groups. Refuse drift rather than publishing mismatched paired data.
    assert results["orders"] == 2
    assert results["game_seeds"] == [42, 123]
    assert results["game_piece_limit"] == 40
    existing = next((m for m in results["models"] if m["name"] == args.model),None)
    if existing and existing["status"] == "complete":
        print("Already complete",args.model); return
    metadata = json.loads(Path(args.metadata).read_text())
    if args.model == "clef-flash:mlx-4bit":
        # Retain the complete readiness response while exposing this explicit
        # runtime's provenance to the compact site exporter.
        cards = metadata.get("models", [])
        card = next((card for card in cards if card.get("model", card.get("name")) == args.model), {})
        for key in ("repo", "checkpoint", "native_source", "backend", "device", "dtype",
                    "quantization", "mlx", "mlx_lm", "cache", "prompt_layout", "option_order",
                    "confidence", "versions", "loader_sha256", "joint_head_sha256", "max_length", "truncation"):
            if key in card and key not in metadata:
                metadata[key] = card[key]
    model = existing or {"name":args.model,"label":args.label,"metadata":metadata,"trials":[],"games":[],"status":"running"}
    if existing and not model["trials"] and "warmup" not in model:
        model["metadata"] = metadata
    if not existing:
        model["protocol"] = {
            "name": "tetris-frozen-v1", "suite_sha256": results["suite_sha256"],
            "suite_cases": 32, "orders": 2, "max_choices_per_call": 26,
            "grouping": "More than 26 legal landings: two ordered halves, then the two native winners.",
            "trial_seed": "9000 + case_index * 101 + order * 100000",
            "game_seeds": [42, 123], "game_piece_limit": 40,
            "warmup": "Distinct Game(98765) with current piece O; loading and warmup excluded.",
            "probabilities": "Native choice distribution; confidence concentration is not compared.",
            "option_order": metadata.get("option_order", "Shuffled API option order; native layouts may differ."),
        }
        results["models"].append(model);results["scope"].append(args.model)
    results["status"] = "running"
    client = BenchmarkDecision(args.model,args.endpoint,timeout=300)

    def publish(active=None):
        results["active"] = active
        results["updated"] = datetime.now(timezone.utc).isoformat()
        model["summary"] = summarize(model)
        atomic_json(result_path,results)
        summary = {k:v for k,v in results.items() if k != "models"}
        summary["models"] = [{k:v for k,v in m.items() if k in ("name","label","status","metadata","summary","cold_start_ms")}
                             for m in results["models"]]
        atomic_json(output/"summary.json",summary)

    publish(f"{args.label}: warmup")
    if "warmup" not in model:
        warm=Game(98765);warm.queue[0]="O"
        model["warmup_snapshot"] = copy.deepcopy(warm.snapshot())
        start=time.perf_counter()
        _,trace=client.choose(warm)
        model["first_request_ms"]=round((time.perf_counter()-start)*1000,1)
        model["warmup"]=trace
    complete={(t["case"],t["order"]) for t in model["trials"]}
    for order in range(2):
        for index,case in enumerate(suite):
            if (case["id"],order) in complete:continue
            game=restore(case["snapshot"]);game.seed=9000+index*101+order*100000
            record={"case":case["id"],"order":order,"accepted":case["accepted"]}
            try:
                move,trace=client.choose(game)
                if trace["source"] != "forced":
                    trace["source"]="native-decision-api"
                record.update(choice=move.id,correct=move.id in case["accepted"],
                              dominated=dominated(move,game.candidates()),move=move.public(),decision=trace)
            except Exception as exc:
                record.update(error=str(exc),correct=False)
            model["trials"].append(record)
            publish(f"{args.label}: board {index+1}/32, order {order+1}/2")
            print(f"{args.label} {case['id']} order={order} correct={record['correct']} "
                  f"ms={record.get('decision',{}).get('latency_ms')} error={record.get('error')}",flush=True)
    for seed in (42,123):
        record=next((g for g in model["games"] if g["seed"]==seed),None)
        if record and record["status"]!="running":continue
        game=Game(seed)
        if record:
            for turn in record["moves"]:
                game.apply(next(m for m in game.candidates() if m.id==turn["move"]["id"]))
        else:
            record={"seed":seed,"moves":[],"status":"running","pieces":0,"lines":0,"score":0,"holes":0,"max_height":0}
            model["games"].append(record)
        while game.pieces<40 and game.candidates():
            try:
                move,trace=client.choose(game)
                if trace["source"] != "forced":
                    trace["source"]="native-decision-api"
                game.apply(move)
                record["moves"].append({"move":move.public(),"decision":trace,"after":copy.deepcopy(game.snapshot())})
                record.update(pieces=game.pieces,lines=game.lines,score=game.score,
                              holes=features(game.board)["holes"],max_height=features(game.board)["max_height"])
            except Exception as exc:
                record.update(status="error",error=str(exc));break
            publish(f"{args.label}: game {seed}, piece {game.pieces}/40")
            print(f"{args.label} seed={seed} piece={game.pieces} lines={game.lines} ms={trace['latency_ms']}",flush=True)
        if record["status"]!="error":record["status"]="piece_limit" if game.pieces>=40 else "game_over"
        publish()
    model["status"]="complete"
    results["status"]="complete"
    publish()
    print(json.dumps(model["summary"],indent=2),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--metadata", required=True)
    run(p.parse_args())


if __name__=="__main__":main()
