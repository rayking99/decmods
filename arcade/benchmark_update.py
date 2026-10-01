"""Fresh paired model probes of the modern, pinned BlockStar source scenarios."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from benchmark import Decision, ROOT as ARCADE, atomic, public

ROOT = ARCADE / "updated"


def load_results():
    path = ROOT / "results.json"
    if path.exists():
        return json.loads(path.read_text())
    path = ARCADE.parent / "docs/data/blockstar-update-results.json.gz"
    if path.exists():
        with gzip.open(path,"rt",encoding="utf-8") as stream:
            return json.load(stream)
    return None


def summarize(model):
    cases = {}
    for name in ("updated-default","large-yard"):
        trials = [t for t in model["trials"] if t["case"] == name]
        valid = [t for t in trials if "decision" in t]
        first = [t["decision"]["latency_ms"] for t in valid if t["order"] == 0]
        repeated = [t["decision"]["latency_ms"] for t in valid if t["order"] == 1]
        cases[name] = {"trials":len(trials),"valid":len(valid),"errors":len(trials)-len(valid),
                       "agreement_pct":round(100*sum(t["correct"] for t in valid)/len(trials),1) if trials else None,
                       "median_ms":statistics.median(first) if first else None,
                       "repeated_median_ms":statistics.median(repeated) if repeated else None,
                       "candidate_count":valid[0]["decision"]["candidate_count"] if valid else None,
                       "choices":[t.get("choice") for t in trials]}
    return cases


def run(args):
    from blockstar_update import BlockStar, build_updated_suite
    suite_file = ROOT / "suite.json"
    if not suite_file.exists():
        cases = build_updated_suite()
        for case in cases:
            case["accepted"] = sorted(case["accepted"])
        atomic(suite_file,cases)
    suite = json.loads(suite_file.read_text())
    suite_hash = hashlib.sha256(suite_file.read_bytes()).hexdigest()
    source = json.loads((ARCADE / "blockstar-update-source.json").read_text())
    results = load_results() or {"version":1,"source_commit":source["commit"],"suite_sha256":suite_hash,
        "started":datetime.now(timezone.utc).isoformat(),"status":"running","hardware":"Apple M3 Max, 128 GB unified memory",
        "method":"Two official initial scenarios from the pinned modern BlockStar source, each in two option orders. Every legal slide reaches the native model through shared 20-choice tournaments; no pruning or fallback. One-step declared-policy agreement; these probes do not establish full-puzzle solve rates. Loading and a distinct small-fixture warmup excluded. First-order full-decision timing and repeated-order timing are reported separately, with only one timing sample per scenario/order. Serialized inference; native precision and layouts differ.",
        "models":[],"unavailable":[{"name":"Jev-Omni","reason":"Official loader requires CUDA; no Mac substitution."}]}
    assert results["suite_sha256"] == suite_hash
    model = next((m for m in results["models"] if m["name"] == args.model),None)
    if model and model["status"] == "complete":
        return
    if not model:
        model = {"name":args.model,"label":args.label,"status":"running","metadata":json.loads(Path(args.metadata).read_text()),"trials":[]}
        results["models"].append(model)
    results["status"] = "running"
    client = Decision(args.model,args.endpoint,BlockStar.policy,args.wire_model)

    def publish(active=None):
        results.update(active=active,updated=datetime.now(timezone.utc).isoformat())
        for record in results["models"]:
            record["summary"] = summarize(record)
        atomic(ROOT / "results.json",results)
        atomic(ROOT / "summary.json",{**{k:v for k,v in results.items() if k != "models"},
            "models":[{k:v for k,v in m.items() if k in ("name","label","status","summary")} for m in results["models"]]})

    if "warmup" not in model:
        from blockstar import make_replay
        warm = make_replay(98765)
        warm.apply(warm.candidates()[0])
        model["warmup_snapshot"] = warm.snapshot()
        _,model["warmup"] = client.choose(warm)
    done = {(t["case"],t["order"]) for t in model["trials"]}
    for order in range(2):
        for case in suite:
            if (case["id"],order) in done:
                continue
            game = BlockStar.restore(case["snapshot"])
            trial = {"case":case["id"],"order":order,"accepted":case["accepted"]}
            publish(f"{args.label}: {case['id']} order {order+1}/2")
            try:
                action,trace = client.choose(game,order)
                game.apply(action)
                trial.update(action=public(action),choice=action.id,correct=action.id in case["accepted"],decision=trace,after=game.snapshot())
            except Exception as exc:
                trial.update(error=str(exc),correct=False)
            model["trials"].append(trial)
            print(args.model,case["id"],"order",order,"correct",trial["correct"],"ms",trial.get("decision",{}).get("latency_ms"),"error",trial.get("error"),flush=True)
            publish()
    model["status"] = "complete"
    publish()
    print("UPDATED MODEL COMPLETE",args.model,flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("model","label","endpoint","metadata"):
        p.add_argument("--"+flag,required=True)
    p.add_argument("--wire-model")
    run(p.parse_args())
