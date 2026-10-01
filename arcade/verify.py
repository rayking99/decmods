"""Reconstruct every benchmark choice, label, API tournament and saved game."""
import hashlib
import json
import math
from pathlib import Path
import random

from benchmark import atomic, load_results, public, registry, summarize

ROOT = Path(__file__).resolve().parent


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def audit_trace(game, trace):
    options = game.candidates()
    assert trace["candidate_count"] == len(options)
    assert normalized(trace["all_options"]) == normalized([public(a) for a in options])
    if len(options) == 1:
        assert trace["source"] == "forced" and trace["choice"] == options[0].id
        assert trace["calls"] == [] and trace["latency_ms"] == 0
        return
    assert trace["source"] == "native-decision-api"
    seen = set()
    finalists = set()
    elapsed = 0
    for call in trace["calls"]:
        request, response = call["request"], call["response"]
        assert normalized(request["state"]) == normalized(game.observation())
        q = request["questions"]["placement"]
        assert q["instructions"] == game.policy and q["type"] == "choice"
        criteria = q["criteria"]
        assert 2 <= len(criteria) <= 20
        assert set(criteria) <= {a.id for a in options}
        for action in options:
            if action.id in criteria:
                assert criteria[action.id] == action.text
        answer = response["answers"]["placement"]
        assert response["model"] == request["model"]
        assert answer == call["answer"] and answer["choice"] in criteria
        probs = answer["probabilities"]
        assert set(probs) == set(criteria)
        assert all(isinstance(v,(float,int)) and math.isfinite(v) and 0 <= v <= 1 for v in probs.values())
        assert abs(sum(probs.values()) - 1) <= .02
        assert answer["type"] == "choice"
        confidence = answer.get("confidence")
        assert isinstance(confidence,(float,int)) and math.isfinite(confidence) and 0 <= confidence <= 1
        assert isinstance(call["latency_ms"],(float,int)) and math.isfinite(call["latency_ms"]) and call["latency_ms"] >= 0
        assert not response.get("usage", {}).get("truncated", False)
        seen.update(criteria)
        finalists.add(answer["choice"])
        elapsed += call["latency_ms"]
    # One leftover singleton may advance without a call; verify groups in full below.
    assert trace["choice"] == trace["calls"][-1]["answer"]["choice"]
    assert normalized(trace["probabilities"]) == normalized(trace["calls"][-1]["answer"]["probabilities"])
    assert isinstance(trace["latency_ms"],(float,int)) and math.isfinite(trace["latency_ms"]) and trace["latency_ms"] >= 0
    assert abs(trace["latency_ms"] - elapsed) <= .2


def audit_tournament(game, trace, order):
    if trace["source"] == "forced":
        return
    seed = int(hashlib.sha256(json.dumps(game.snapshot(), sort_keys=True).encode()).hexdigest()[:16],16)
    options = game.candidates()
    random.Random(seed + order).shuffle(options)
    calls = iter(trace["calls"])

    def choose(group):
        call = next(calls)
        assert list(call["request"]["questions"]["placement"]["criteria"]) == [a.id for a in group]
        assert normalized(call["options"]) == normalized([public(a) for a in group])
        return next(a for a in group if a.id == call["answer"]["choice"])

    while len(options) > 20:
        winners = []
        for index in range(0,len(options),20):
            group = options[index:index+20]
            winners.append(choose(group) if len(group)>1 else group[0])
        options = winners
    winner = choose(options)
    assert winner.id == trace["choice"]
    assert next(calls, None) is None


def main():
    suite_path = ROOT / "suite.json"
    suite = json.loads(suite_path.read_text())
    results = load_results()
    assert results is not None, "No raw benchmark evidence found"
    assert results["suite_sha256"] == hashlib.sha256(suite_path.read_bytes()).hexdigest()
    classes = registry()
    case_map = {c["id"]:c for c in suite}
    assert len(suite) == len(case_map) == 24
    assert {c["game"] for c in suite} == set(classes)
    assert all(sum(c["game"] == name for c in suite) == 8 for name in classes)
    for case in suite:
        game = classes[case["game"]].restore(case["snapshot"])
        assert set(case["accepted"]) == set(game.accepted())
    trials, moves, errors = 0, 0, 0
    paired = {}
    assert len({m["name"] for m in results["models"]}) == len(results["models"])
    for model in results["models"]:
        assert model["status"] == "complete"
        wire_model = "kev-latest" if model["name"] in ("kev:4b-mps","kev:4b-mlx") else model["name"]
        def check_model(trace):
            if trace["source"] != "forced":
                assert trace["model"] == model["name"]
                assert all(c["request"]["model"] == wire_model for c in trace["calls"])
        assert len(model["trials"]) == 48
        assert {(t["case"],t["order"]) for t in model["trials"]} == {(c["id"],o) for c in suite for o in range(2)}
        assert set(model["warmups"]) == set(classes)
        for name, cls in classes.items():
            case = [c for c in suite if c["game"] == name][-1]
            warm = cls.restore(case["snapshot"])
            audit_trace(warm,model["warmups"][name])
            audit_tournament(warm,model["warmups"][name],0)
            check_model(model["warmups"][name])
        for trial in model["trials"]:
            case = case_map[trial["case"]]
            assert trial["game"] == case["game"] and trial["accepted"] == case["accepted"]
            if "error" in trial:
                assert not trial["correct"]
                assert not any(key in trial for key in ("decision","action","choice"))
                errors += 1
                continue
            assert "decision" in trial
            game = classes[case["game"]].restore(case["snapshot"])
            trace = trial["decision"]
            check_model(trace)
            audit_trace(game, trace)
            audit_tournament(game, trace, trial["order"])
            assert trial["choice"] == trace["choice"]
            assert trial["correct"] == (trial["choice"] in case["accepted"])
            action = next(a for a in game.candidates() if a.id == trial["choice"])
            assert normalized(public(action)) == normalized(trial["action"])
            # Compare initial group inputs across models. Later finalists legitimately differ.
            first = trace["calls"][0]["request"] if trace["calls"] else None
            key = (trial["case"],trial["order"])
            identity = {k:v for k,v in first.items() if k != "model"} if first else None
            if key in paired:
                assert normalized(paired[key]) == normalized(identity)
            else:
                paired[key] = identity
            trials += 1
        assert len(model["games"]) == len(classes) and {g["game"] for g in model["games"]} == set(classes)
        for record in model["games"]:
            from blockstar import make_replay
            initial = make_replay(42) if record["game"] == "blockstar" else classes[record["game"]](42)
            assert normalized(record["initial"]) == normalized(initial.snapshot())
            assert record["seed"] == 42
            game = classes[record["game"]].restore(record["initial"])
            for turn in record["moves"]:
                check_model(turn["decision"])
                audit_trace(game,turn["decision"])
                audit_tournament(game,turn["decision"],0)
                action = next(a for a in game.candidates() if a.id == turn["action"]["id"])
                assert normalized(public(action)) == normalized(turn["action"])
                assert action.id == turn["decision"]["choice"]
                game.apply(action)
                assert normalized(game.snapshot()) == normalized(turn["after"])
                moves += 1
            assert game.turn == record["turn"]
            final = game.snapshot()
            for key,value in final.items():
                if key not in ("game","board","rng","rng_state","target","pieces","order","last_move","seed","status"):
                    assert normalized(record[key]) == normalized(value)
            assert len(record["moves"]) <= results["limits"][record["game"]]
            if record["status"] == "error":
                assert isinstance(record.get("error"),str) and record["error"]
            elif game.done:
                assert record["status"] == final.get("status","finished")
            else:
                assert record["status"] == "turn_limit" and len(record["moves"]) == results["limits"][record["game"]]
        assert normalized(summarize(model)) == normalized(model["summary"])
    report = {"verified":True, "models":len(results["models"]), "suite_cases":len(suite),
              "decisions":trials, "recorded_errors":errors, "reconstructed_moves":moves,
              "suite_sha256":results["suite_sha256"],
              "checks":["paired input identity", "oracle labels", "every legal action and tournament", "native choices and probability distributions", "deterministic physics, opponent and RNG replay", "recomputed summaries"]}
    atomic(ROOT / "verification.json",report)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
