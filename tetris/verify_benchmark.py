"""Audit paired model inputs, policy labels, native choices, and game traces."""
import hashlib
import argparse
import json
from pathlib import Path

from benchmark import (BENCH_POLICY, acceptable_moves, compact_description, compact_state,
                       dominated, restore, summarize)
from game import Game

ROOT = Path(__file__).resolve().parent / "benchmarks"


def audit_decision(game, selected, decision, model):
    candidates = {m.id:m for m in game.candidates()}
    assert json.loads(json.dumps(candidates[selected.id].public())) == selected.public_json
    calls = decision["calls"]
    if decision["source"] == "forced":
        assert len(candidates) == 1 and not calls
        return None
    assert decision["source"] in ("ollama", "native-decision-api") and calls
    assert decision["choice"] == selected.id
    for call in calls:
        request, response = call["request"], call["response"]
        assert request["model"] == response["model"] == model
        assert request["state"] == compact_state(game.observation())
        assert request["questions"]["placement"]["instructions"] == BENCH_POLICY
        criteria = request["questions"]["placement"]["criteria"]
        assert 2 <= len(criteria) <= 26
        for move_id, description in criteria.items():
            assert description == compact_description(candidates[move_id])
        answer = response["answers"]["placement"]
        assert answer["choice"] in criteria
        assert answer == call["answer"]
        assert set(answer["probabilities"]) == set(criteria)
        assert abs(sum(answer["probabilities"].values())-1) < .02
    assert calls[-1]["answer"]["choice"] == selected.id
    groups = calls[:-1] if len(calls) > 1 else calls
    ids = [move_id for call in groups for move_id in call["request"]["questions"]["placement"]["criteria"]]
    assert len(ids) == len(set(ids)) == len(candidates)
    assert set(ids) == set(candidates)
    if len(calls) > 1:
        assert set(calls[-1]["request"]["questions"]["placement"]["criteria"]) == {c["answer"]["choice"] for c in groups}
    assert abs(sum(c["latency_ms"] for c in calls)-decision["latency_ms"]) <= .3
    return [{"state":c["request"]["state"],"questions":c["request"]["questions"]} for c in groups]


class Selected:
    def __init__(self, move):
        self.id = move["id"]
        self.public_json = move


def verify(result_path=None, baseline=None):
    results = json.loads((result_path or ROOT / "results.json").read_text())
    if baseline:
        previous = json.loads(baseline.read_text())
        assert previous["suite_sha256"] == results["suite_sha256"]
        results["models"] = previous["models"] + results["models"]
    suite = json.loads((ROOT / "suite.json").read_text())
    assert results["status"] == "complete"
    assert results["suite_sha256"] == hashlib.sha256((ROOT / "suite.json").read_bytes()).hexdigest()
    cases = {c["id"]:c for c in suite}
    paired_inputs, total_trials, total_moves = {}, 0, 0
    for model in results["models"]:
        assert model["status"] == "complete"
        assert model["summary"] == summarize(model)
        assert len(model["trials"]) == len(suite)*results["orders"]
        assert len({(t["case"],t["order"]) for t in model["trials"]}) == len(model["trials"])
        for trial in model["trials"]:
            game = restore(cases[trial["case"]]["snapshot"])
            assert trial["accepted"] == acceptable_moves(game.candidates())
            if trial.get("error"):
                assert not trial["correct"]
                continue
            inputs = audit_decision(game,Selected(trial["move"]),trial["decision"],model["name"])
            chosen = next(m for m in game.candidates() if m.id == trial["choice"])
            assert trial["correct"] == (trial["choice"] in trial["accepted"])
            assert trial["dominated"] == dominated(chosen,game.candidates())
            key = (trial["case"],trial["order"])
            encoded = json.dumps(inputs)
            if key in paired_inputs:
                assert encoded == paired_inputs[key], "Models received different board/group inputs"
            paired_inputs[key] = encoded
            total_trials += 1
        assert {g["seed"] for g in model["games"]} == set(results["game_seeds"])
        for record in model["games"]:
            game = Game(record["seed"])
            for turn in record["moves"]:
                audit_decision(game,Selected(turn["move"]),turn["decision"],model["name"])
                game.apply(next(m for m in game.candidates() if m.id == turn["move"]["id"]))
                assert game.snapshot() == turn["after"]
                total_moves += 1
            assert (game.pieces,game.lines,game.score) == (record["pieces"],record["lines"],record["score"])
            assert record["status"] in ("game_over","piece_limit","error")
            if record["status"] == "game_over":
                assert not game.candidates()
            if record["status"] == "piece_limit":
                assert game.pieces == results["game_piece_limit"]
    return {"verified":True,"models":len(results["models"]),"paired_trials":len(paired_inputs),
            "valid_model_trials":total_trials,"game_moves_reconstructed":total_moves,
            "suite_sha256":results["suite_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=ROOT/"results.json")
    parser.add_argument("--baseline",type=Path)
    args = parser.parse_args()
    result = verify(args.results,args.baseline)
    (args.results.parent / "verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
