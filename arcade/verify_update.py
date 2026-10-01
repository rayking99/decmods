"""Audit the modern BlockStar supplement without altering the original run."""

import hashlib
import json

from benchmark import atomic, public
from benchmark_update import ROOT, load_results, summarize
from blockstar import BlockStar as OriginalBlockStar, make_replay
from blockstar_update import (SOURCE_COMMIT, BlockStar, make_large_yard,
                              make_original, source_metadata)
from run_models import SPECS
from verify import audit_trace, audit_tournament, normalized


def check_model(trace, model_name):
    if trace["source"] == "forced":
        return
    wire_name = "kev-latest" if model_name in ("kev:4b-mps", "kev:4b-mlx") else model_name
    assert trace["model"] == model_name
    assert all(call["request"]["model"] == wire_name for call in trace["calls"])


def initial_requests(game, trace):
    """Compare every first-round group; subsequent model finalists can differ."""
    candidates = game.candidates()
    if len(candidates) == 1:
        return []
    count = (1 if len(candidates) <= 20 else
             sum(len(candidates[index:index + 20]) > 1 for index in range(0, len(candidates), 20)))
    return [{key: value for key, value in call["request"].items() if key != "model"}
            for call in trace["calls"][:count]]


def audit_results(results, suite, suite_sha256, source, expected_models=None):
    """Return a report only after pins, native decisions and physics agree."""
    assert results is not None, "No updated BlockStar evidence found"
    assert results["status"] in ("complete", "complete_with_unavailable")
    assert results["source_commit"] == source["commit"] == SOURCE_COMMIT
    assert results["suite_sha256"] == suite_sha256
    case_map = {case["id"]: case for case in suite}
    assert len(suite) == len(case_map) == 2
    assert set(case_map) == {"updated-default", "large-yard"}
    expected_snapshots = {"updated-default": make_original().snapshot(),
                          "large-yard": make_large_yard().snapshot()}
    for case in suite:
        assert case["source_commit"] == SOURCE_COMMIT
        assert case["snapshot"]["source_commit"] == SOURCE_COMMIT
        assert case["game"] == BlockStar.name
        assert normalized(case["snapshot"]) == normalized(expected_snapshots[case["id"]])
        game = BlockStar.restore(case["snapshot"])
        assert case["accepted"] == sorted(game.accepted())
        assert case["candidate_count"] == len(game.candidates())

    expected_models = set(expected_models if expected_models is not None else (spec[0] for spec in SPECS))
    names = [model["name"] for model in results["models"]]
    assert names and len(set(names)) == len(names)
    assert set(names) <= expected_models
    unavailable = {record["name"] for record in results.get("unavailable", [])}
    assert expected_models <= set(names) | unavailable, "An intended runtime has no result or unavailability record"
    paired = {}
    decisions = errors = applied_moves = warmups = 0
    expected_pairs = {(case["id"], order) for case in suite for order in (0, 1)}
    expected_warmup = make_replay(98765)
    expected_warmup.apply(expected_warmup.candidates()[0])
    for model in results["models"]:
        assert model["status"] == "complete"
        assert len(model["trials"]) == 4
        assert {(trial["case"], trial["order"]) for trial in model["trials"]} == expected_pairs
        assert normalized(model["warmup_snapshot"]) == normalized(expected_warmup.snapshot())
        warm = OriginalBlockStar.restore(model["warmup_snapshot"])
        # The runner intentionally sends the updated policy with an old small
        # board's observation/options, keeping this warmup distinct from tests.
        warm.policy = BlockStar.policy
        check_model(model["warmup"], model["name"])
        audit_trace(warm, model["warmup"])
        audit_tournament(warm, model["warmup"], 0)
        warmups += 1
        for trial in model["trials"]:
            assert type(trial["order"]) is int and trial["order"] in (0, 1)
            case = case_map[trial["case"]]
            assert trial["accepted"] == case["accepted"]
            if "error" in trial:
                assert isinstance(trial["error"], str) and trial["error"]
                assert trial["correct"] is False
                assert not any(key in trial for key in ("decision", "action", "choice", "after"))
                errors += 1
                continue
            assert "decision" in trial and "after" in trial
            game = BlockStar.restore(case["snapshot"])
            trace = trial["decision"]
            check_model(trace, model["name"])
            audit_trace(game, trace)
            audit_tournament(game, trace, trial["order"])
            assert trial["choice"] == trace["choice"]
            assert trial["correct"] == (trial["choice"] in case["accepted"])
            action = next(action for action in game.candidates() if action.id == trial["choice"])
            assert normalized(public(action)) == normalized(trial["action"])
            key = (trial["case"], trial["order"])
            identity = normalized(initial_requests(game, trace))
            if key in paired:
                assert paired[key] == identity
            else:
                paired[key] = identity
            game.apply(action)
            assert normalized(game.snapshot()) == normalized(trial["after"])
            decisions += 1
            applied_moves += 1
        assert normalized(summarize(model)) == normalized(model["summary"])
    return {
        "verified": True, "source_commit": SOURCE_COMMIT, "suite_sha256": suite_sha256,
        "suite_cases": len(suite), "models": len(results["models"]),
        "decisions": decisions, "recorded_errors": errors,
        "reconstructed_moves": applied_moves, "warmups": warmups,
        "checks": ["official initial scenarios and common source revision", "all four unique paired trials per model",
                   "every first-round group has paired inputs", "oracle labels and native wire aliases",
                   "every legal slide and complete 20-choice tournaments", "native choices, confidence and probabilities",
                   "distinct captured warmups", "single-move deterministic physics", "recomputed summaries"],
    }


def main():
    suite_path = ROOT / "suite.json"
    suite = json.loads(suite_path.read_text())
    suite_sha256 = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    report = audit_results(load_results(), suite, suite_sha256, source_metadata())
    atomic(ROOT / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
