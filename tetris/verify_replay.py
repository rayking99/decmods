"""Reconstruct every saved board and verify actions against raw model answers."""
import argparse
import json
from pathlib import Path
import statistics

from decision import describe
from game import Game


def verify(path):
    replay = json.loads(Path(path).read_text())
    describe_option, observation = describe, lambda g:g.observation()
    if replay["model"].get("policy") == "benchmark":
        from benchmark import compact_description, compact_state
        describe_option, observation = compact_description, lambda g:compact_state(g.observation())
    game, latencies, api_latencies = Game(replay["seed"]), [], []
    calls = 0
    for entry in replay["decisions"]:
        candidates = {move.id: move for move in game.candidates()}
        decision = entry["decision"]
        selected = candidates[decision["choice"]]
        assert json.loads(json.dumps(selected.public())) == entry["move"], "Move differs from simulation"
        assert entry["turn"] == game.pieces + 1, "Turn sequence is not continuous"
        native = decision["calls"]
        if decision["source"] in ("ollama", "native-decision-api"):
            assert native, "Missing raw model evidence"
            assert native[-1]["response"]["answers"]["placement"]["choice"] == selected.id, "Executed choice differs from model"
            for call in native:
                request, response = call["request"], call["response"]
                assert request["model"] == response["model"] == decision["model"]
                assert request["state"] == observation(game), "Model saw a different board"
                criteria = request["questions"]["placement"]["criteria"]
                assert 2 <= len(criteria) <= 26
                for move_id, description in criteria.items():
                    assert description == describe_option(candidates[move_id]), "Incorrect option observation"
                answer = response["answers"]["placement"]
                assert answer["choice"] in criteria
                assert set(answer["probabilities"]) == set(criteria)
                assert abs(sum(answer["probabilities"].values()) - 1) < .02
                api_latencies.append(call["latency_ms"])
            groups = native[:-1] if len(native) > 1 else native
            covered = [move_id for call in groups for move_id in call["request"]["questions"]["placement"]["criteria"]]
            assert len(covered) == len(set(covered)) == len(candidates), "Not every legal landing was offered"
            assert set(covered) == set(candidates)
            if len(native) > 1:
                finalists = {call["answer"]["choice"] for call in native[:-1]}
                assert set(native[-1]["request"]["questions"]["placement"]["criteria"]) == finalists
            calls += len(native)
        else:
            assert decision["source"] == "forced" and len(candidates) == 1 and not native
        latencies.append(decision["latency_ms"])
        game.apply(selected)
        assert game.snapshot() == entry["after"], "Recorded board differs from engine reconstruction"
    assert game.snapshot() == replay["final"], "Final state differs from reconstructed game"
    return {"replay": str(Path(path).resolve()), "verified": True,
            "model_digest": replay["model"].get("digest"), "pieces": game.pieces,
            "lines": game.lines, "score": game.score, "api_calls": calls,
            "mean_placement_ms": round(statistics.mean(latencies), 1) if latencies else None,
            "median_api_ms": round(statistics.median(api_latencies), 1) if api_latencies else None,
            "max_height": game.snapshot()["metrics"]["max_height"],
            "holes": game.snapshot()["metrics"]["holes"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?")
    args = parser.parse_args()
    target = Path(args.path) if args.path else max((Path(__file__).parent / "runs").glob("*.json"), key=lambda p: p.stat().st_mtime)
    print(json.dumps(verify(target), indent=2))
