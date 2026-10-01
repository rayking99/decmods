"""Reject mutated benchmark evidence without inference or large raw artifacts."""

import contextlib
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

# The command-line audit/export modules use sibling imports, as their CLI does.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark
import blockstar
import build_site_data
import verify
from games import Action

try:
    from .test_decision import FakeNativeAPI
except ImportError:
    from test_decision import FakeNativeAPI


class MiniGame:
    """Small auditable rules, with the same public contract as real engines."""
    name = "mini"
    policy = "Either legal move is tied for best."

    def __init__(self, seed=42):
        self.seed = seed
        self.turn = 0
        self.board = [[0]]

    def candidates(self):
        return [Action(identifier, f"Advance using {identifier}", {"gain": 1}, [[self.turn + 1]])
                for identifier in ("move-a", "move-b")]

    def accepted(self):
        return {"move-a", "move-b"}

    @property
    def done(self):
        return False

    def observation(self):
        return f"{self.name} seed={self.seed} turn={self.turn}"

    def snapshot(self):
        return {"game": self.name, "seed": self.seed, "turn": self.turn, "score": self.turn,
                "status": "playing", "done": False, "board": copy.deepcopy(self.board)}

    @classmethod
    def restore(cls, snapshot):
        game = cls(snapshot["seed"])
        game.turn = snapshot["turn"]
        game.board = copy.deepcopy(snapshot["board"])
        return game

    def apply(self, action):
        if action.id not in self.accepted():
            raise ValueError("Illegal move")
        self.turn += 1
        self.board = [[self.turn]]


class Mini2048(MiniGame):
    name = "2048"


class MiniConnect4(MiniGame):
    name = "Connect Four"


class MiniBlockStar(MiniGame):
    name = "blockstar"


MINI_REGISTRY = {"2048": Mini2048, "connect4": MiniConnect4, "blockstar": MiniBlockStar}


def make_fixture():
    suite = []
    for name, cls in MINI_REGISTRY.items():
        for index in range(8):
            game = cls(100 + index)
            game.turn = index
            game.board = [[index]]
            suite.append({"id": f"{name}-{index}", "game": name, "snapshot": game.snapshot(),
                          "accepted": sorted(game.accepted())})
    model = {"name": "nimble", "label": "Fixture", "status": "complete", "metadata": {},
             "trials": [], "games": [], "warmups": {}}
    api = FakeNativeAPI(equal=True)
    with patch("decision.urlopen", side_effect=api):
        for name, cls in MINI_REGISTRY.items():
            client = benchmark.Decision("nimble", "http://127.0.0.1:9911", cls.policy)
            cases = [case for case in suite if case["game"] == name]
            _, model["warmups"][name] = client.choose(cls.restore(cases[-1]["snapshot"]))
            for order in range(2):
                for case in cases:
                    game = cls.restore(case["snapshot"])
                    action, trace = client.choose(game, order)
                    model["trials"].append({"game": name, "case": case["id"], "order": order,
                        "accepted": case["accepted"], "choice": action.id, "correct": True,
                        "action": benchmark.public(action), "decision": trace})
            game = cls(42)
            initial = game.snapshot()
            action, trace = client.choose(game)
            game.apply(action)
            final = game.snapshot()
            model["games"].append({"game": name, "seed": 42, "initial": initial,
                "moves": [{"action": benchmark.public(action), "decision": trace, "after": final}],
                "turn": game.turn, "score": final["score"], "done": final["done"], "status": "turn_limit"})
    with patch.object(benchmark, "registry", return_value=MINI_REGISTRY):
        model["summary"] = benchmark.summarize(model)
    results = {"status": "complete", "orders": 2, "game_seeds": [42],
               "limits": {name: 1 for name in MINI_REGISTRY}, "models": [model]}
    return suite, results


class AuditMutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite, cls.results = make_fixture()

    def audit(self, results):
        results = copy.deepcopy(results)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            suite_path = root / "suite.json"
            benchmark.atomic(suite_path, self.suite)
            results["suite_sha256"] = hashlib.sha256(suite_path.read_bytes()).hexdigest()
            with (patch.object(verify, "ROOT", root),
                  patch.object(verify, "load_results", return_value=results),
                  patch.object(verify, "registry", return_value=MINI_REGISTRY),
                  patch.object(benchmark, "registry", return_value=MINI_REGISTRY),
                  patch.object(blockstar, "make_replay", MiniBlockStar),
                  contextlib.redirect_stdout(io.StringIO())):
                # Recompute summaries, so structural audit checks must detect the mutation.
                for model in results["models"]:
                    model["summary"] = benchmark.summarize(model)
                verify.main()
            return json.loads((root / "verification.json").read_text())

    def test_complete_synthetic_evidence_passes_and_excludes_warmup_timing(self):
        report = self.audit(self.results)
        self.assertTrue(report["verified"])
        self.assertEqual(report["decisions"], 48)
        self.assertEqual(report["reconstructed_moves"], 3)
        for summary in self.results["models"][0]["summary"].values():
            self.assertEqual(summary["timing_samples"], 7)
            self.assertEqual(summary["valid"], 16)

    def test_duplicate_trials_reject_even_if_summary_is_recomputed(self):
        altered = copy.deepcopy(self.results)
        altered["models"][0]["trials"].append(copy.deepcopy(altered["models"][0]["trials"][0]))
        with self.assertRaises(AssertionError):
            self.audit(altered)

    def test_missing_game_rejects_even_if_summary_is_recomputed(self):
        altered = copy.deepcopy(self.results)
        altered["models"][0]["games"].pop()
        with self.assertRaises(AssertionError):
            self.audit(altered)

    def test_trial_cannot_claim_both_success_and_error(self):
        altered = copy.deepcopy(self.results)
        trial = altered["models"][0]["trials"][0]
        trial["error"] = "Added error must not bypass trace verification"
        trial["correct"] = False
        with self.assertRaises(AssertionError):
            self.audit(altered)

    def test_final_score_status_and_initial_seed_mutations_reject(self):
        for key, value in (("score", 999), ("status", "won"), ("seed", 999)):
            with self.subTest(key=key):
                altered = copy.deepcopy(self.results)
                altered["models"][0]["games"][0][key] = value
                with self.assertRaises(AssertionError):
                    self.audit(altered)

    def test_fabricated_matching_request_response_model_rejects(self):
        altered = copy.deepcopy(self.results)
        trace = altered["models"][0]["trials"][0]["decision"]
        trace["model"] = "other-family"
        for call in trace["calls"]:
            call["request"]["model"] = call["response"]["model"] = "other-family"
        with self.assertRaises(AssertionError):
            self.audit(altered)

    def test_mutated_warmup_state_rejects(self):
        altered = copy.deepcopy(self.results)
        altered["models"][0]["warmups"]["2048"]["calls"][0]["request"]["state"] = "different board"
        with self.assertRaises(AssertionError):
            self.audit(altered)

    def test_confidence_probability_timing_and_saved_options_mutations_reject(self):
        def confidence(trace):
            trace["calls"][0]["answer"]["confidence"] = float("inf")
            trace["calls"][0]["response"]["answers"]["placement"]["confidence"] = float("inf")

        def missing_confidence(trace):
            del trace["calls"][0]["answer"]["confidence"]
            # answer and response intentionally share identity in real trace objects.
            trace["calls"][0]["response"]["answers"]["placement"].pop("confidence", None)

        def probabilities(trace):
            trace["probabilities"] = {"move-a": 1, "move-b": 0}

        def negative_latency(trace):
            trace["latency_ms"] = trace["calls"][0]["latency_ms"] = -1

        def options(trace):
            trace["calls"][0]["options"][0]["after"] = [[999]]

        def reordered_group(trace):
            question = trace["calls"][0]["request"]["questions"]["placement"]
            question["criteria"] = dict(reversed(list(question["criteria"].items())))

        for mutate in (confidence, missing_confidence, probabilities, negative_latency,
                       options, reordered_group):
            with self.subTest(case=mutate.__name__):
                altered = copy.deepcopy(self.results)
                mutate(altered["models"][0]["trials"][0]["decision"])
                with self.assertRaises(AssertionError):
                    self.audit(altered)


class EvidenceExportTests(unittest.TestCase):
    def test_clean_snapshot_removes_only_rng_fields(self):
        snapshot = {"board": [[2, 4]], "rng_state": [1, 2, 3], "rng": [4], "seed": 42,
                    "turn": 6, "score": 12, "status": "playing", "custom": {"note": "retained"}}
        before = copy.deepcopy(snapshot)
        clean = build_site_data.clean_snapshot(snapshot)
        self.assertEqual(clean, {key: value for key, value in snapshot.items()
                                 if key not in ("rng", "rng_state")})
        self.assertEqual(snapshot, before)

    def test_gzip_fallback_preserves_full_json_and_local_raw_takes_priority(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "arcade"
            root.mkdir()
            archive = Path(folder) / "docs/data/arcade-results.json.gz"
            archive.parent.mkdir(parents=True)
            evidence = {"models": [{"name": "fixture", "unicode": "日本語", "rng_state": [0, 1]}]}
            with gzip.open(archive, "wt", encoding="utf-8") as stream:
                json.dump(evidence, stream, ensure_ascii=False)
            with patch.object(benchmark, "ROOT", root):
                self.assertEqual(benchmark.load_results(), evidence)
                benchmark.atomic(root / "results.json", {"local": True})
                self.assertEqual(benchmark.load_results(), {"local": True})


if __name__ == "__main__":
    unittest.main()
