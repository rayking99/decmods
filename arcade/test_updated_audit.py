"""Supplemental source-pin, warmup, alias and one-move evidence mutations."""

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import Decision, public
from benchmark_update import summarize
from blockstar import make_replay
from blockstar_update import BlockStar, SOURCE_COMMIT, build_updated_suite, source_metadata
from verify_update import audit_results

try:
    from .test_decision import FakeNativeAPI
except ImportError:
    from test_decision import FakeNativeAPI


def updated_fixture():
    suite = build_updated_suite()
    for case in suite:
        case["accepted"] = sorted(case["accepted"])
    # Persisted evidence has independent JSON values, unlike shared dict objects.
    suite = json.loads(json.dumps(suite))
    models = []
    api = FakeNativeAPI(equal=True)
    with patch("decision.urlopen", side_effect=api):
        for name in ("nimble", "kev:4b-mlx"):
            client = Decision(name, "http://127.0.0.1:9911", BlockStar.policy,
                              wire_model="kev-latest" if name.startswith("kev:") else None)
            model = {"name": name, "label": name, "status": "complete", "trials": []}
            warm = make_replay(98765)
            warm.apply(warm.candidates()[0])
            model["warmup_snapshot"] = warm.snapshot()
            _, model["warmup"] = client.choose(warm)
            for order in (0, 1):
                for case in suite:
                    game = BlockStar.restore(case["snapshot"])
                    action, trace = client.choose(game, order)
                    game.apply(action)
                    model["trials"].append({"case": case["id"], "order": order,
                        "accepted": case["accepted"], "choice": action.id,
                        "correct": action.id in case["accepted"], "action": public(action),
                        "decision": trace, "after": game.snapshot()})
            model["summary"] = summarize(model)
            models.append(model)
    results = {"status": "complete", "source_commit": SOURCE_COMMIT, "suite_sha256": "fixture-hash",
               "models": models, "unavailable": []}
    return suite, json.loads(json.dumps(results))


class UpdatedAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite, cls.results = updated_fixture()

    def audit(self, results=None, suite=None):
        return audit_results(results if results is not None else self.results,
                             suite if suite is not None else self.suite, "fixture-hash", source_metadata(),
                             expected_models={"nimble", "kev:4b-mlx"})

    def test_two_scenarios_two_orders_two_models_replay_and_alias_pass(self):
        report = self.audit()
        self.assertTrue(report["verified"])
        self.assertEqual(report["models"], 2)
        self.assertEqual(report["suite_cases"], 2)
        self.assertEqual(report["decisions"], 8)
        self.assertEqual(report["reconstructed_moves"], 8)
        self.assertEqual(report["warmups"], 2)

    def test_incomplete_run_wrong_pin_and_duplicate_trial_reject(self):
        mutations = (
            lambda value: value.update(status="running"),
            lambda value: value.update(source_commit="other-revision"),
            lambda value: value.update(suite_sha256="other-suite"),
            lambda value: value["models"][0]["trials"].append(copy.deepcopy(value["models"][0]["trials"][0])),
        )
        for mutate in mutations:
            altered = copy.deepcopy(self.results)
            mutate(altered)
            with self.assertRaises(AssertionError):
                self.audit(altered)

    def test_suite_must_be_the_official_initial_board(self):
        suite = copy.deepcopy(self.suite)
        game = BlockStar.restore(suite[0]["snapshot"])
        game.apply(game.candidates()[0])
        suite[0]["snapshot"] = game.snapshot()
        suite[0]["accepted"] = sorted(game.accepted())
        suite[0]["candidate_count"] = len(game.candidates())
        with self.assertRaises(AssertionError):
            self.audit(suite=suite)

    def test_after_snapshot_and_saved_action_mutations_reject(self):
        for field in ("after", "action"):
            with self.subTest(field=field):
                altered = copy.deepcopy(self.results)
                trial = altered["models"][0]["trials"][0]
                if field == "after":
                    trial["after"]["turn"] += 1
                else:
                    trial["action"]["metrics"]["distance"] = 999
                with self.assertRaises(AssertionError):
                    self.audit(altered)

    def test_old_warmup_board_uses_the_updated_policy_instructions(self):
        altered = copy.deepcopy(self.results)
        altered["models"][0]["warmup"]["calls"][0]["request"]["questions"]["placement"]["instructions"] = "old policy"
        with self.assertRaises(AssertionError):
            self.audit(altered)

    def test_fabricated_kev_alias_is_rejected_even_when_request_matches_response(self):
        altered = copy.deepcopy(self.results)
        trace = altered["models"][1]["trials"][0]["decision"]
        for call in trace["calls"]:
            call["request"]["model"] = call["response"]["model"] = "kev:4b-mps"
        with self.assertRaises(AssertionError):
            self.audit(altered)

    def test_success_error_and_summary_mutations_reject(self):
        altered = copy.deepcopy(self.results)
        altered["models"][0]["trials"][0].update(error="Must not bypass trace audit", correct=False)
        with self.assertRaises(AssertionError):
            self.audit(altered)
        altered = copy.deepcopy(self.results)
        altered["models"][0]["summary"]["updated-default"]["candidate_count"] += 1
        with self.assertRaises(AssertionError):
            self.audit(altered)


if __name__ == "__main__":
    unittest.main()
