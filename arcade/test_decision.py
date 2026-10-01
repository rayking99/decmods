"""Native wire-contract and lossless tournament checks without model inference."""

import copy
import hashlib
import io
import json
import random
import unittest
from unittest.mock import patch

try:
    from .benchmark import Decision
    from .games import Action
except ImportError:
    from benchmark import Decision
    from games import Action


class FakeGame:
    def __init__(self, count):
        self.options = [Action(f"move-{index:03d}", f"legal option {index}",
                               {"value": index}, [[index]]) for index in range(count)]

    def candidates(self):
        return self.options[:]

    def snapshot(self):
        return {"game": "fixture", "seed": 17, "turn": 0, "board": [[0]], "count": len(self.options)}

    def observation(self):
        return "fixture turn=0 board=0"


class FakeNativeAPI:
    """Build real native response JSON from the criteria actually transmitted."""
    def __init__(self, choice=None, alter=None, equal=False):
        self.calls = []
        self.choice = choice
        self.alter = alter
        self.equal = equal

    def __call__(self, request, timeout):
        body = json.loads(request.data)
        self.calls.append((request.full_url, timeout, body))
        ids = list(body["questions"]["placement"]["criteria"])
        chosen = self.choice(ids) if self.choice else ids[-1]
        probabilities = ({identifier: 1 / len(ids) for identifier in ids} if self.equal else
                         {identifier: int(identifier == chosen) for identifier in ids})
        raw = {"model": body["model"], "answers": {"placement": {
            "type": "choice", "choice": chosen, "probabilities": probabilities,
            "confidence": 0 if self.equal else 1}}}
        if self.alter:
            self.alter(raw)
        return io.BytesIO(json.dumps(raw).encode())


class DecisionContractTests(unittest.TestCase):
    def setUp(self):
        self.client = Decision("variant-mlx", "http://127.0.0.1:9911", "maximize value",
                               wire_model="native-canonical")

    def test_native_payload_uses_actual_choice_and_preserves_wire_alias(self):
        game = FakeGame(7)
        before = game.snapshot()
        api = FakeNativeAPI(choice=lambda ids: "move-002")
        with patch("decision.urlopen", side_effect=api):
            action, trace = self.client.choose(game)
        url, timeout, body = api.calls[0]
        self.assertEqual(url, "http://127.0.0.1:9911/v1/systemone")
        self.assertEqual(timeout, 300)
        self.assertEqual(body["model"], "native-canonical")
        self.assertEqual(body["state"], game.observation())
        placement = body["questions"]["placement"]
        self.assertEqual(placement["type"], "choice")
        self.assertEqual(placement["instructions"], "maximize value")
        self.assertEqual(placement["criteria"], {option.id: option.text for option in game.options})
        self.assertEqual(action.id, "move-002")
        self.assertEqual(trace["model"], "variant-mlx")
        self.assertEqual(trace["calls"][0]["response"]["model"], "native-canonical")
        self.assertEqual(trace["candidate_count"], 7)
        self.assertEqual(len(trace["all_options"]), 7)
        self.assertEqual(game.snapshot(), before)

    def test_every_legal_option_reaches_model_in_recursive_tournaments(self):
        for count in (2, 20, 21, 40, 41, 400, 401, 421):
            with self.subTest(count=count):
                game = FakeGame(count)
                # A strict ranking should retain the best choice at every stage.
                api = FakeNativeAPI(choice=max)
                with patch("decision.urlopen", side_effect=api):
                    action, trace = self.client.choose(game)
                groups = [list(call[2]["questions"]["placement"]["criteria"]) for call in api.calls]
                self.assertTrue(all(2 <= len(group) <= 20 for group in groups))
                self.assertEqual(set().union(*(set(group) for group in groups)),
                                 {option.id for option in game.options})
                self.assertEqual(action.id, game.options[-1].id)
                self.assertEqual(len(trace["calls"]), len(groups))
                self.assertEqual(trace["candidate_count"], count)
                self.assertEqual(trace["latency_ms"],
                                 round(sum(call["latency_ms"] for call in trace["calls"]), 1))

    def test_last_singleton_is_carried_forward_and_tied_choices_are_valid(self):
        game = FakeGame(21)
        seed = int(hashlib.sha256(json.dumps(game.snapshot(), sort_keys=True).encode()).hexdigest()[:16], 16)
        order = game.options[:]
        random.Random(seed).shuffle(order)
        singleton = order[-1].id
        api = FakeNativeAPI(choice=lambda ids: singleton if singleton in ids else ids[0], equal=True)
        with patch("decision.urlopen", side_effect=api):
            action, trace = self.client.choose(game)
        groups = [list(call[2]["questions"]["placement"]["criteria"]) for call in api.calls]
        self.assertEqual([len(group) for group in groups], [20, 2])
        self.assertNotIn(singleton, groups[0])
        self.assertIn(singleton, groups[1])
        self.assertEqual(action.id, singleton)
        self.assertEqual(len(trace["calls"]), 2)

    def test_option_orders_are_reproducible_and_can_change(self):
        game = FakeGame(7)
        orders = []
        for order in (0, 0, 1):
            api = FakeNativeAPI(equal=True)
            with patch("decision.urlopen", side_effect=api):
                self.client.choose(game, order)
            orders.append(list(api.calls[0][2]["questions"]["placement"]["criteria"]))
        self.assertEqual(orders[0], orders[1])
        self.assertNotEqual(orders[0], orders[2])
        self.assertEqual(set(orders[0]), set(orders[2]))

    def test_invalid_native_outputs_raise_without_fallback(self):
        def wrong_model(raw):
            raw["model"] = "other-family"

        def wrong_type(raw):
            raw["answers"]["placement"]["type"] = "text"

        def invented_choice(raw):
            raw["answers"]["placement"]["choice"] = "invented"

        def incomplete(raw):
            probabilities = raw["answers"]["placement"]["probabilities"]
            probabilities.pop(next(iter(probabilities)))

        def extra_choice(raw):
            raw["answers"]["placement"]["probabilities"]["invented"] = 0

        def nonfinite(raw):
            probabilities = raw["answers"]["placement"]["probabilities"]
            probabilities[next(iter(probabilities))] = float("nan")

        def wrong_total(raw):
            probabilities = raw["answers"]["placement"]["probabilities"]
            for identifier in probabilities:
                probabilities[identifier] = 0

        def negative(raw):
            probabilities = raw["answers"]["placement"]["probabilities"]
            probabilities[next(iter(probabilities))] = -0.1

        def bad_confidence(raw):
            raw["answers"]["placement"]["confidence"] = float("inf")

        def missing_confidence(raw):
            del raw["answers"]["placement"]["confidence"]

        for alter in (wrong_model, wrong_type, invented_choice, incomplete, extra_choice,
                      nonfinite, wrong_total, negative, bad_confidence, missing_confidence):
            with self.subTest(case=alter.__name__):
                game = FakeGame(4)
                before = copy.deepcopy(game.options)
                api = FakeNativeAPI(alter=alter)
                with patch("decision.urlopen", side_effect=api):
                    with self.assertRaises(RuntimeError):
                        self.client.choose(game)
                self.assertEqual(len(api.calls), 1)
                self.assertEqual(game.options, before)

    def test_invalid_tournament_group_aborts_before_final_choice(self):
        def invalid(raw):
            raw["answers"]["placement"]["choice"] = "invented"
        api = FakeNativeAPI(alter=invalid)
        with patch("decision.urlopen", side_effect=api):
            with self.assertRaises(RuntimeError):
                self.client.choose(FakeGame(41))
        self.assertEqual(len(api.calls), 1)

    def test_forced_and_empty_states_do_not_call_native_api(self):
        with patch("decision.urlopen") as http:
            game = FakeGame(1)
            action, trace = self.client.choose(game)
            self.assertIs(action, game.options[0])
            self.assertEqual(trace["source"], "forced")
            self.assertEqual(trace["latency_ms"], 0)
            self.assertEqual(trace["calls"], [])
            self.assertEqual(len(trace["all_options"]), 1)
            with self.assertRaises(ValueError):
                self.client.choose(FakeGame(0))
        http.assert_not_called()


if __name__ == "__main__":
    unittest.main()
