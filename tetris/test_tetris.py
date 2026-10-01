import io
import json
import random
import unittest
from unittest.mock import patch

from decision import OllamaDecision
from game import Game, HEIGHT, WIDTH, features, rotations


class PhysicsTests(unittest.TestCase):
    def test_unique_rotations(self):
        self.assertEqual({p: len(rotations(p)) for p in "IOTSZJL"},
                         {"I": 2, "O": 1, "T": 4, "S": 2, "Z": 2, "J": 4, "L": 4})

    def test_every_piece_on_empty_board(self):
        expected = {"I": 17, "O": 9, "T": 34, "S": 17, "Z": 17, "J": 34, "L": 34}
        for piece, count in expected.items():
            game = Game()
            game.queue[0] = piece
            moves = game.candidates()
            self.assertEqual(len(moves), count)
            self.assertEqual(len({m.cells for m in moves}), count)
            for move in moves:
                self.assertEqual(len(set(move.cells)), 4)
                self.assertTrue(all(0 <= x < WIDTH and 0 <= y < HEIGHT for x, y in move.cells))
                self.assertEqual(sum(bool(v) for row in move.result for v in row), 4)

    def test_four_line_clear_and_score(self):
        game = Game()
        game.queue[0] = "I"
        for y in range(16, 20):
            game.board[y] = ["J"] * 9 + [0]
        move = next(m for m in game.candidates() if m.rotation == 90 and m.x == 9)
        self.assertEqual(move.cleared, 4)
        game.apply(move)
        self.assertEqual((game.lines, game.score, game.pieces), (4, 800, 1))
        self.assertEqual(game.board, [[0] * 10 for _ in range(20)])

    def test_holes_and_height(self):
        board = [[0] * 10 for _ in range(20)]
        board[17][0], board[19][0] = "I", "I"
        self.assertEqual(features(board), {"heights": [3] + [0] * 9, "holes": 1,
                                          "max_height": 3, "aggregate_height": 3, "bumpiness": 3})

    def test_seven_bag_and_seed(self):
        a, b = Game(123), Game(123)
        sequence = []
        for _ in range(14):
            self.assertEqual(a.queue, b.queue)
            sequence.append(a.queue.pop(0))
            b.queue.pop(0)
            a._fill_queue()
            b._fill_queue()
        for bag in (sequence[:7], sequence[7:]):
            self.assertEqual(set(bag), set("IOTSZJL"))

    def test_topout(self):
        game = Game()
        game.board = [["Z"] * 10 for _ in range(20)]
        self.assertEqual(game.candidates(), [])

    def test_stale_move_rejected(self):
        game = Game()
        move = game.candidates()[0]
        for x, y in move.cells:
            game.board[y][x] = "Z"
        with self.assertRaises(ValueError):
            game.apply(move)

    def test_playouts_preserve_cells_and_clear_full_rows(self):
        rng = random.Random(8)
        for seed in range(8):
            game = Game(seed)
            for turn in range(50):
                options = game.candidates()
                if not options:
                    break
                old_cells = sum(bool(v) for row in game.board for v in row)
                move = rng.choice(options)
                game.apply(move)
                new_cells = sum(bool(v) for row in game.board for v in row)
                self.assertEqual(new_cells, old_cells + 4 - 10 * move.cleared)
                self.assertEqual(len(game.board), HEIGHT)
                self.assertTrue(all(len(row) == WIDTH and not all(row) for row in game.board))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.game = Game()
        self.game.queue[0] = "O"
        self.moves = self.game.candidates()
        self.adapter = OllamaDecision()

    def response(self, answer=None):
        probs = {m.id: 1 / len(self.moves) for m in self.moves}
        return {"model": "nimble", "answers": {"placement": answer or {
            "type": "choice", "choice": self.moves[-1].id, "probabilities": probs, "confidence": 0}},
                "usage": {"input_tokens": 123, "output_tokens": 1}}

    def test_native_api_contract_and_real_choice(self):
        with patch("decision.urlopen", return_value=io.BytesIO(json.dumps(self.response()).encode())) as http:
            move, result = self.adapter.choose(self.game)
        req = http.call_args.args[0]
        self.assertEqual(req.full_url, "http://127.0.0.1:11434/v1/systemone")
        body = json.loads(req.data)
        self.assertEqual(body["questions"]["placement"]["type"], "choice")
        self.assertEqual(set(body["questions"]["placement"]["criteria"]), {m.id for m in self.moves})
        self.assertEqual(move.id, self.moves[-1].id)
        self.assertEqual(result["source"], "ollama")
        self.assertEqual(self.game.pieces, 0)

    def test_invalid_choice_stops_instead_of_fallback(self):
        raw = self.response()
        raw["answers"]["placement"]["choice"] = "invented"
        with patch("decision.urlopen", return_value=io.BytesIO(json.dumps(raw).encode())):
            with self.assertRaises(RuntimeError):
                self.adapter.choose(self.game)

    def test_nonfinite_probability_rejected(self):
        raw = self.response()
        raw["answers"]["placement"]["probabilities"][self.moves[0].id] = float("nan")
        with patch("decision.urlopen", return_value=io.BytesIO(json.dumps(raw).encode())):
            with self.assertRaises(RuntimeError):
                self.adapter.choose(self.game)

    def test_incomplete_distribution_rejected(self):
        raw = self.response()
        del raw["answers"]["placement"]["probabilities"][self.moves[0].id]
        with patch("decision.urlopen", return_value=io.BytesIO(json.dumps(raw).encode())):
            with self.assertRaises(RuntimeError):
                self.adapter.choose(self.game)

    def test_all_landings_reach_model_in_tournament(self):
        self.game.queue[0] = "T"
        seen = []
        def fake_request(state, options):
            seen.append([m.id for m in options])
            return {"answer": {"choice": options[-1].id,
                    "probabilities": {m.id: 1/len(options) for m in options}, "confidence": 0},
                    "latency_ms": 12, "options": [m.public() for m in options]}
        with patch.object(self.adapter, "request", side_effect=fake_request):
            move, result = self.adapter.choose(self.game)
        self.assertEqual(len(seen), 3)
        self.assertEqual(set(seen[0] + seen[1]), {m.id for m in self.game.candidates()})
        self.assertTrue(all(2 <= len(batch) <= 26 for batch in seen))
        self.assertEqual(seen[2], [seen[0][-1], seen[1][-1]])
        self.assertEqual(move.id, seen[2][-1])
        self.assertEqual(result["latency_ms"], 36)

    def test_no_move_no_model_call(self):
        self.game.board = [["I"] * 10 for _ in range(20)]
        with patch.object(self.adapter, "request") as request:
            with self.assertRaises(ValueError):
                self.adapter.choose(self.game)
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
