"""Rules, replay, and bounded-oracle checks for the dependency-free engines."""

import copy
import json
import unittest
from unittest.mock import patch

try:
    from .games import (ConnectFour, Game2048, _heuristic, _minimax,
                        _winner, build_suite)
except ImportError:
    from games import (ConnectFour, Game2048, _heuristic, _minimax,
                       _winner, build_suite)


class Game2048Tests(unittest.TestCase):
    def test_merges_once_and_preserves_tile_mass(self):
        board = [[2, 2, 2, 2], [2, 2, 4, 0], [4, 0, 4, 4], [0, 0, 0, 0]]
        after, gain = Game2048.slide(board, "left")
        self.assertEqual(after, [[4, 4, 0, 0], [4, 4, 0, 0], [8, 4, 0, 0], [0] * 4])
        self.assertEqual(gain, 20)
        self.assertEqual(sum(map(sum, board)), sum(map(sum, after)))
        self.assertEqual(board[0], [2, 2, 2, 2])

    def test_all_four_directions(self):
        board = [[0] * 4 for _ in range(4)]
        board[0][0] = board[1][0] = 2
        up, gain = Game2048.slide(board, "up")
        down, down_gain = Game2048.slide(board, "down")
        right, right_gain = Game2048.slide(board, "right")
        self.assertEqual(up[0], [4, 0, 0, 0])
        self.assertEqual(down[3], [4, 0, 0, 0])
        self.assertEqual(right[:2], [[0, 0, 0, 2], [0, 0, 0, 2]])
        self.assertEqual((gain, down_gain, right_gain), (4, 4, 0))

    def test_candidates_are_pre_spawn_and_do_not_consume_randomness(self):
        game = Game2048(51)
        game.board = [[2, 2, 4, 0]] + [[0] * 4 for _ in range(3)]
        before = game.snapshot()
        choices = game.candidates()
        self.assertEqual(game.snapshot(), before)
        self.assertEqual(game.accepted(), {"left", "right"})
        left = next(item for item in choices if item.id == "left")
        self.assertEqual(left.after[0], [4, 4, 0, 0])
        self.assertEqual(sum(map(sum, left.after)), sum(map(sum, game.board)))
        game.apply(left)
        self.assertEqual(game.score, 4)
        self.assertEqual(game.turn, 1)
        # Exactly one random 2/4 is introduced only after the committed move.
        self.assertIn(sum(map(sum, game.board)) - sum(map(sum, left.after)), (2, 4))

    def test_json_restore_replays_identical_random_spawns(self):
        original = Game2048(13)
        for _ in range(7):
            original.apply(sorted(original.accepted())[0])
        restored = Game2048.restore(json.loads(json.dumps(original.snapshot())))
        for _ in range(12):
            self.assertEqual(original.snapshot(), restored.snapshot())
            self.assertEqual(original.candidates(), restored.candidates())
            if original.done:
                break
            action = sorted(original.accepted())[0]
            original.apply(action)
            restored.apply(action)
        self.assertEqual(original.snapshot(), restored.snapshot())

    def test_illegal_move_is_transactional_and_game_over_has_no_actions(self):
        game = Game2048(2)
        game.board = [[2, 0, 0, 0]] + [[0] * 4 for _ in range(3)]
        before = game.snapshot()
        with self.assertRaises(ValueError):
            game.apply("left")
        self.assertEqual(game.snapshot(), before)
        game.board = [[2, 4, 2, 4], [4, 2, 4, 2]] * 2
        self.assertTrue(game.done)
        self.assertEqual(game.candidates(), [])
        self.assertEqual(game.accepted(), set())
        self.assertEqual(game.snapshot()["status"], "game_over")


def tactical_board(player):
    other = 3 - player
    return [[0] * 7 for _ in range(4)] + [
        [0, 0, 0, 0, other, 0, 0], [player, player, player, 0, other, other, 0]]


class ConnectFourTests(unittest.TestCase):
    def test_winning_move_ends_before_opponent_reply(self):
        game = ConnectFour(9)
        game.board = tactical_board(1)
        game.turn = 3
        choices = game.candidates()
        win = next(item for item in choices if item.id == "column-4")
        self.assertTrue(win.metrics["immediate_win"])
        self.assertEqual(game.accepted(), {"column-4"})
        self.assertEqual(sum(tile == 2 for row in win.after for tile in row), 3)
        game.apply(win)
        self.assertTrue(game.done)
        self.assertEqual(game.status, "won")
        self.assertEqual(game.score, 1)
        self.assertEqual(game.turn, 4)
        self.assertIsNone(game.last_opponent_column)
        self.assertEqual(sum(tile == 2 for row in game.board for tile in row), 3)
        with self.assertRaises(ValueError):
            game.apply("column-1")

    def test_minimax_blocks_immediate_horizontal_and_vertical_losses(self):
        horizontal = ConnectFour()
        horizontal.board = tactical_board(2)
        self.assertEqual(horizontal.accepted(), {"column-4"})
        choice = next(item for item in horizontal.candidates() if item.id == "column-4")
        self.assertTrue(choice.metrics["blocks_immediate"])
        horizontal.apply("column-4")
        self.assertNotEqual(horizontal.status, "lost")

        vertical = ConnectFour()
        vertical.board = [[0] * 7 for _ in range(3)] + [
            [2, 0, 0, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0, 0], [2, 1, 0, 1, 0, 0, 0]]
        self.assertEqual(vertical.accepted(), {"column-1"})

    def test_opponent_takes_available_win(self):
        game = ConnectFour()
        game.board = tactical_board(2)
        game.apply("column-7")
        self.assertEqual(game.last_opponent_column, 4)
        self.assertEqual(game.status, "lost")
        self.assertEqual(game.score, -1)

    def test_diagonal_wins_and_full_board_draw(self):
        board = [[0] * 7 for _ in range(6)]
        for row, column in ((5, 0), (4, 1), (3, 2), (2, 3)):
            board[row][column] = 1
            for support in range(row + 1, 6):
                board[support][column] = 2
        self.assertEqual(_winner(board), 1)
        self.assertEqual(_winner([row[::-1] for row in board]), 1)

        draw = ConnectFour()
        draw.board = [[1, 1, 2, 2, 1, 1, 2], [2, 2, 1, 1, 2, 2, 1]] * 3
        self.assertEqual(_winner(draw.board), 0)
        self.assertEqual(draw.status, "draw")
        self.assertEqual(draw.candidates(), [])
        # Filling the final two top-row cells reaches a draw through apply().
        draw.board = copy.deepcopy(draw.board)
        draw.board[0][0] = draw.board[0][6] = 0
        draw.apply("column-1")
        self.assertEqual(draw.status, "draw")
        self.assertEqual(draw.last_opponent_column, 7)

    def test_full_column_and_malformed_actions_are_rejected(self):
        game = ConnectFour()
        for row in range(6):
            game.board[row][0] = 1 if row % 2 else 2
        before = game.snapshot()
        self.assertNotIn("column-1", {choice.id for choice in game.candidates()})
        for action in ("column-1", "column-0", "column-8", "column-01", "4", "left"):
            with self.assertRaises(ValueError):
                game.apply(action)
            self.assertEqual(game.snapshot(), before)

    def test_json_restore_and_previews_are_deterministic(self):
        game = ConnectFour(17)
        game.apply("column-4")
        before = game.snapshot()
        previews = game.candidates()
        self.assertEqual(game.snapshot(), before)
        restored = ConnectFour.restore(json.loads(json.dumps(before)))
        self.assertEqual(previews, restored.candidates())
        action = sorted(game.accepted())[0]
        game.apply(action)
        restored.apply(action)
        self.assertEqual(game.snapshot(), restored.snapshot())
        self.assertEqual(sum(tile == 1 for row in game.board for tile in row), 2)
        self.assertEqual(sum(tile == 2 for row in game.board for tile in row), 2)

    def test_all_minimax_ties_are_accepted(self):
        with patch(f"{ConnectFour.__module__}._minimax", return_value=10):
            self.assertEqual(ConnectFour().accepted(), {f"column-{column}" for column in range(1, 8)})

    def test_alpha_beta_matches_exhaustive_two_ply_search(self):
        def exhaustive(board, depth, player):
            winner = _winner(board)
            if winner:
                return (100_000 + depth) * (1 if winner == 1 else -1)
            legal = [column for column in range(7) if not board[0][column]]
            if not legal:
                return 0
            if not depth:
                return _heuristic(board)
            values = []
            for column in legal:
                child = copy.deepcopy(board)
                row = next(row for row in range(5, -1, -1) if child[row][column] == 0)
                child[row][column] = player
                values.append(exhaustive(child, depth - 1, 3 - player))
            return max(values) if player == 1 else min(values)

        for board in (ConnectFour().board, tactical_board(1), tactical_board(2)):
            original = copy.deepcopy(board)
            for player in (1, 2):
                self.assertEqual(_minimax(board, 2, player), exhaustive(board, 2, player))
            self.assertEqual(board, original)


class SuiteTests(unittest.TestCase):
    def test_suite_has_eight_replayable_cases_per_game(self):
        first = build_suite()
        second = build_suite()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        self.assertEqual(len({case["id"] for case in first}), 16)
        self.assertEqual(sum(case["game"] == "2048" for case in first), 8)
        self.assertEqual(sum(case["game"] == "Connect Four" for case in first), 8)
        for case in json.loads(json.dumps(first)):
            engine = Game2048 if case["game"] == "2048" else ConnectFour
            game = engine.restore(case["snapshot"])
            self.assertEqual(sorted(game.accepted()), case["accepted"])
            self.assertTrue(case["accepted"])


if __name__ == "__main__":
    unittest.main()
