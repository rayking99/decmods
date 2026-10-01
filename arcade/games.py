"""Deterministic game engines and frozen, reproducible decision benchmarks.

The action oracle is deliberately bounded: 2048 is a one-move ranking, and
Connect Four searches four plies after the player's proposed move. Agreement
with these oracles is not a claim of optimal full-game play.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Action:
    id: str
    text: str
    metrics: dict[str, Any]
    after: list[list[int]]


def _as_tuple(value: Any) -> Any:
    """Convert the JSON representation of Random.getstate back to tuples."""
    return tuple(_as_tuple(item) for item in value) if isinstance(value, list) else value


def _action_id(action: Action | str) -> str:
    return action.id if isinstance(action, Action) else str(action)


class Game2048:
    name = "2048"
    policy = (
        "Choose a legal slide by lexicographically maximizing immediate merge "
        "score gained, then empty cells after the slide, then the largest tile. "
        "All ties are accepted. Evaluate before the random spawn. This is a "
        "one-move benchmark, not an optimal full-game strategy."
    )
    directions = ("up", "right", "down", "left")

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.turn = 0
        self.score = 0
        self.board = [[0] * 4 for _ in range(4)]
        self._rng = random.Random(seed)
        self._spawn()
        self._spawn()

    def _spawn(self) -> None:
        empty = [(row, column) for row in range(4) for column in range(4)
                 if self.board[row][column] == 0]
        if empty:
            row, column = self._rng.choice(empty)
            self.board[row][column] = 2 if self._rng.random() < 0.9 else 4

    @staticmethod
    def _merge(line: list[int]) -> tuple[list[int], int]:
        tiles = [tile for tile in line if tile]
        merged: list[int] = []
        gain = 0
        index = 0
        while index < len(tiles):
            if index + 1 < len(tiles) and tiles[index] == tiles[index + 1]:
                tile = 2 * tiles[index]
                merged.append(tile)
                gain += tile
                index += 2
            else:
                merged.append(tiles[index])
                index += 1
        return merged + [0] * (4 - len(merged)), gain

    @classmethod
    def slide(cls, board: list[list[int]], direction: str) -> tuple[list[list[int]], int]:
        """Return a new board, before spawning; each tile merges at most once."""
        if direction not in cls.directions:
            raise ValueError(f"Unknown slide: {direction}")
        result = [[0] * 4 for _ in range(4)]
        total = 0
        for index in range(4):
            coordinates = ([(index, column) for column in range(4)]
                           if direction in ("left", "right")
                           else [(row, index) for row in range(4)])
            if direction in ("right", "down"):
                coordinates.reverse()
            line, gain = cls._merge([board[row][column] for row, column in coordinates])
            total += gain
            for (row, column), tile in zip(coordinates, line):
                result[row][column] = tile
        return result, total

    def candidates(self) -> list[Action]:
        result = []
        for direction in self.directions:
            after, gain = self.slide(self.board, direction)
            if after == self.board:
                continue
            empty = sum(tile == 0 for row in after for tile in row)
            largest = max(tile for row in after for tile in row)
            metrics = {"gain": gain, "empty": empty, "max_tile": largest}
            result.append(Action(direction, f"{direction}: gain {gain}, empty {empty}, "
                                 f"largest {largest}", metrics, after))
        return result

    def accepted(self) -> set[str]:
        actions = self.candidates()
        if not actions:
            return set()
        def rank(action: Action) -> tuple[int, int, int]:
            return (action.metrics["gain"], action.metrics["empty"], action.metrics["max_tile"])
        best = max(map(rank, actions))
        return {action.id for action in actions if rank(action) == best}

    @property
    def done(self) -> bool:
        return not self.candidates()

    def apply(self, action: Action | str) -> dict[str, Any]:
        identifier = _action_id(action)
        candidate = next((item for item in self.candidates() if item.id == identifier), None)
        if candidate is None:
            raise ValueError(f"Illegal 2048 move: {identifier}")
        self.board = copy.deepcopy(candidate.after)
        self.score += candidate.metrics["gain"]
        self.turn += 1
        self._spawn()
        return self.snapshot()

    def observation(self) -> str:
        rows = "/".join(",".join(str(tile) for tile in row) for row in self.board)
        return f"2048 turn={self.turn} score={self.score} board={rows}"

    def snapshot(self) -> dict[str, Any]:
        return {
            "game": self.name, "seed": self.seed, "turn": self.turn,
            "score": self.score, "status": "game_over" if self.done else "playing",
            "done": self.done, "max_tile": max(map(max, self.board)),
            "board": copy.deepcopy(self.board),
            # getstate is a tuple tree; the snapshot must already be JSON-safe.
            "rng_state": _as_list(self._rng.getstate()),
        }

    @classmethod
    def restore(cls, snapshot: dict[str, Any]) -> Game2048:
        board = snapshot["board"]
        if (len(board) != 4 or any(len(row) != 4 for row in board)
                or any(not isinstance(tile, int) or tile < 0 or (tile and tile & (tile - 1))
                       for row in board for tile in row)):
            raise ValueError("A 2048 board must contain a 4×4 grid of zero or power-of-two tiles")
        game = cls.__new__(cls)
        game.seed = snapshot["seed"]
        game.turn = snapshot["turn"]
        game.score = snapshot["score"]
        game.board = copy.deepcopy(board)
        game._rng = random.Random()
        game._rng.setstate(_as_tuple(snapshot["rng_state"]))
        return game


def _as_list(value: Any) -> Any:
    return [_as_list(item) for item in value] if isinstance(value, tuple) else value


_COLUMN_ORDER = (3, 2, 4, 1, 5, 0, 6)
_WIN_SCORE = 100_000
_INFINITY = 1_000_000
_WINDOWS = (
    tuple(tuple((row, column + offset) for offset in range(4))
          for row in range(6) for column in range(4))
    + tuple(tuple((row + offset, column) for offset in range(4))
            for row in range(3) for column in range(7))
    + tuple(tuple((row + offset, column + offset) for offset in range(4))
            for row in range(3) for column in range(4))
    + tuple(tuple((row + offset, column - offset) for offset in range(4))
            for row in range(3) for column in range(3, 7))
)


def _drop(board: list[list[int]], column: int, player: int) -> int:
    for row in range(5, -1, -1):
        if board[row][column] == 0:
            board[row][column] = player
            return row
    raise ValueError(f"Column {column + 1} is full")


def _wins_at(board: list[list[int]], row: int, column: int, player: int) -> bool:
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (-1, 1):
            r, c = row + sign * dr, column + sign * dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r][c] == player:
                count += 1
                r += sign * dr
                c += sign * dc
        if count >= 4:
            return True
    return False


def _winner(board: list[list[int]]) -> int:
    for window in _WINDOWS:
        values = [board[row][column] for row, column in window]
        if values[0] and all(value == values[0] for value in values):
            return values[0]
    return 0


def _heuristic(board: list[list[int]]) -> int:
    # A symmetric evaluation: threats, pairs, singles, and center occupation.
    score = 3 * sum(1 if row[3] == 1 else -1 if row[3] == 2 else 0 for row in board)
    weights = (0, 1, 8, 64, _WIN_SCORE)
    for window in _WINDOWS:
        one = two = 0
        for row, column in window:
            one += board[row][column] == 1
            two += board[row][column] == 2
        if not two:
            score += weights[one]
        if not one:
            score -= weights[two]
    return score


def _search(board: list[list[int]], depth: int, player: int,
            alpha: int, beta: int) -> int:
    legal = [column for column in _COLUMN_ORDER if not board[0][column]]
    if not legal:
        return 0
    if not depth:
        return _heuristic(board)
    best = -_INFINITY if player == 1 else _INFINITY
    for column in legal:
        row = _drop(board, column, player)
        if _wins_at(board, row, column, player):
            value = (_WIN_SCORE + depth - 1) * (1 if player == 1 else -1)
        else:
            value = _search(board, depth - 1, 3 - player, alpha, beta)
        board[row][column] = 0
        if player == 1:
            best = max(best, value)
            alpha = max(alpha, best)
        else:
            best = min(best, value)
            beta = min(beta, best)
        if alpha >= beta:
            break
    return best


def _minimax(board: list[list[int]], depth: int, player: int) -> int:
    winner = _winner(board)
    if winner:
        return (_WIN_SCORE + depth) * (1 if winner == 1 else -1)
    return _search(board, depth, player, -_INFINITY, _INFINITY)


def _immediate_wins(board: list[list[int]], player: int) -> set[int]:
    result = set()
    for column in range(7):
        if board[0][column]:
            continue
        row = _drop(board, column, player)
        if _wins_at(board, row, column, player):
            result.add(column)
        board[row][column] = 0
    return result


class ConnectFour:
    name = "Connect Four"
    policy = (
        "Play as player 1. Choose the legal column with the highest minimax "
        "score after your move, searching four further plies with player 2 "
        "to move first. Terminal wins outrank the symmetric window and center "
        "heuristic; quicker wins and later losses are preferred. All best-score "
        "ties are accepted. Player 2 replies with the same deterministic "
        "four-ply minimax search, breaking ties toward the center."
    )

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.turn = 0
        self.board = [[0] * 7 for _ in range(6)]
        self.last_opponent_column: int | None = None

    @property
    def status(self) -> str:
        winner = _winner(self.board)
        return "won" if winner == 1 else "lost" if winner == 2 else (
            "draw" if all(self.board[0]) else "playing")

    @property
    def done(self) -> bool:
        return self.status != "playing"

    @property
    def score(self) -> int:
        return 1 if self.status == "won" else -1 if self.status == "lost" else 0

    def candidates(self) -> list[Action]:
        if self.done:
            return []
        working = copy.deepcopy(self.board)
        threats = _immediate_wins(working, 2)
        result = []
        for column in range(7):
            if working[0][column]:
                continue
            row = _drop(working, column, 1)
            win = _wins_at(working, row, column, 1)
            remaining_threats = _immediate_wins(working, 2)
            score = _minimax(working, 4, 2)
            metrics = {
                "minimax": score, "immediate_win": win,
                "blocks_immediate": bool(threats) and not remaining_threats,
                "opponent_immediate_wins": len(remaining_threats),
                "column": column + 1, "search_plies": 4,
            }
            after = copy.deepcopy(working)
            working[row][column] = 0
            result.append(Action(f"column-{column + 1}",
                                 f"Column {column + 1}: minimax {score}, win {int(win)}, "
                                 f"block {int(metrics['blocks_immediate'])}", metrics, after))
        return result

    def accepted(self) -> set[str]:
        actions = self.candidates()
        if not actions:
            return set()
        best = max(action.metrics["minimax"] for action in actions)
        return {action.id for action in actions if action.metrics["minimax"] == best}

    def apply(self, action: Action | str) -> dict[str, Any]:
        identifier = _action_id(action)
        if self.done:
            raise ValueError("Connect Four is already finished")
        try:
            column = int(identifier.removeprefix("column-")) - 1
        except ValueError as exc:
            raise ValueError(f"Illegal Connect Four move: {identifier}") from exc
        if identifier != f"column-{column + 1}" or not 0 <= column < 7 or self.board[0][column]:
            raise ValueError(f"Illegal Connect Four move: {identifier}")
        _drop(self.board, column, 1)
        self.turn += 1
        self.last_opponent_column = None
        if not self.done:
            best_column = None
            best_score = _INFINITY
            for reply in _COLUMN_ORDER:
                if self.board[0][reply]:
                    continue
                row = _drop(self.board, reply, 2)
                value = _minimax(self.board, 3, 1)
                self.board[row][reply] = 0
                if value < best_score:
                    best_column, best_score = reply, value
            if best_column is not None:
                _drop(self.board, best_column, 2)
                self.last_opponent_column = best_column + 1
        return self.snapshot()

    def observation(self) -> str:
        rows = "/".join("".join(str(tile) for tile in row) for row in self.board)
        return f"ConnectFour turn={self.turn} player=1 board(top-to-bottom)={rows}; columns=1..7"

    def snapshot(self) -> dict[str, Any]:
        return {
            "game": self.name, "seed": self.seed, "turn": self.turn,
            "score": self.score, "status": self.status, "done": self.done,
            "board": copy.deepcopy(self.board), "player": 1,
            "last_opponent_column": self.last_opponent_column,
        }

    @classmethod
    def restore(cls, snapshot: dict[str, Any]) -> ConnectFour:
        board = snapshot["board"]
        if (len(board) != 6 or any(len(row) != 7 for row in board)
                or any(tile not in (0, 1, 2) for row in board for tile in row)):
            raise ValueError("A Connect Four board must contain a 6×7 grid of 0, 1, or 2")
        for column in range(7):
            seen_piece = False
            for row in range(6):
                if board[row][column]:
                    seen_piece = True
                elif seen_piece:
                    raise ValueError("Connect Four pieces cannot float")
        game = cls(snapshot["seed"])
        game.board = copy.deepcopy(board)
        game.turn = snapshot["turn"]
        game.last_opponent_column = snapshot.get("last_opponent_column")
        return game


def build_suite() -> list[dict[str, Any]]:
    """Return 16 fixed cases: eight per game, never a fresh random workload.

    Each case is a decision before player 1 acts. Seeds, trajectory turns,
    tactical fixtures, and the oracle policy are fixed in source. Snapshots
    contain sufficient state to replay subsequent actions exactly.
    """
    result = []
    game = Game2048(seed=20261001)
    for index, target in enumerate((0, 3, 6, 9, 12, 15, 18, 21), 1):
        while game.turn < target and not game.done:
            accepted = game.accepted()
            game.apply(next(direction for direction in game.directions if direction in accepted))
        result.append({"id": f"2048-{index:02d}", "game": game.name,
                       "snapshot": game.snapshot(), "accepted": sorted(game.accepted())})

    connect = ConnectFour(seed=20261001)
    for index in range(1, 5):
        result.append({"id": f"connect-four-{index:02d}", "game": connect.name,
                       "snapshot": connect.snapshot(), "accepted": sorted(connect.accepted())})
        if not connect.done:
            best = connect.accepted()
            connect.apply(next(f"column-{column + 1}" for column in _COLUMN_ORDER
                               if f"column-{column + 1}" in best))

    fixtures = (
        ("horizontal-win", [[0] * 7 for _ in range(4)] +
         [[0, 0, 0, 0, 2, 0, 0], [1, 1, 1, 0, 2, 2, 0]]),
        ("horizontal-block", [[0] * 7 for _ in range(4)] +
         [[0, 0, 0, 0, 1, 0, 0], [2, 2, 2, 0, 1, 1, 0]]),
        ("vertical-win", [[0] * 7 for _ in range(3)] +
         [[1, 0, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0, 0], [1, 2, 0, 2, 0, 0, 0]]),
        ("vertical-block", [[0] * 7 for _ in range(3)] +
         [[2, 0, 0, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0, 0], [2, 1, 0, 1, 0, 0, 0]]),
    )
    for label, board in fixtures:
        connect = ConnectFour(seed=20261001)
        connect.board = board
        connect.turn = 3
        result.append({"id": f"connect-four-{label}", "game": connect.name,
                       "snapshot": connect.snapshot(), "accepted": sorted(connect.accepted())})
    return result
