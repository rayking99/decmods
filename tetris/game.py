"""Deterministic placement-mode Tetris: physics and observations, no AI policy."""
from dataclasses import dataclass
import random

WIDTH, HEIGHT = 10, 20
SHAPES = {
    "I": ((0, 0), (1, 0), (2, 0), (3, 0)),
    "O": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "T": ((1, 0), (0, 1), (1, 1), (2, 1)),
    "S": ((1, 0), (2, 0), (0, 1), (1, 1)),
    "Z": ((0, 0), (1, 0), (1, 1), (2, 1)),
    "J": ((0, 0), (0, 1), (1, 1), (2, 1)),
    "L": ((2, 0), (0, 1), (1, 1), (2, 1)),
}


def normalize(cells):
    min_x, min_y = min(x for x, y in cells), min(y for x, y in cells)
    return tuple(sorted((x - min_x, y - min_y) for x, y in cells))


def rotations(piece):
    cells, result = normalize(SHAPES[piece]), []
    for angle in (0, 90, 180, 270):
        if cells not in [shape for _, shape in result]:
            result.append((angle, cells))
        cells = normalize([(-y, x) for x, y in cells])
    return result


def features(board):
    heights, holes = [], 0
    for x in range(WIDTH):
        top = next((y for y in range(HEIGHT) if board[y][x]), HEIGHT)
        heights.append(HEIGHT - top)
        holes += sum(not board[y][x] for y in range(top, HEIGHT))
    return {
        "heights": heights, "holes": holes,
        "max_height": max(heights), "aggregate_height": sum(heights),
        "bumpiness": sum(abs(a - b) for a, b in zip(heights, heights[1:])),
    }


@dataclass(frozen=True)
class Move:
    id: str
    piece: str
    rotation: int
    x: int
    y: int
    cells: tuple
    cleared: int
    result: tuple
    metrics: dict

    def public(self):
        return {
            "id": self.id, "piece": self.piece, "rotation": self.rotation,
            "x": self.x, "y": self.y, "cells": self.cells,
            "cleared": self.cleared, "metrics": self.metrics,
        }


class Game:
    def __init__(self, seed=42):
        self.seed, self.rng = seed, random.Random(seed)
        self.board = [[0] * WIDTH for _ in range(HEIGHT)]
        self.queue, self.pieces, self.lines, self.score = [], 0, 0, 0
        self._fill_queue()

    def _fill_queue(self):
        while len(self.queue) < 7:
            bag = list(SHAPES)
            self.rng.shuffle(bag)
            self.queue.extend(bag)

    def candidates(self):
        piece, result = self.queue[0], []
        for angle, shape in rotations(piece):
            width = max(x for x, y in shape) + 1
            for x in range(WIDTH - width + 1):
                def fits(y):
                    return all(0 <= y + dy < HEIGHT and not self.board[y + dy][x + dx]
                               for dx, dy in shape)
                if not fits(0):
                    continue
                y = 0
                while fits(y + 1):
                    y += 1
                cells = tuple((x + dx, y + dy) for dx, dy in shape)
                board = [row[:] for row in self.board]
                for cx, cy in cells:
                    board[cy][cx] = piece
                remaining = [row for row in board if not all(row)]
                cleared = HEIGHT - len(remaining)
                board = [[0] * WIDTH for _ in range(cleared)] + remaining
                result.append(Move(f"r{angle}_c{x + 1}", piece, angle, x, y, cells,
                                   cleared, tuple(tuple(row) for row in board), features(board)))
        return result

    def apply(self, move):
        legal = {m.id: m for m in self.candidates()}
        if move.id not in legal or legal[move.id] != move:
            raise ValueError("Move is not legal for the current board")
        self.board = [list(row) for row in move.result]
        self.pieces += 1
        self.lines += move.cleared
        self.score += (0, 100, 300, 500, 800)[move.cleared]
        self.queue.pop(0)
        self._fill_queue()

    def snapshot(self):
        return {"board": self.board, "current": self.queue[0], "next": self.queue[1:6],
                "pieces": self.pieces, "lines": self.lines, "score": self.score,
                "seed": self.seed, "metrics": features(self.board)}

    def observation(self):
        return {
            "current_piece": self.queue[0], "next_piece": self.queue[1],
            "board_top_to_bottom": ["".join("#" if v else "." for v in row) for row in self.board],
            "current_metrics": features(self.board),
            "definitions": "Columns start at 1 on the left. Holes are empty cells below a filled cell. "
                           "Heights and holes are measured AFTER completed rows disappear. "
                           "Each option is a legal rotate, shift, then hard-drop placement.",
        }
