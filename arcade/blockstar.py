"""Independent, deterministic adapter for the pinned BlockStar block puzzle.

Only rules and layout data are reproduced: importing this module never runs the
upstream stochastic driver or AgentFormer training. The oracle is a disclosed
one-step heuristic, not a claim of globally optimal multi-block planning.
"""
from collections import deque
from dataclasses import dataclass
import copy
import random
import time

SOURCE_COMMIT = "f40b338af841bcbcbe402770e5e07610b3397c14"
DIRECTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
PIECES = ("A", "C", "E", "F", "G", "K", "B", "L", "P", "Q", "T", "U", "V", "M", "D", "W")
# This is Puzzle.topological_sort() on the pinned original start/target pair,
# not PIECES: its DFS also visits cyclic dependencies.
ORIGINAL_ORDER = ("A", "K", "B", "E", "L", "D", "C", "W", "V", "M", "F", "U", "Q", "G", "P", "T")
ORIGINAL_START = (
    "W----------------L", "W------BBB-------L", "W------BBB---TTT--",
    "W------------TTT--", "WCCC-VVV----------", "-CCC-VVV--------UU",
    "-CCC------------UU", "-------PP---KK----", "-------PP---------",
    "-------PP-------QQ", "-------PP---GG--QQ", "------------GG----",
    "------------GG----", "------------GG----", "------------GGMM--",
    "-----------FFFFFF-", "AAAAA------FFFFFF-", "AAAAA---------DD--",
    "AAAAA---------DD--", "E-------------DD--",
)
ORIGINAL_TARGET = (
    "AAAAACCCFFFFFFUUQQ", "AAAAACCCFFFFFFUUQQ", "AAAAACCCMMGGPP----",
    "KKELDDWVVVGGPP----", "BBBLDDWVVVGGPP----", "BBB-DDWTTTGGPP----",
    "------WTTTGG------", "------W-----------", "------------------",
    "------------------", "------------------", "------------------",
    "------------------", "------------------", "------------------",
    "------------------", "------------------", "------------------",
    "------------------", "------------------",
)


@dataclass(frozen=True)
class Action:
    id: str
    text: str
    metrics: dict
    after: tuple


def _board(rows):
    return tuple(tuple(row) for row in rows)


def positions(board, piece):
    """Source-compatible row-major cell ordering, using (row, column)."""
    return tuple((r, c) for r, row in enumerate(board) for c, cell in enumerate(row) if cell == piece)


def movable_distances(board, piece):
    """All slide limits. Every intermediate cell must be empty or the same block."""
    cells = positions(board, piece)
    if not cells:
        return {}
    own, result = set(cells), {}
    height, width = len(board), len(board[0])
    for direction, (dr, dc) in DIRECTIONS.items():
        distance = 0
        while True:
            step = distance + 1
            shifted = ((r + dr * step, c + dc * step) for r, c in cells)
            if not all(0 <= r < height and 0 <= c < width and
                       (board[r][c] == "-" or (r, c) in own) for r, c in shifted):
                break
            distance = step
        if distance:
            result[direction] = distance
    return result


def translate(board, piece, direction, distance):
    """Rigid translation with a swept collision check; no rotations or jumps."""
    if direction not in DIRECTIONS or not isinstance(distance, int) or distance < 1:
        raise ValueError("A slide needs a known direction and a positive integer distance")
    if distance > movable_distances(board, piece).get(direction, 0):
        raise ValueError("Slide crosses another piece or the board boundary")
    dr, dc = DIRECTIONS[direction]
    cells, new = positions(board, piece), [list(row) for row in board]
    for r, c in cells:
        new[r][c] = "-"
    for r, c in cells:
        new[r + dr * distance][c + dc * distance] = piece
    return _board(new)


def upstream_piece_path(board, target, piece):
    """Independently reproduce pinned Puzzle.a_star_search's static-block path.

Its score sums Manhattan distance over every cell while each one-cell slide
costs one. That heuristic can overestimate; this is the upstream reference
algorithm, not an optimal shortest-path assertion. Other blocks remain fixed.
"""
    start, end = positions(board, piece), positions(target, piece)
    if not start or not end:
        return []
    height, width = len(board), len(board[0])
    def heuristic(cells):
        return sum(abs(r - tr) + abs(c - tc) for (r, c), (tr, tc) in zip(cells, end))
    opened, closed = [start], set()
    g, scores, parents = {start: 0}, {start: heuristic(start)}, {}
    while opened:
        current = min(opened, key=scores.__getitem__)
        if current == end:
            path = [current]
            while current in parents:
                current = parents[current]
                path.append(current)
            return list(reversed(path))
        opened.remove(current)
        closed.add(current)
        # Upstream get_neighbors order is left, right, up, down.
        for dr, dc in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            neighbor = tuple((r + dr, c + dc) for r, c in current)
            if not all(0 <= r < height and 0 <= c < width and
                       board[r][c] in ("-", piece) for r, c in neighbor) or neighbor in closed:
                continue
            cost = g[current] + 1
            if neighbor not in opened or cost < g.get(neighbor, float("inf")):
                parents[neighbor], g[neighbor] = current, cost
                scores[neighbor] = cost + heuristic(neighbor)
                opened.append(neighbor)
    return []


def path_to_slides(piece, path):
    result = []
    for before, after in zip(path, path[1:]):
        dr, dc = after[0][0] - before[0][0], after[0][1] - before[0][1]
        direction = next(key for key, delta in DIRECTIONS.items() if delta == (dr, dc))
        if result and result[-1][1] == direction:
            previous = result[-1]
            result[-1] = (piece, direction, previous[2] + 1)
        else:
            result.append((piece, direction, 1))
    return result


def _prefix(board, target, order):
    count = 0
    for piece in order:
        if positions(board, piece) != positions(target, piece):
            break
        count += 1
    return count


def _distance(board, target, piece):
    current, goal = positions(board, piece)[0], positions(target, piece)[0]
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])


def _route_blockers(board, target, piece):
    """Fewest occupied cells in the two vertical/horizontal L-shaped corridors.

This reproduces the target-zone idea, with a precisely stated cell count for
the benchmark oracle. It does not promise that either corridor is the best
route among all possible detours.
"""
    cells, goal = positions(board, piece), positions(target, piece)[0]
    start = cells[0]
    best = None
    for axes in ((0, 1), (1, 0)):
        shift = [0, 0]
        zone = set(cells)
        for axis in axes:
            end_shift = goal[axis] - start[axis]
            step = 1 if end_shift > shift[axis] else -1
            while shift[axis] != end_shift:
                shift[axis] += step
                zone.update((r + shift[0], c + shift[1]) for r, c in cells)
        blocked = sum(board[r][c] not in ("-", piece) for r, c in zone)
        best = blocked if best is None else min(best, blocked)
    return best


class Game:
    name = "blockstar"
    policy = (
        "Choose a legal BlockStar slide by this exact lexicographic priority: "
        "first maximize ordered_complete (the consecutive target pieces already placed "
        "from the start of target_order); then minimize locked_disturbed (pieces of "
        "the currently completed prefix displaced by this slide); then minimize "
        "active_blocked_cells (other blocks' cells in the better of the active piece's "
        "vertical-then-horizontal and horizontal-then-vertical target corridors); "
        "then minimize active_distance (anchor Manhattan distance of the first "
        "unfinished target piece); then minimize total_distance (sum of all pieces' "
        "anchor Manhattan distances); then minimize slide_distance. These metrics "
        "are measured AFTER the slide. Any remaining exact tie is acceptable. "
        "This is a one-step progress heuristic, not globally optimal puzzle search."
    )

    def __init__(self, seed=42, board=None, target=None, order=None, pieces=None, source_case=None):
        self.seed, self.turn, self.distance_moved = seed, 0, 0
        self.board = [list(row) for row in (board if board is not None else ORIGINAL_START)]
        self.target = [list(row) for row in (target if target is not None else ORIGINAL_TARGET)]
        self.order = list(order if order is not None else ORIGINAL_ORDER)
        self.pieces = list(pieces if pieces is not None else (PIECES if board is None else self.order))
        self.source_case = source_case or "original-20x18"
        self.last_move = None
        self._validate()

    def _validate(self):
        if not self.board or not self.board[0] or len(self.target) != len(self.board):
            raise ValueError("Board and target must have matching nonempty dimensions")
        width = len(self.board[0])
        if any(len(row) != width for row in self.board + self.target):
            raise ValueError("Board and target must have matching rectangular dimensions")
        labels = {cell for row in self.board for cell in row if cell != "-"}
        if labels != set(self.pieces) or labels != set(self.order) or len(self.order) != len(labels):
            raise ValueError("Every block must occur once in pieces and target_order")
        if {cell for row in self.target for cell in row if cell != "-"} != labels:
            raise ValueError("Target and board must contain the same blocks")
        for piece in self.pieces:
            cells, goal = positions(self.board, piece), positions(self.target, piece)
            anchor, goal_anchor = cells[0], goal[0]
            shape = {(r-anchor[0], c-anchor[1]) for r, c in cells}
            goal_shape = {(r-goal_anchor[0], c-goal_anchor[1]) for r, c in goal}
            if shape != goal_shape or len(shape) != (max(r for r,c in shape)+1)*(max(c for r,c in shape)+1):
                raise ValueError("BlockStar supports rigid filled rectangles with unchanged target shapes")

    @property
    def done(self):
        return self.board == self.target

    def _metrics(self, after, distance):
        prefix = _prefix(after, self.target, self.order)
        before_prefix = _prefix(self.board, self.target, self.order)
        locked_disturbed = sum(positions(after,p) != positions(self.target,p) for p in self.order[:before_prefix])
        active = self.order[prefix] if prefix < len(self.order) else None
        return {
            "ordered_complete": prefix,
            "locked_disturbed": locked_disturbed,
            "active_blocked_cells": _route_blockers(after,self.target,active) if active else 0,
            "active_distance": _distance(after,self.target,active) if active else 0,
            "total_distance": sum(_distance(after,self.target,p) for p in self.pieces),
            "slide_distance": distance,
        }

    @staticmethod
    def reference_key(action):
        m = action.metrics
        return (-m["ordered_complete"], m["locked_disturbed"], m["active_blocked_cells"],
                m["active_distance"], m["total_distance"], m["slide_distance"])

    def candidates(self):
        if self.done:
            return []
        actions = []
        for piece in self.pieces:
            for direction, maximum in movable_distances(self.board,piece).items():
                for distance in range(1, maximum+1):
                    after = translate(self.board,piece,direction,distance)
                    metrics = self._metrics(after,distance)
                    actions.append(Action(f"{piece}_{direction}_{distance}",
                                          f"{piece} {direction} {distance}: "
                                          f"prefix={metrics['ordered_complete']} locked={metrics['locked_disturbed']} "
                                          f"blockers={metrics['active_blocked_cells']} active={metrics['active_distance']} "
                                          f"total={metrics['total_distance']} slide={distance}",
                                          metrics,after))
        return actions

    def accepted(self):
        actions = self.candidates()
        if not actions:
            return []
        best = min(map(self.reference_key,actions))
        return [action.id for action in actions if self.reference_key(action) == best]

    def apply(self, action):
        legal = {candidate.id: candidate for candidate in self.candidates()}
        if action.id not in legal or action != legal[action.id]:
            raise ValueError("Action does not match a legal slide on the current board")
        self.board = [list(row) for row in action.after]
        self.turn += 1
        self.distance_moved += action.metrics["slide_distance"]
        self.last_move = action.id

    def observation(self):
        prefix = _prefix(self.board,self.target,self.order)
        return {
            "board_top_to_bottom": ["".join(row) for row in self.board],
            "target_top_to_bottom": ["".join(row) for row in self.target],
            "target_order": list(self.order), "ordered_complete": prefix,
            "active_piece": self.order[prefix] if prefix < len(self.order) else None,
            "source_case": self.source_case,
            "definitions": "Rows run top to bottom, columns left to right. '-' is empty. "
                           "A letter denotes one rigid rectangular block. Each choice slides "
                           "one block, without rotation, through empty cells; intermediate "
                           "collisions are forbidden. All legal positive distances are shown. "
                           "Blocks may legally leave a target; the oracle favors ordered progress. "
                           "Option labels: prefix=ordered_complete, locked=locked_disturbed, "
                           "blockers=active_blocked_cells, active=active_distance, "
                           "total=total_distance, slide=slide_distance.",
        }

    def snapshot(self):
        return {"board":copy.deepcopy(self.board), "target":copy.deepcopy(self.target),
                "order":list(self.order), "pieces":list(self.pieces), "seed":self.seed,
                "turn":self.turn, "distance_moved":self.distance_moved,
                "last_move":self.last_move, "source_case":self.source_case,
                "game":self.name, "status":"solved" if self.done else "playing",
                "ordered_complete":_prefix(self.board,self.target,self.order),
                "score":_prefix(self.board,self.target,self.order)}

    @classmethod
    def restore(cls, snapshot):
        game = cls(snapshot["seed"],snapshot["board"],snapshot["target"],snapshot["order"],
                   snapshot["pieces"],snapshot.get("source_case"))
        game.turn, game.distance_moved = snapshot["turn"],snapshot.get("distance_moved",0)
        game.last_move = snapshot.get("last_move")
        return game


BlockStar = Game


def restore(snapshot):
    return Game.restore(snapshot)


def reference_solve(game, max_nodes=100000, max_seconds=10):
    """Bounded BFS over all legal positive-distance slides for SMALL fixtures.

    A solved result proves the minimum slide count on that fixture. Node/time
    limits return an explicit unfinished status, never a solved claim. This is
    a new baseline, distinct from upstream single-piece A* and random driver.
    """
    started = time.perf_counter()
    initial, goal = _board(game.board), _board(game.target)
    queue, parents = deque([initial]), {initial:None}
    edge, expanded = {},0
    while queue:
        if time.perf_counter()-started >= max_seconds:
            status = "time_limit"
            break
        board = queue.popleft()
        if board == goal:
            moves = []
            while parents[board] is not None:
                moves.append(edge[board])
                board = parents[board]
            return {"status":"solved", "moves":list(reversed(moves)), "slide_count":len(moves),
                    "expanded":expanded, "discovered":len(parents), "elapsed_seconds":time.perf_counter()-started,
                    "method":"bounded exhaustive BFS over all legal multi-block slides"}
        expanded += 1
        for piece in game.pieces:
            for direction, limit in movable_distances(board,piece).items():
                for distance in range(1,limit+1):
                    after = translate(board,piece,direction,distance)
                    if after in parents:
                        continue
                    if len(parents) >= max_nodes:
                        return {"status":"node_limit", "moves":[], "slide_count":None,
                                "expanded":expanded, "discovered":len(parents),
                                "elapsed_seconds":time.perf_counter()-started,
                                "method":"bounded exhaustive BFS over all legal multi-block slides"}
                    parents[after],edge[after] = board,f"{piece}_{direction}_{distance}"
                    queue.append(after)
    else:
        status = "unsolvable"
    return {"status":status,"moves":[],"slide_count":None,"expanded":expanded,
            "discovered":len(parents),"elapsed_seconds":time.perf_counter()-started,
            "method":"bounded exhaustive BFS over all legal multi-block slides"}


def _fixture(case_id, start, target, order, seed):
    return Game(seed,start,target,order,source_case=case_id)


def make_replay(seed=42):
    """Tractable puzzle using upstream movement rules; NOT the original board."""
    return _fixture("fixture-replay-five-blocks",("-----","AABB-","AAECC","DD---"),
                    ("AABB-","AA---","CC-DD","----E"),("A","B","C","D","E"),seed)


def build_suite():
    fixtures = [
        Game(42),
        _fixture("fixture-direct",("-----","B----","-----","-AA--"),
                 ("AA---","-----","-----","----B"),("A","B"),101),
        _fixture("fixture-target-obstructed",("BB---","-----","-A---","-A---"),
                 ("A----","A----","-----","---BB"),("A","B"),102),
        _fixture("fixture-preserve-prefix",("AA---","-C---","BB---","-----"),
                 ("AA---","BB---","-----","----C"),("A","B","C"),103),
        _fixture("fixture-two-corridors",("--B--","-----","--C--","-----","AA---"),
                 ("---AA","-----","-----","-----","B---C"),("A","B","C"),104),
        _fixture("fixture-out-of-order",("----B","--CC-","-----","-AA--"),
                 ("AA--B","-----","-----","--CC-"),("A","B","C"),105),
        _fixture("fixture-final-piece",("AA--B","-----","CC---","--D--"),
                 ("AA--B","D----","CC---","-----"),("A","B","C","D"),106),
        make_replay(107),
    ]
    cases = []
    for game in fixtures:
        cases.append({"id":game.source_case,"game":game.name,"snapshot":game.snapshot(),
                      "accepted":game.accepted(),"candidate_count":len(game.candidates()),
                      "source":"pinned original layout" if game.source_case.startswith("original-")
                               else "independent tractable fixture using pinned movement rules",
                      "evaluation":"agreement with declared one-step ordered-progress policy"})
    return cases
