"""Independent adapter for the updated BlockStar release, kept separate from v1.

The frozen original benchmark keeps its old source pin and ordering contract.
This supplement supports the new permanent terrain and connected rigid shapes.
No upstream implementation or model training code is imported or vendored.
"""
import json
from pathlib import Path

try:
    from .blockstar import Game as OriginalGame, positions
except ImportError:  # Direct CLI execution places arcade/ itself on sys.path.
    from blockstar import Game as OriginalGame, positions


SOURCE_COMMIT = "fed370f11044b56a8fabd449316cdfe0ca12ad6f"
METADATA_PATH = Path(__file__).with_name("blockstar-update-source.json")


def source_metadata():
    return json.loads(METADATA_PATH.read_text())


class UpdatedBlockStar(OriginalGame):
    policy = (
        "For this updated BlockStar decision benchmark, target_order is the sorted "
        "movable-piece labels. This is an explicit evaluation policy; the upstream "
        "solver discovers its own order and imposes no mandatory target order. "
        + OriginalGame.policy
    )

    def __init__(self, seed=42, board=None, target=None, order=None, pieces=None, source_case=None):
        if board is None:
            scenario = source_metadata()["scenarios"]["updated-default"]
            board, target = scenario["start"],scenario["target"]
            source_case = source_case or "updated-default"
        labels = sorted({cell for row in board for cell in row if cell not in ("-","#")})
        super().__init__(seed,board,target,order if order is not None else labels,
                         pieces if pieces is not None else labels,source_case)

    def _validate(self):
        if not self.board or not self.board[0] or len(self.target) != len(self.board):
            raise ValueError("Board and target must have matching nonempty dimensions")
        width = len(self.board[0])
        if any(len(row) != width for row in self.board+self.target):
            raise ValueError("Board and target must have matching rectangular dimensions")
        if any(not isinstance(cell,str) or not cell for row in self.board+self.target for cell in row):
            raise ValueError("Every board cell must be a nonempty string")
        if positions(self.board,"#") != positions(self.target,"#"):
            raise ValueError("Permanent # terrain must occupy identical cells in both boards")
        labels = {cell for row in self.board for cell in row if cell not in ("-","#")}
        target_labels = {cell for row in self.target for cell in row if cell not in ("-","#")}
        if labels != target_labels or labels != set(self.pieces) or labels != set(self.order):
            raise ValueError("Board, target, pieces and target_order must name the same movable blocks")
        if len(self.pieces) != len(labels) or len(self.order) != len(labels):
            raise ValueError("Each block must occur once in pieces and target_order")
        for piece in self.pieces:
            shapes = []
            for board in (self.board,self.target):
                cells = positions(board,piece)
                top,left = min(r for r,c in cells),min(c for r,c in cells)
                shapes.append({(r-top,c-left) for r,c in cells})
            if shapes[0] != shapes[1]:
                raise ValueError("A block must keep the same rigid shape without rotation")
            reached, todo = set(),[next(iter(shapes[0]))]
            while todo:
                cell = todo.pop()
                if cell in reached:
                    continue
                reached.add(cell)
                row,col = cell
                todo.extend(neighbor for neighbor in ((row-1,col),(row+1,col),(row,col-1),(row,col+1))
                            if neighbor in shapes[0] and neighbor not in reached)
            if reached != shapes[0]:
                raise ValueError("A label must denote one connected rigid block")

    def observation(self):
        result = super().observation()
        result["source_commit"] = SOURCE_COMMIT
        result["definitions"] = result["definitions"].replace(
            "one rigid rectangular block", "one connected rigid block"
        ) + " '#' cells are permanent terrain; they cannot move or be crossed."
        return result

    def snapshot(self):
        result = super().snapshot()
        result["source_commit"] = SOURCE_COMMIT
        result["fixed_cells"] = [list(cell) for cell in positions(self.board,"#")]
        return result

    @classmethod
    def restore(cls, snapshot):
        source = snapshot.get("source_commit",SOURCE_COMMIT)
        if source != SOURCE_COMMIT:
            raise ValueError("Updated BlockStar snapshot belongs to a different source revision")
        return super().restore(snapshot)


BlockStar = UpdatedBlockStar


def restore(snapshot):
    return UpdatedBlockStar.restore(snapshot)


def _make(case_id, seed):
    scenario = source_metadata()["scenarios"][case_id]
    return UpdatedBlockStar(seed,scenario["start"],scenario["target"],source_case=case_id)


def make_original(seed=42):
    return _make("updated-default",seed)


def make_large_yard(seed=42):
    return _make("large-yard",seed)


def build_updated_suite():
    cases = []
    for game in (make_original(),make_large_yard()):
        actions = game.candidates()
        cases.append({"id":game.source_case,"game":game.name,"snapshot":game.snapshot(),
                      "accepted":game.accepted(),"candidate_count":len(actions),
                      "source":f"official updated BlockStar scenario at {SOURCE_COMMIT}",
                      "source_commit":SOURCE_COMMIT,
                      "evaluation":"one-step agreement with explicitly declared sorted-goal-prefix heuristic"})
    return cases


build_suite = build_updated_suite


def replay_reference():
    """Recorded source discovery and certificate replay, with distinct time fields."""
    return source_metadata()["replay_reference"]
