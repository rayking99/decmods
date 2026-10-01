"""Portable rule checks plus frozen, pinned-upstream equivalence evidence."""
import hashlib
import json
from pathlib import Path
import unittest

from arcade.blockstar import (
    Action, Game, ORIGINAL_ORDER, PIECES, build_suite, make_replay,
    movable_distances, positions, reference_solve, restore, translate,
    upstream_piece_path,
)


PROVENANCE = json.loads(Path(__file__).with_name("blockstar-source.json").read_text())
EQUIVALENCE = PROVENANCE["compatibility"]["source_equivalence"]


class BlockStarRulesTest(unittest.TestCase):
    def test_original_layout_and_source_target_order(self):
        game = Game()
        self.assertEqual((len(game.board),len(game.board[0])),(20,18))
        self.assertEqual(game.order,list(ORIGINAL_ORDER))
        self.assertEqual(len(game.pieces),16)
        self.assertNotEqual(game.order,list(PIECES))
        for piece in game.pieces:
            self.assertEqual(movable_distances(game.board,piece),EQUIVALENCE["movable_distances"][piece])

    def test_every_original_legal_action_matches_pinned_source_endpoint_hash(self):
        actions = Game().candidates()
        self.assertEqual(len(actions),EQUIVALENCE["original_initial_legal_count"])
        endpoints = [{"id":a.id,"after":["".join(row) for row in a.after]} for a in actions]
        serialized = json.dumps(endpoints,sort_keys=True,separators=(",",":")).encode()
        self.assertEqual(hashlib.sha256(serialized).hexdigest(),
                         EQUIVALENCE["original_initial_endpoint_sha256"])
        self.assertEqual(len({a.id for a in actions}),len(actions))

    def test_pinned_upstream_single_piece_astar_path_equivalence(self):
        game = Game()
        for piece,expected in EQUIVALENCE["piece_paths"].items():
            path = upstream_piece_path(game.board,game.target,piece)
            self.assertEqual(len(path),expected["positions_in_path"],piece)
            digest = hashlib.sha256(json.dumps(path,separators=(",",":")).encode()).hexdigest()
            self.assertEqual(digest,expected["path_sha256"],piece)

    def test_all_measured_upstream_full_puzzle_traces_replay_and_solve(self):
        for trial in PROVENANCE["upstream_measured_baseline"]["trials"]:
            board = Game().board
            for piece,direction,distance in trial["moves"]:
                board = translate(board,piece,direction,distance)
            self.assertEqual([list(row) for row in board],Game().target,trial["seed"])
            self.assertEqual(len(trial["moves"]),trial["move_count"])

    def test_swept_collision_prevents_jumping_over_block(self):
        board = [list("A--B-")]
        self.assertEqual(movable_distances(board,"A"),{"right":2})
        with self.assertRaises(ValueError):
            translate(board,"A","right",4)
        self.assertEqual(translate(board,"A","right",2),(tuple("--AB-"),))

    def test_rigid_block_preserves_shape_through_every_slide(self):
        game = Game()
        for action in game.candidates():
            piece = action.id.split("_")[0]
            before,after = positions(game.board,piece),positions(action.after,piece)
            shape_before = {(r-before[0][0],c-before[0][1]) for r,c in before}
            shape_after = {(r-after[0][0],c-after[0][1]) for r,c in after}
            self.assertEqual(shape_before,shape_after,action.id)

    def test_boundary_zero_negative_and_unknown_moves_are_rejected(self):
        for direction,distance in (("up",1),("right",0),("right",-1),("rotate",1),("right",1.5)):
            with self.assertRaises(ValueError):
                translate([list("A--")],"A",direction,distance)

    def test_targets_are_legally_movable_and_policy_protects_completed_prefix(self):
        case = next(c for c in build_suite() if c["id"]=="fixture-preserve-prefix")
        game = restore(case["snapshot"])
        moves = game.candidates()
        displaced = [a for a in moves if a.id.startswith("A_")]
        self.assertTrue(displaced,"Upstream permits moving pieces already in target")
        self.assertTrue(all(a.metrics["locked_disturbed"]==1 for a in displaced))
        self.assertTrue(all(a.id not in game.accepted() for a in displaced))

    def test_oracle_retains_every_exact_policy_tie(self):
        case = next(c for c in build_suite() if c["id"]=="fixture-two-corridors")
        game = restore(case["snapshot"])
        self.assertEqual(game.accepted(),["A_up_3","A_right_3"])
        best = min(map(game.reference_key,game.candidates()))
        self.assertEqual(game.accepted(),[a.id for a in game.candidates() if game.reference_key(a)==best])

    def test_snapshot_roundtrip_and_reject_stale_or_forged_action(self):
        game = make_replay()
        action = min(game.candidates(),key=game.reference_key)
        game.apply(action)
        snapshot = json.loads(json.dumps(game.snapshot()))
        restored = restore(snapshot)
        self.assertEqual(restored.snapshot(),snapshot)
        self.assertEqual(restored.candidates(),game.candidates())
        with self.assertRaises(ValueError):
            game.apply(action)
        current = game.candidates()[0]
        with self.assertRaises(ValueError):
            game.apply(Action(current.id,current.text,current.metrics,tuple()))

    def test_benchmark_has_one_original_and_seven_frozen_tractable_cases(self):
        suite = build_suite()
        self.assertEqual(len(suite),8)
        self.assertEqual(len({c["id"] for c in suite}),8)
        self.assertEqual(sum(c["source"]=="pinned original layout" for c in suite),1)
        self.assertEqual(suite,build_suite())
        for case in suite:
            game = restore(case["snapshot"])
            self.assertEqual(case["accepted"],game.accepted())
            self.assertEqual(case["candidate_count"],len(game.candidates()))
            self.assertTrue(case["accepted"])

    def test_replay_oracle_finishes_nine_slides_bfs_proves_minimum_seven(self):
        game = make_replay()
        result = reference_solve(game,max_nodes=20000,max_seconds=10)
        self.assertEqual(result["status"],"solved")
        self.assertEqual(result["slide_count"],7)
        optimal = make_replay()
        for move_id in result["moves"]:
            optimal.apply(next(a for a in optimal.candidates() if a.id==move_id))
        self.assertTrue(optimal.done)
        for turn in range(9):
            self.assertFalse(game.done)
            game.apply(min(game.candidates(),key=game.reference_key))
        self.assertTrue(game.done)
        self.assertEqual(game.snapshot()["status"],"solved")
        self.assertEqual(game.accepted(),[])
        self.assertEqual(game.candidates(),[])

    def test_reference_limits_are_unfinished_outcomes(self):
        limited = reference_solve(make_replay(),max_nodes=1)
        self.assertEqual(limited["status"],"node_limit")
        self.assertIsNone(limited["slide_count"])
        self.assertEqual(limited["moves"],[])
        timed = reference_solve(make_replay(),max_seconds=0)
        self.assertEqual(timed["status"],"time_limit")

    def test_invalid_or_changed_target_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            Game(1,("AA-","---"),("A--","A--"),("A",))
        with self.assertRaises(ValueError):
            Game(1,("A-","--"),("A-","--"),("A","A"))


if __name__ == "__main__":
    unittest.main()
