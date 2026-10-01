"""Rule parity and reference certificates for the separate updated-source adapter."""
import hashlib
import json
import unittest

from arcade.blockstar import Game as OldGame, positions, reference_solve, translate
from arcade.blockstar_update import (
    SOURCE_COMMIT, BlockStar, build_updated_suite, make_large_yard,
    make_original, restore, source_metadata,
)


class UpdatedBlockStarTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = source_metadata()
        cls.suite = build_updated_suite()

    def test_source_pin_and_two_official_scenarios(self):
        self.assertEqual(self.metadata["commit"],SOURCE_COMMIT)
        self.assertEqual([c["id"] for c in self.suite],["updated-default","large-yard"])
        self.assertEqual([c["candidate_count"] for c in self.suite],[183,393])
        self.assertEqual(make_original().order,sorted(make_original().pieces))
        self.assertIn("imposes no mandatory target order",BlockStar.policy)

    def test_original_geometry_and_all_legal_slides_stay_identical(self):
        old,new = OldGame(),make_original()
        self.assertEqual(old.board,new.board)
        self.assertEqual(old.target,new.target)
        self.assertEqual({a.id:a.after for a in old.candidates()},
                         {a.id:a.after for a in new.candidates()})
        self.assertNotEqual(old.order,new.order,"Old benchmark contract must stay separate")

    def test_every_updated_initial_endpoint_matches_official_referee_frozen_hash(self):
        evidence = {r["case_id"]:r for r in self.metadata["compatibility"]["updated_source_endpoint_equivalence"]}
        for case in self.suite:
            actions = restore(case["snapshot"]).candidates()
            endpoints = [{"id":a.id,"after":["".join(row) for row in a.after]} for a in actions]
            digest = hashlib.sha256(json.dumps(endpoints,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            self.assertEqual(digest,evidence[case["id"]]["endpoint_sha256"])
            self.assertEqual(len(actions),evidence[case["id"]]["legal_actions"])

    def test_terrain_has_two_components_and_never_enters_action_space(self):
        game = make_large_yard()
        terrain = set(positions(game.board,"#"))
        self.assertEqual((len(game.board),len(game.board[0]),len(game.pieces)),(28,30,24))
        self.assertEqual(len(terrain),48)
        components = []
        while terrain:
            todo,component = [terrain.pop()],set()
            while todo:
                cell = todo.pop()
                component.add(cell)
                r,c = cell
                for neighbor in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                    if neighbor in terrain:
                        terrain.remove(neighbor)
                        todo.append(neighbor)
            components.append(len(component))
        self.assertEqual(sorted(components),[20,28])
        self.assertNotIn("#",game.pieces)
        fixed = positions(game.board,"#")
        for action in game.candidates():
            self.assertFalse(action.id.startswith("#_"))
            self.assertEqual(positions(action.after,"#"),fixed)

    def test_fixed_wall_forces_three_slide_detour(self):
        game = BlockStar(1,("-----","A-#--","-----"),("-----","--#-A","-----"))
        self.assertNotIn("A_right_4",[a.id for a in game.candidates()])
        with self.assertRaises(ValueError):
            translate(game.board,"A","right",4)
        result = reference_solve(game,max_nodes=1000)
        self.assertEqual(result["status"],"solved")
        self.assertEqual(result["slide_count"],3)
        for move in result["moves"]:
            game.apply(next(a for a in game.candidates() if a.id==move))
        self.assertTrue(game.done)

    def test_connected_nonrectangular_shapes_slide_without_rotation(self):
        game = BlockStar(1,("A---","AA--"),("--A-","--AA"))
        game.apply(next(a for a in game.candidates() if a.id=="A_right_2"))
        self.assertTrue(game.done)
        with self.assertRaises(ValueError):
            BlockStar(1,("A-A",),("A-A",))
        with self.assertRaises(ValueError):
            BlockStar(1,("AA","--"),("A-","A-"))

    def test_fixed_terrain_must_match_and_duplicate_piece_labels_are_invalid(self):
        with self.assertRaises(ValueError):
            BlockStar(1,("A-#",),("#-A",))
        with self.assertRaises(ValueError):
            BlockStar(1,("A--",),("--A",),("A",),("A","A"))

    def test_snapshot_serialization_and_source_identity(self):
        game = make_large_yard()
        game.apply(next(a for a in game.candidates() if a.id=="A_right_2"))
        snapshot = json.loads(json.dumps(game.snapshot()))
        self.assertEqual(restore(snapshot).snapshot(),snapshot)
        self.assertEqual(snapshot["source_commit"],SOURCE_COMMIT)
        snapshot["source_commit"] = "another-source"
        with self.assertRaises(ValueError):
            restore(snapshot)

    def test_official_certificate_and_fresh_native_discovery_both_solve_full_boards(self):
        references = {r["case_id"]:r for r in self.metadata["replay_reference"]}
        for case in self.suite:
            game = restore(case["snapshot"])
            official = self.metadata["scenarios"][case["id"]]
            native = references[case["id"]]
            self.assertEqual(len(official["certificate_moves"]),official["certificate_score"])
            for moves in (official["certificate_moves"],native["moves"]):
                board,fixed = game.board,positions(game.board,"#")
                for piece,direction,distance in moves:
                    board = translate(board,piece,direction,distance)
                    self.assertEqual(positions(board,"#"),fixed)
                self.assertEqual([list(row) for row in board],game.target)
            self.assertEqual(native["successes"]+native["failures"],native["trials"])
            self.assertTrue(native["solved"])
            self.assertGreater(native["discovery_elapsed_seconds"],0)
            self.assertGreater(native["certificate_replay_elapsed_seconds"],0)
            self.assertTrue(native["fresh_score_matches_official_certificate"])

    def test_reference_scores_are_measured_and_not_an_optimality_claim(self):
        references = self.metadata["replay_reference"]
        self.assertEqual([r["slide_count"] for r in references],[37,64])
        self.assertEqual([r["coordinate_lower_bound"] for r in references],[30,44])
        self.assertTrue(all(r["slide_count"]>r["coordinate_lower_bound"] for r in references))
        self.assertFalse(references[0]["fresh_moves_match_official_certificate"])
        self.assertTrue(references[1]["fresh_moves_match_official_certificate"])


if __name__ == "__main__":
    unittest.main()
