"""Exercise append/resume and the frozen protocol without loading any model."""

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class HFFrozenBenchmarkTests(unittest.TestCase):
    def test_clef_append_preserves_prior_records_protocol_and_resume(self):
        # Both game harnesses have a module named benchmark. An isolated
        # interpreter gives this test the same imports as the actual CLI.
        source = Path(__file__).resolve().parent
        code = textwrap.dedent(r'''
            import argparse
            import contextlib
            import copy
            import hashlib
            import io
            import json
            from pathlib import Path
            import re
            import sys
            import tempfile
            from unittest.mock import patch

            source = Path(sys.argv[1])
            sys.path.insert(0, str(source / "hf"))
            import benchmark_hf as runner
            import benchmark
            from game import Game

            calls = []
            def native(request, timeout):
                assert request.full_url == "http://127.0.0.1:11444/v1/systemone"
                body = json.loads(request.data)
                assert body["model"] == "clef-flash:mlx-4bit"
                criteria = body["questions"]["placement"]["criteria"]
                assert 2 <= len(criteria) <= 26
                # Use the disclosed policy to keep game trajectories meaningful.
                def rank(identifier):
                    values = {key: int(value) for key, value in
                              re.findall(r"(\w+)=(-?\d+)", criteria[identifier])}
                    return (values["holes"], -values["lines"], values["peak"],
                            values["total_height"], values["roughness"])
                chosen = min(criteria, key=rank)
                answer = {"type": "choice", "choice": chosen,
                          "probabilities": {key: int(key == chosen) for key in criteria}, "confidence": 1}
                calls.append(body)
                response = {"model": body["model"], "answers": {"placement": answer},
                            "usage": {"input_tokens": 23, "output_tokens": 0, "truncated": False}}
                return io.BytesIO(json.dumps(response).encode())

            suite_bytes = (source / "benchmarks/suite.json").read_bytes()
            suite_hash = hashlib.sha256(suite_bytes).hexdigest()
            old_names = ["laya:en-mps", "kev-latest", "julia-1:mps", "winnow:e4b",
                         "kev:4b-mlx", "laya:multilingual-mps", "clm:8b-mps", "lev:4b-mps"]
            old = [{"name": name, "label": name, "status": "complete", "metadata": {"original": index},
                    "trials": [], "games": [], "summary": {"sentinel": index}}
                   for index, name in enumerate(old_names)]
            previous = {"version": 2, "suite_sha256": suite_hash, "orders": 2, "game_seeds": [42, 123],
                        "game_piece_limit": 40, "scope": old_names[:], "models": copy.deepcopy(old),
                        "status": "complete", "method": "existing frozen protocol"}
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                output = root / "benchmarks/huggingface"
                output.mkdir(parents=True)
                (root / "benchmarks/suite.json").write_bytes(suite_bytes)
                baseline = root / "benchmarks/results.json"
                baseline.write_bytes(b'{"sentinel":"Earlier Ollama evidence must stay untouched"}')
                before_baseline = baseline.read_bytes()
                result_path = output / "results.json"
                result_path.write_text(json.dumps(previous))
                metadata_path = root / "metadata.json"
                metadata_path.write_text(json.dumps({"models": [{"model": "clef-flash:mlx-4bit",
                    "repo": "mlx-community/clef-flash-4bit", "checkpoint": "pinned-fixture",
                    "native_source": "native-fixture", "backend": "native MLX", "device": "gpu",
                    "dtype": "4-bit quantized", "quantization": {"bits": 4},
                    "option_order": "Native encoder sorts choice IDs; tournament group membership can still vary",
                    "versions": {"mlx": "fixture"}, "max_length": 16384, "truncation": False,
                    "loader_sha256": "loader-fixture", "joint_head_sha256": "head-fixture"}]}))
                args = argparse.Namespace(model="clef-flash:mlx-4bit", label="Clef Flash 9B · MLX 4-bit",
                                          endpoint="http://127.0.0.1:11444", metadata=str(metadata_path))
                runner.ROOT = root
                with patch("decision.urlopen", side_effect=native), contextlib.redirect_stdout(io.StringIO()):
                    runner.run(args)
                results = json.loads(result_path.read_text())
                assert results["models"][:-1] == old
                assert results["scope"] == old_names + [args.model]
                assert baseline.read_bytes() == before_baseline
                assert (root / "benchmarks/suite.json").read_bytes() == suite_bytes
                model = results["models"][-1]
                assert model["status"] == "complete"
                assert model["metadata"]["repo"] == "mlx-community/clef-flash-4bit"
                assert model["metadata"]["native_source"] == "native-fixture"
                assert model["metadata"]["quantization"] == {"bits": 4}
                assert model["metadata"]["max_length"] == 16384
                assert model["metadata"]["truncation"] is False
                assert model["metadata"]["loader_sha256"] == "loader-fixture"
                assert model["metadata"]["joint_head_sha256"] == "head-fixture"
                assert model["metadata"]["versions"] == {"mlx": "fixture"}
                assert model["protocol"]["option_order"] == model["metadata"]["option_order"]
                assert model["protocol"]["suite_sha256"] == suite_hash
                assert model["protocol"]["max_choices_per_call"] == 26
                assert model["protocol"]["game_piece_limit"] == 40
                assert len(model["trials"]) == len({(row["case"], row["order"]) for row in model["trials"]}) == 64
                assert {(row["case"], row["order"]) for row in model["trials"]} == {
                    (case["id"], order) for case in json.loads(suite_bytes) for order in (0, 1)}
                assert {game["seed"] for game in model["games"]} == {42, 123}
                assert all(game["pieces"] == 40 and game["status"] == "piece_limit" for game in model["games"])
                warm = Game(98765)
                warm.queue[0] = "O"
                assert model["warmup_snapshot"] == warm.snapshot()
                assert model["warmup"]["calls"][0]["request"]["state"] == benchmark.compact_state(warm.observation())
                # Compare the frozen input groups against the existing measured
                # baseline; final group winners can legitimately differ.
                actual_old = json.loads((source / "benchmarks/results.json").read_text())["models"][0]
                prior = {(trial["case"], trial["order"]): trial for trial in actual_old["trials"]}
                for trial in model["trials"]:
                    expected_calls = prior[(trial["case"], trial["order"])]["decision"]["calls"]
                    actual_calls = trial["decision"]["calls"]
                    expected_first = expected_calls[:-1] if len(expected_calls) > 1 else expected_calls
                    actual_first = actual_calls[:-1] if len(actual_calls) > 1 else actual_calls
                    assert [{key: value for key, value in call["request"].items() if key != "model"}
                            for call in actual_first] == [
                            {key: value for key, value in call["request"].items() if key != "model"}
                            for call in expected_first]
                count = len(calls)
                before_resume = result_path.read_bytes()
                with patch("decision.urlopen", side_effect=native), contextlib.redirect_stdout(io.StringIO()):
                    runner.run(args)
                assert len(calls) == count and result_path.read_bytes() == before_resume
                altered = copy.deepcopy(results)
                altered["orders"] = 1
                result_path.write_text(json.dumps(altered))
                with patch("decision.urlopen", side_effect=native):
                    try:
                        runner.run(args)
                    except AssertionError:
                        pass
                    else:
                        raise AssertionError("Protocol drift must be rejected")
                assert len(calls) == count
                print("Clef append, 64 paired trials, two 40-piece games, warmup, resume and drift checks passed")
        ''')
        result = subprocess.run([sys.executable, "-c", code, str(source)], text=True,
                                capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
