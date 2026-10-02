"""Protect the published Clef identity without importing MLX or model weights."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class ClefCheckpointTests(unittest.TestCase):
    def test_manifest_accepts_pinned_files_and_rejects_checkpoint_changes(self):
        source = Path(__file__).resolve().parent / "hf/clef_runtime/server.py"
        spec = importlib.util.spec_from_file_location("clef_adapter_test", source)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            contents = {"config.json": b'{"bits":4}',
                        "joint_head.safetensors": b"original native head",
                        "model-00001-of-00001.safetensors": b"original backbone shard"}
            siblings = []
            for name, content in contents.items():
                (checkpoint / name).write_bytes(content)
                entry = {"rfilename": name, "size": len(content)}
                if name.endswith(".safetensors"):
                    entry["lfs"] = {"sha256": hashlib.sha256(content).hexdigest()}
                else:
                    entry["blobId"] = hashlib.sha1(
                        f"blob {len(content)}\0".encode() + content).hexdigest()
                siblings.append(entry)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"id": adapter.REPO, "sha": adapter.REVISION,
                                            "siblings": siblings}))
            with patch.object(adapter, "SOURCE_RECORD", manifest):
                hashes = adapter.verify_checkpoint(checkpoint)
                self.assertEqual(hashes, {name: hashlib.sha256(content).hexdigest()
                                          for name, content in contents.items()})
                # Same-size changes must fail for both LFS weights and Git blobs.
                for name, content in contents.items():
                    with self.subTest(changed=name):
                        (checkpoint / name).write_bytes(b"!" + content[1:])
                        with self.assertRaisesRegex(RuntimeError, "content mismatch"):
                            adapter.verify_checkpoint(checkpoint)
                        (checkpoint / name).write_bytes(content)
                extra = checkpoint / "model-extra.safetensors"
                extra.write_bytes(b"unlisted weights")
                with self.assertRaisesRegex(RuntimeError, "inference files differ"):
                    adapter.verify_checkpoint(checkpoint)
                extra.unlink()
                (checkpoint / "joint_head.safetensors").unlink()
                with self.assertRaisesRegex(RuntimeError, "inference files differ"):
                    adapter.verify_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
