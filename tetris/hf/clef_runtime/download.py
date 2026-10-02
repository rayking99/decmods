"""Download the exact Clef Flash MLX variant evaluated by Decision Lab."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import HF_ROOT, REPO, REVISION


def main():
    from huggingface_hub import snapshot_download
    target = snapshot_download(REPO, revision=REVISION,
        local_dir=HF_ROOT / "models/clef-flash-4bit", token=False,
        ignore_patterns=["__pycache__/*", "*.pyc"])
    print(target)


if __name__ == "__main__":
    main()
