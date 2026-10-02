"""Benchmark the explicit Clef Flash MLX 4-bit runtime using frozen Tetris inputs.

The native server must already be ready. This launcher never loads a replacement
model, changes the decision grouping, or regenerates the benchmark suite.
"""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
MODEL = "clef-flash:mlx-4bit"
LABEL = "Clef Flash 9B · MLX 4-bit"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Captured /v1/models readiness response and pinned native provenance")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11444")
    args = parser.parse_args()
    # A separate interpreter keeps the Tetris benchmark module distinct from
    # arcade/benchmark.py, whose protocol intentionally uses smaller groups.
    subprocess.run([sys.executable, str(ROOT / "benchmark_hf.py"),
                    "--model", MODEL, "--label", LABEL, "--endpoint", args.endpoint,
                    "--metadata", str(args.metadata.resolve())], check=True)


if __name__ == "__main__":
    main()
