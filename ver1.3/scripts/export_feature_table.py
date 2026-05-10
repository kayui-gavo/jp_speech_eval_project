from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.evaluation_log import export_feature_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert evaluation JSONL logs into a training CSV feature table.")
    parser.add_argument("--jsonl", nargs="+", required=True, help="One or more evaluation JSONL files.")
    parser.add_argument("--csv", required=True, help="Output CSV path.")
    args = parser.parse_args()

    count = export_feature_table(args.jsonl, args.csv)
    print(f"Exported {count} rows to {args.csv}")


if __name__ == "__main__":
    main()
