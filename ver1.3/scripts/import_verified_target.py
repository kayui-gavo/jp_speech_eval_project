from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.target_specs import (
    build_verified_target_entry,
    parse_ints,
    parse_pitch_labels,
    validate_target_spec,
)
from jp_speech_eval.text_frontend import build_text_info, split_mora, text_to_kana
from jp_speech_eval.verified_targets import default_verified_targets_path, load_verified_targets


def _print_openjtalk_default(text: str) -> None:
    info = build_text_info(text)
    print("OpenJTalk draft")
    print(f"  Kana         : {info.kana}")
    print(f"  Mora         : {'・'.join(info.moras)}")
    print(f"  Target pitch : {' '.join(info.target_pitch)}")
    print(f"  Pitch source : {info.pitch_target_source}")
    if info.accent_phrases:
        print("  Accent phrases:")
        for idx, phrase in enumerate(info.accent_phrases, start=1):
            moras = "・".join(phrase.get("moras") or [])
            print(f"    {idx}. {moras}  accent={phrase.get('accent_position')}")
    print()


def _summarize_entry(entry: Dict[str, Any]) -> None:
    validation = validate_target_spec(entry)
    print("Verified target candidate")
    print(f"  Text         : {entry['text']}")
    print(f"  Kana         : {entry['kana']}")
    print(f"  Mora         : {'・'.join(entry['moras'])}")
    print(f"  Target pitch : {' '.join(entry['target_pitch'])}")
    print(f"  Source       : {entry['pitch_target_source']}")
    if entry.get("accent_phrases"):
        print("  Accent phrases:")
        for idx, phrase in enumerate(entry["accent_phrases"], start=1):
            moras = "・".join(phrase.get("moras") or [])
            print(f"    {idx}. {moras}  accent={phrase.get('accent_position')}")
    special = entry.get("special_mora", {}).get("by_type", {})
    print(f"  Special mora : {json.dumps(special, ensure_ascii=False, sort_keys=True)}")
    if validation.warnings:
        print("  Warnings     :")
        for warning in validation.warnings:
            print(f"    - {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Import a manually verified Japanese pronunciation target. "
            "Use this after checking OJAD/Suzuki-kun or another trusted source; "
            "the script validates the schema but does not query OJAD automatically."
        )
    )
    parser.add_argument("--text", required=True, help="Exact sentence text used for lookup")
    parser.add_argument("--kana", default=None, help="Optional katakana reading; defaults to pyopenjtalk g2p")
    parser.add_argument("--target-pitch", required=True, help='H/L labels, e.g. "L H H L" or "L,H,H,L"')
    parser.add_argument("--source", default="ojad_verified", help="Provenance label, e.g. ojad_verified/manual")
    parser.add_argument("--phrase-lengths", default=None, help="Optional comma-separated mora counts per accent phrase")
    parser.add_argument("--accent-positions", default=None, help="Optional comma-separated accent positions per phrase")
    parser.add_argument("--note", default=None, help="Optional provenance note")
    parser.add_argument("--source-url", default=None, help="Optional OJAD or dictionary URL")
    parser.add_argument("--verified-by", default=None, help="Optional verifier name")
    parser.add_argument("--out", default=None, help="Target JSON path; defaults to configs/verified_accent_targets.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print without writing")
    parser.add_argument("--show-openjtalk-default", action="store_true", help="Print the current automatic target first")
    args = parser.parse_args()

    if args.show_openjtalk_default:
        _print_openjtalk_default(args.text)

    kana = args.kana or text_to_kana(args.text)
    moras = split_mora(kana)
    target_pitch = parse_pitch_labels(args.target_pitch)
    if len(target_pitch) != len(moras):
        raise ValueError(
            f"target pitch count ({len(target_pitch)}) does not match mora count ({len(moras)}): "
            f"{'・'.join(moras)}"
        )

    entry = build_verified_target_entry(
        text=args.text,
        kana=kana,
        target_pitch=target_pitch,
        source=args.source,
        phrase_lengths=parse_ints(args.phrase_lengths),
        accent_positions=parse_ints(args.accent_positions),
        note=args.note,
        source_url=args.source_url,
        verified_by=args.verified_by,
    )
    _summarize_entry(entry)

    if args.dry_run:
        print("\nDry run only; no file was written.")
        return

    out_path = Path(args.out) if args.out else default_verified_targets_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_verified_targets(out_path)
    data[args.text] = entry
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nSaved verified target: {out_path}")


if __name__ == "__main__":
    main()
