from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from jp_speech_eval.text_frontend import split_mora, text_to_kana
from jp_speech_eval.verified_targets import default_verified_targets_path, load_verified_targets


def _parse_ints(raw: str | None) -> List[int]:
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _build_accent_phrases(
    moras: List[str],
    phrase_lengths: List[int],
    accent_positions: List[int],
) -> List[Dict[str, Any]]:
    if not phrase_lengths:
        return []
    if sum(phrase_lengths) != len(moras):
        raise ValueError("phrase lengths must sum to the mora count")
    if accent_positions and len(accent_positions) != len(phrase_lengths):
        raise ValueError("accent positions must match the number of phrase lengths")
    if not accent_positions:
        accent_positions = [0 for _ in phrase_lengths]

    out: List[Dict[str, Any]] = []
    offset = 0
    for length, accent_position in zip(phrase_lengths, accent_positions):
        phrase_moras = moras[offset: offset + length]
        if accent_position < 0 or accent_position > len(phrase_moras):
            raise ValueError("accent position must be within the phrase mora count")
        out.append({
            "words": [],
            "moras": phrase_moras,
            "accent_position": accent_position,
            "chain_flags": [],
            "chain_rules": [],
        })
        offset += length
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a manually verified Japanese accent target.")
    parser.add_argument("--text", required=True, help="Exact sentence text used for lookup")
    parser.add_argument("--kana", default=None, help="Optional katakana reading; defaults to pyopenjtalk g2p")
    parser.add_argument("--target-pitch", required=True, help='Space-separated H/L labels, e.g. "L H H L"')
    parser.add_argument("--source", default="ojad_verified", help="Provenance label, e.g. ojad_verified")
    parser.add_argument("--phrase-lengths", default=None, help="Optional comma-separated mora counts per accent phrase")
    parser.add_argument("--accent-positions", default=None, help="Optional comma-separated accent positions per phrase")
    parser.add_argument("--note", default=None, help="Optional provenance note")
    parser.add_argument("--out", default=None, help="Target JSON path; defaults to configs/verified_accent_targets.json")
    args = parser.parse_args()

    kana = args.kana or text_to_kana(args.text)
    moras = split_mora(kana)
    target_pitch = [part.strip().upper() for part in args.target_pitch.split() if part.strip()]
    if any(label not in {"H", "L"} for label in target_pitch):
        raise ValueError("target pitch must contain only H or L labels")
    if len(target_pitch) != len(moras):
        raise ValueError(f"target pitch count ({len(target_pitch)}) does not match mora count ({len(moras)})")

    phrase_lengths = _parse_ints(args.phrase_lengths)
    accent_positions = _parse_ints(args.accent_positions)
    accent_phrases = _build_accent_phrases(moras, phrase_lengths, accent_positions)

    out_path = Path(args.out) if args.out else default_verified_targets_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_verified_targets(out_path)
    entry: Dict[str, Any] = {
        "kana": kana,
        "moras": moras,
        "target_pitch": target_pitch,
        "pitch_target_source": args.source,
    }
    if accent_phrases:
        entry["accent_phrases"] = accent_phrases
    if args.note:
        entry["note"] = args.note
    data[args.text] = entry
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Saved verified accent target: {out_path}")
    print(f"Text         : {args.text}")
    print(f"Kana         : {kana}")
    print(f"Mora         : {'・'.join(moras)}")
    print(f"Target pitch : {' '.join(target_pitch)}")
    print(f"Source       : {args.source}")


if __name__ == "__main__":
    main()
