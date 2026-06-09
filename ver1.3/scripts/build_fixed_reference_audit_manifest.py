from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping


FIELDNAMES = [
    "sample_id",
    "audio_path",
    "target_text",
    "audio_type",
    "reference_type",
    "expected_behavior",
    "notes",
]


GROUP_TO_EXPECTED_BEHAVIOR: Dict[str, str] = {
    "jvs_native_clear": "native_should_score_high",
    "jvs_native_same_text_diff_speaker": "native_should_score_high",
    "janon_learner_clear": "clear_learner_should_score",
    "janon_learner_bad": "bad_learner_should_not_score_high",
    "self_recording_clear": "clear_learner_should_score",
    "self_recording_bad_pronunciation": "bad_learner_should_not_score_high",
    "wrong_japanese_sentence": "content_mismatch_should_not_score",
    "english_or_chinese_speech": "content_mismatch_should_not_score",
    "partial_target": "content_mismatch_should_not_score",
    "random_speech": "content_mismatch_should_not_score",
    "noise_or_silence": "recording_bad_should_not_score",
    "fallback_alignment_case": "alignment_bad_should_not_score",
    "weak_reference": "weak_reference_should_not_score",
    "pitch_unverified": "pitch_unverified_should_suppress_pitch",
}


GROUP_TO_REFERENCE_TYPE: Dict[str, str] = {
    "jvs_native_clear": "human_reference",
    "jvs_native_same_text_diff_speaker": "human_reference",
    "weak_reference": "weak_reference",
}


def _read_rows(path: Path) -> Iterable[Mapping[str, str]]:
    dialect = "excel-tab" if path.suffix.lower() == ".tsv" else "excel"
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f, dialect=dialect)


def normalize_row(row: Mapping[str, str]) -> Dict[str, str]:
    group = (row.get("group") or row.get("audio_type") or "").strip()
    expected = (row.get("expected_behavior") or GROUP_TO_EXPECTED_BEHAVIOR.get(group, "")).strip()
    reference_type = (row.get("reference_type") or GROUP_TO_REFERENCE_TYPE.get(group, "tts_reference")).strip()
    return {
        "sample_id": (row.get("sample_id") or Path(row.get("audio_path", "")).stem).strip(),
        "audio_path": (row.get("audio_path") or "").strip(),
        "target_text": (row.get("target_text") or "").strip(),
        "audio_type": group,
        "reference_type": reference_type,
        "expected_behavior": expected,
        "notes": (row.get("notes") or "").strip(),
    }


def build_manifest(input_path: str | Path, output_path: str | Path) -> None:
    rows = [normalize_row(row) for row in _read_rows(Path(input_path))]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed-reference audit manifest from a small manual list.")
    parser.add_argument("--input", required=True, help="CSV/TSV with sample_id,audio_path,target_text,group,notes.")
    parser.add_argument("--out", required=True, help="Output audit manifest CSV.")
    args = parser.parse_args()
    build_manifest(args.input, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
