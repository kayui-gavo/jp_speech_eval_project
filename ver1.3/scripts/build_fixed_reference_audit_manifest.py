from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


FIELDNAMES = [
    "sample_id",
    "audio_path",
    "target_text",
    "audio_type",
    "reference_type",
    "expected_behavior",
    "notes",
]

INPUT_REQUIRED_COLUMNS = ("sample_id", "audio_path", "target_text")

ALLOWED_EXPECTED_BEHAVIORS = {
    "native_should_score_high",
    "clear_learner_should_score",
    "bad_learner_should_not_score_high",
    "content_mismatch_should_not_score",
    "recording_bad_should_not_score",
    "alignment_bad_should_not_score",
    "weak_reference_should_not_score",
    "pitch_unverified_should_suppress_pitch",
}


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


@dataclass(frozen=True)
class ManifestValidationReport:
    total_rows: int
    expected_behavior_counts: Dict[str, int]
    audio_type_counts: Dict[str, int]
    missing_audio_path_count: int
    unknown_expected_behavior_count: int
    warnings: List[str]
    errors: List[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_text(self) -> str:
        lines = [
            f"total rows: {self.total_rows}",
            f"expected_behavior counts: {_format_counts(self.expected_behavior_counts)}",
            f"audio_type counts: {_format_counts(self.audio_type_counts)}",
            f"missing audio_path count: {self.missing_audio_path_count}",
            f"unknown expected_behavior count: {self.unknown_expected_behavior_count}",
        ]
        if self.warnings:
            lines.append("warnings:")
            lines.extend(f"- {item}" for item in self.warnings)
        if self.errors:
            lines.append("errors:")
            lines.extend(f"- {item}" for item in self.errors)
        return "\n".join(lines)


def _format_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{key}:{value}" for key, value in sorted(counts.items())) or "-"


def _dialect(path: Path) -> str:
    return "excel-tab" if path.suffix.lower() == ".tsv" else "excel"


def _read_rows(path: Path) -> Iterable[Mapping[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f, dialect=_dialect(path))


def _fieldnames(path: Path) -> Sequence[str]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, dialect=_dialect(path))
        return tuple(reader.fieldnames or ())


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


def validate_rows(rows: Sequence[Mapping[str, str]], *, strict: bool = False) -> ManifestValidationReport:
    warnings: List[str] = []
    errors: List[str] = []
    missing_audio = 0
    unknown_expected = 0
    expected_counts: Counter[str] = Counter()
    audio_type_counts: Counter[str] = Counter()

    for index, row in enumerate(rows, start=2):
        sample_id = row.get("sample_id") or f"row_{index}"
        audio_path = str(row.get("audio_path") or "").strip()
        audio_type = str(row.get("audio_type") or "").strip()
        reference_type = str(row.get("reference_type") or "").strip()
        expected = str(row.get("expected_behavior") or "").strip()
        expected_counts[expected or "(empty)"] += 1
        audio_type_counts[audio_type or "(empty)"] += 1

        if not audio_type:
            message = f"{sample_id}: audio_type is empty"
            (errors if strict else warnings).append(message)
        if not reference_type:
            message = f"{sample_id}: reference_type is empty"
            (errors if strict else warnings).append(message)
        if expected not in ALLOWED_EXPECTED_BEHAVIORS:
            unknown_expected += 1
            message = f"{sample_id}: unknown expected_behavior={expected or '(empty)'}"
            (errors if strict else warnings).append(message)
        if not audio_path or not Path(audio_path).exists():
            missing_audio += 1
            message = f"{sample_id}: audio_path does not exist: {audio_path or '(empty)'}"
            (errors if strict else warnings).append(message)

    return ManifestValidationReport(
        total_rows=len(rows),
        expected_behavior_counts=dict(expected_counts),
        audio_type_counts=dict(audio_type_counts),
        missing_audio_path_count=missing_audio,
        unknown_expected_behavior_count=unknown_expected,
        warnings=warnings,
        errors=errors,
    )


def build_manifest(input_path: str | Path, output_path: str | Path, *, strict: bool = False) -> ManifestValidationReport:
    input_file = Path(input_path)
    fields = set(_fieldnames(input_file))
    missing_columns = [name for name in INPUT_REQUIRED_COLUMNS if name not in fields]
    if missing_columns:
        raise ValueError(f"missing required columns: {', '.join(missing_columns)}")
    rows = [normalize_row(row) for row in _read_rows(input_file)]
    report = validate_rows(rows, strict=strict)
    if strict and not report.ok:
        return report
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed-reference audit manifest from a small manual list.")
    parser.add_argument("--input", required=True, help="CSV/TSV with sample_id,audio_path,target_text,group,notes.")
    parser.add_argument("--out", help="Output audit manifest CSV. Required unless --dry-run is used.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print a summary without writing a manifest.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero if paths are missing or fields are invalid.")
    args = parser.parse_args()
    try:
        input_file = Path(args.input)
        fields = set(_fieldnames(input_file))
        missing_columns = [name for name in INPUT_REQUIRED_COLUMNS if name not in fields]
        if missing_columns:
            raise ValueError(f"missing required columns: {', '.join(missing_columns)}")
        rows = [normalize_row(row) for row in _read_rows(input_file)]
        report = validate_rows(rows, strict=args.strict)
        print(report.to_text())
        if args.strict and not report.ok:
            raise SystemExit(1)
        if args.dry_run:
            return
        if not args.out:
            raise ValueError("--out is required unless --dry-run is used")
        build_manifest(args.input, args.out, strict=args.strict)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
