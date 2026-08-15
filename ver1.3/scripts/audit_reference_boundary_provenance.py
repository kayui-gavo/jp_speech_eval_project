from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def _fallback_provenance(method: str) -> tuple[float, str, str]:
    name = str(method or "").strip().lower()
    if name in {"existing_label", "external_lab", "external_textgrid", "mfa_japanese"} or "verified" in name:
        return 0.90, "verified_phone_alignment", name or "metadata_inferred"
    if "audio_query" in name or "engine_duration" in name:
        return 0.65, "engine_duration_prior", name or "metadata_inferred"
    if "equal" in name:
        return 0.15, "equal_fallback", name or "metadata_inferred"
    return 0.35, "unknown_alignment", name or "metadata_inferred"


def _looks_like_sentence_cache(raw: Dict[str, Any]) -> bool:
    return (
        isinstance(raw, dict)
        and isinstance(raw.get("moras"), list)
        and isinstance(raw.get("ref_mora_boundaries"), list)
        and "text" in raw
        and "sr" in raw
    )


def inspect_cache_json(path: str | Path) -> Dict[str, Any] | None:
    file_path = Path(path)
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not _looks_like_sentence_cache(raw):
        return None

    method = str(raw.get("ref_boundary_method") or "equal_mora")
    default_conf, default_tier, default_source = _fallback_provenance(method)
    confidence = float(raw.get("ref_boundary_confidence", default_conf))
    tier = str(raw.get("ref_boundary_tier") or default_tier)
    source = raw.get("ref_boundary_source") or default_source
    mora_count = len(raw.get("moras") or [])
    boundary_count = len(raw.get("ref_mora_boundaries") or [])
    one_to_one = mora_count > 0 and mora_count == boundary_count

    if tier == "verified_phone_alignment" and confidence >= 0.80 and one_to_one:
        migration_state = "verified"
    elif tier == "engine_duration_prior" and one_to_one:
        migration_state = "intermediate"
    elif tier == "equal_fallback":
        migration_state = "needs_alignment_migration"
    else:
        migration_state = "review_required"

    return {
        "cache_json": str(file_path),
        "text": str(raw.get("text") or ""),
        "reference_id": str(raw.get("reference_id") or ""),
        "reference_source": str(raw.get("reference_source") or ""),
        "boundary_method": method,
        "boundary_tier": tier,
        "boundary_confidence": round(confidence, 4),
        "boundary_source": str(source or ""),
        "mora_count": mora_count,
        "boundary_count": boundary_count,
        "one_to_one": one_to_one,
        "migration_state": migration_state,
    }


def _iter_json_files(roots: Sequence[str | Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root_value in roots:
        root = Path(root_value)
        if root.is_file() and root.suffix.lower() == ".json":
            candidates = [root]
        elif root.is_dir():
            candidates = sorted(root.rglob("*.json"))
        else:
            continue
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def audit_reference_boundaries(roots: Sequence[str | Path]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in _iter_json_files(roots):
        row = inspect_cache_json(path)
        if row is not None:
            rows.append(row)

    tier_counts = Counter(str(row["boundary_tier"]) for row in rows)
    state_counts = Counter(str(row["migration_state"]) for row in rows)
    verified = int(state_counts.get("verified", 0))
    total = len(rows)
    summary = {
        "cache_count": total,
        "verified_count": verified,
        "verified_rate": round(verified / total, 4) if total else 0.0,
        "tier_counts": dict(sorted(tier_counts.items())),
        "migration_state_counts": dict(sorted(state_counts.items())),
        "ready_for_precise_local_feedback_count": verified,
        "precise_local_feedback_policy": "verified_phone_alignment_only",
    }
    return rows, summary


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cache_json",
        "text",
        "reference_id",
        "reference_source",
        "boundary_method",
        "boundary_tier",
        "boundary_confidence",
        "boundary_source",
        "mora_count",
        "boundary_count",
        "one_to_one",
        "migration_state",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory sentence-cache reference-boundary provenance without running models."
    )
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="Cache JSON file or directory. Repeat for multiple roots.",
    )
    parser.add_argument("--csv", default=None, help="Optional per-cache CSV output path.")
    parser.add_argument("--json", dest="json_out", default=None, help="Optional summary JSON output path.")
    args = parser.parse_args()

    roots = args.root or ["assets/reference_cache", "cache"]
    rows, summary = audit_reference_boundaries(roots)

    if args.csv:
        _write_csv(Path(args.csv), rows)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for row in rows:
        print(
            f"[{row['migration_state']}] {row['boundary_tier']:<26} "
            f"conf={float(row['boundary_confidence']):.2f} "
            f"mora={row['mora_count']}/{row['boundary_count']} "
            f"{row['cache_json']}"
        )


if __name__ == "__main__":
    main()
