#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent


def _count(root: Path, patterns: tuple[str, ...]) -> int:
    if not root.exists():
        return 0
    total = 0
    for pattern in patterns:
        total += sum(1 for _ in root.rglob(pattern))
    return total


def inspect_resources(jvs_root: Path, janon_root: Path) -> dict:
    return {
        "jvs": {
            "exists": jvs_root.exists(),
            "transcripts": _count(jvs_root, ("transcripts_utf8.txt", "*.txt")),
            "phone_labels": _count(jvs_root, ("*.lab", "*.phones", "*.phn")),
            "textgrids": _count(jvs_root, ("*.TextGrid", "*.textgrid")),
            "duration_labels": _count(jvs_root, ("*duration*",)),
        },
        "janon": {
            "exists": janon_root.exists(),
            "transcripts": 1 if (janon_root / "data.csv").exists() else 0,
            "phone_labels": _count(janon_root, ("*.lab", "*.phones", "*.phn")),
            "textgrids": _count(janon_root, ("*.TextGrid", "*.textgrid")),
            "duration_labels": _count(janon_root, ("*duration*",)),
        },
        "project": {
            "textgrid_parser": (ROOT / "src" / "jp_speech_eval" / "alignment_evidence" / "textgrid_parser.py").exists(),
            "alignment_cache_dirs": [str(path) for path in (ROOT / "results").glob("*alignment*")],
            "mfa_command": shutil.which("mfa") or "",
        },
    }


def write_report(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Alignment resource audit",
        "",
        "This report checks whether local datasets already contain phone/mora alignment labels before running MFA.",
        "",
        "## JVS",
        f"- exists: {data['jvs']['exists']}",
        f"- transcripts/text files: {data['jvs']['transcripts']}",
        f"- phone labels: {data['jvs']['phone_labels']}",
        f"- TextGrid files: {data['jvs']['textgrids']}",
        f"- duration labels: {data['jvs']['duration_labels']}",
        "",
        "## JANON",
        f"- exists: {data['janon']['exists']}",
        f"- transcript manifest: {data['janon']['transcripts']}",
        f"- phone labels: {data['janon']['phone_labels']}",
        f"- TextGrid files: {data['janon']['textgrids']}",
        f"- duration labels: {data['janon']['duration_labels']}",
        "",
        "## Project",
        f"- TextGrid parser available: {data['project']['textgrid_parser']}",
        f"- alignment cache dirs: {data['project']['alignment_cache_dirs']}",
        f"- MFA command: {data['project']['mfa_command'] or 'not found'}",
        "",
        "## Conclusion",
    ]
    if data["jvs"]["textgrids"] or data["jvs"]["phone_labels"]:
        lines.append("- Existing JVS labels may be usable. Prefer existing_label adapter first.")
    else:
        lines.append("- No per-utterance JVS phone/TextGrid labels found. Use MFA as optional offline backend, or fall back to non-threshold coverage audit.")
    if not data["project"]["mfa_command"]:
        lines.append("- MFA is not installed in the current environment, so MFA alignment should be reported as skipped.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = inspect_resources(PROJECT_ROOT / "JVS", PROJECT_ROOT / "JANON")
    report = ROOT / "reports" / "alignment_resource_audit.md"
    write_report(data, report)
    print(json.dumps({"ok": True, "report": str(report), "data": data}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
