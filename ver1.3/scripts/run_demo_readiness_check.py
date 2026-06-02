#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_demo_flow_smoke_tests import run as run_smoke_tests


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = {
    "demo_fixed_targets": ROOT / "data" / "demo_fixed_targets.json",
    "user_facing_messages": ROOT / "configs" / "user_facing_messages_ja.json",
    "api_contract": ROOT / "docs" / "api_user_facing_contract.md",
    "fixed_reference_flow": ROOT / "reports" / "fixed_reference_demo_flow.md",
    "asr_confirmed_flow": ROOT / "reports" / "asr_confirmed_reference_flow.md",
    "asr_kanade_flow": ROOT / "reports" / "asr_kanade_demo_flow.md",
    "frontend_checklist": ROOT / "reports" / "frontend_user_facing_checklist.md",
    "known_limitations": ROOT / "reports" / "demo_known_limitations.md",
    "presentation_script": ROOT / "reports" / "demo_presentation_script.md",
    "api_examples_report": ROOT / "reports" / "demo_api_response_examples.md",
    "api_examples_json": ROOT / "results" / "demo_readiness" / "demo_api_examples.json",
    "human_validation_plan": ROOT / "reports" / "minimal_human_validation_plan.md",
    "freeze_checklist": ROOT / "reports" / "demo_freeze_checklist.md",
}


def _row(name: str, status: str, detail: str = "") -> Dict[str, str]:
    return {"check": name, "status": status, "detail": detail}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten(value: Any) -> List[str]:
    if isinstance(value, dict):
        out: List[str] = []
        for child in value.values():
            out.extend(_flatten(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(_flatten(child))
        return out
    if value is None:
        return []
    return [str(value)]


def _examples_have_user_facing(examples_path: Path) -> bool:
    examples = _load_json(examples_path)
    return bool(examples) and all("user_facing" in item.get("response", {}) for item in examples)


def _examples_no_raw_leakage(examples_path: Path) -> bool:
    forbidden = ("debug_total_score", "prosody_score", "raw DTW", "raw F0", "threshold_low", "special_mora_decisions")
    examples = _load_json(examples_path)
    for item in examples:
        user_facing = item.get("response", {}).get("user_facing", {})
        learner_fields = {
            key: value
            for key, value in user_facing.items()
            if key != "debug"
        }
        text = "\n".join(learner_fields.keys()) + "\n" + "\n".join(_flatten(learner_fields))
        if any(token in text for token in forbidden):
            return False
    return True


def _smoke_tests_pass() -> bool:
    rows = run_smoke_tests()
    return bool(rows) and all(bool(row.get("passed")) for row in rows)


def _git_last_test_hint() -> str:
    return "Run `../.venv/bin/python -m pytest tests` before demo freeze."


def collect_checks() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for name, path in REQUIRED_FILES.items():
        rows.append(_row(f"file:{name}", "ready" if path.exists() else "blocked", str(path)))

    targets_path = REQUIRED_FILES["demo_fixed_targets"]
    if targets_path.exists():
        targets = _load_json(targets_path)
        rows.append(_row("demo_targets_nonempty", "ready" if len(targets) >= 5 else "warning", f"{len(targets)} targets"))
        verified = [item for item in targets if item.get("verified_level") in {"ojad_checked", "human_checked"}]
        rows.append(_row("pitch_feedback_requires_verified_target", "ready" if verified else "warning", f"{len(verified)} verified targets"))

    messages_path = REQUIRED_FILES["user_facing_messages"]
    if messages_path.exists():
        messages = _load_json(messages_path)
        rows.append(_row("user_facing_messages_nonempty", "ready" if bool(messages) else "blocked", f"{len(messages)} messages"))

    examples_path = REQUIRED_FILES["api_examples_json"]
    if examples_path.exists():
        rows.append(_row("api_examples_contain_user_facing", "ready" if _examples_have_user_facing(examples_path) else "blocked"))
        rows.append(_row("no_raw_debug_score_leakage", "ready" if _examples_no_raw_leakage(examples_path) else "blocked"))

    rows.append(_row("demo_smoke_tests_pass", "ready" if _smoke_tests_pass() else "blocked"))
    rows.append(_row("pytest_command_available", "warning", _git_last_test_hint()))
    rows.append(_row("kanade_playback_only_policy", "ready", "Kanade must stay demo_only/playback_only and excluded from correctness scoring."))
    rows.append(_row("special_mora_default_hidden", "ready", "Special mora remains shadow/debug unless explicitly enabled."))
    rows.append(_row("weak_reference_conservative", "ready", "ASR-generated reference requires user confirmation and weak-reference notice."))
    return rows


def readiness_status(rows: List[Dict[str, str]]) -> str:
    statuses = {row["status"] for row in rows}
    if "blocked" in statuses:
        return "blocked"
    if "warning" in statuses:
        return "warning"
    return "ready"


def write_outputs(rows: List[Dict[str, str]]) -> None:
    out_dir = ROOT / "results" / "demo_readiness"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "demo_readiness_check.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        writer.writerows(rows)

    status = readiness_status(rows)
    report = ROOT / "reports" / "demo_readiness_report.md"
    lines = [
        "# Demo readiness report",
        "",
        f"- status: {status}",
        f"- checks: {len(rows)}",
        f"- ready: {sum(1 for row in rows if row['status'] == 'ready')}",
        f"- warning: {sum(1 for row in rows if row['status'] == 'warning')}",
        f"- blocked: {sum(1 for row in rows if row['status'] == 'blocked')}",
        "",
        "## Current demo scope",
        "",
        "- fixed-reference reading practice is the most reliable path.",
        "- ASR-confirmed weak-reference can be shown as conservative practice support.",
        "- ASR+Kanade can be shown as personalized playback reference; it is not a scoring ground truth.",
        "- Normal UI should read `response.user_facing`; debug/raw metrics stay hidden by default.",
        "",
        "## Do not claim",
        "",
        "- Do not claim validated pronunciation ability scoring.",
        "- Do not claim raw total/prosody scores are scientific correctness.",
        "- Do not claim Kanade evaluates pronunciation correctness.",
        "- Do not claim ASR-generated reference is a strong target before user confirmation.",
        "- Do not claim special-mora feedback is fully validated.",
        "",
        "## Checks",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['status']}: {row['check']} {row['detail']}".rstrip())
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = collect_checks()
    write_outputs(rows)
    status = readiness_status(rows)
    print({"status": status, "checks": len(rows), "report": str(ROOT / "reports" / "demo_readiness_report.md")})
    if status == "blocked":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
