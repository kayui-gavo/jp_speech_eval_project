from __future__ import annotations

import csv
from pathlib import Path

import pytest

from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "audit" / "phone_target_snapshot_v1.csv"


def _rows():
    with SNAPSHOT.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_frozen_phone_target_snapshot_uses_one_frontend_version() -> None:
    rows = _rows()
    assert len(rows) == 21
    assert {row["frontend_distribution"] for row in rows} == {"pyopenjtalk-plus"}
    assert {row["frontend_version"] for row in rows} == {"0.4.1.post8"}


def test_pyopenjtalk_plus_reproduces_frozen_target_snapshot() -> None:
    pytest.importorskip("pyopenjtalk")
    for row in _rows():
        evidence = build_japanese_target_evidence(row["target_text"])
        assert evidence.frontend_distribution == "pyopenjtalk-plus"
        assert evidence.frontend_ambiguous is False
        assert evidence.frontend_version == row["frontend_version"]
        assert evidence.reading_kana == row["kana"]
        assert evidence.phones == row["phones"].split()
        assert evidence.moras == row["moras"].split()
