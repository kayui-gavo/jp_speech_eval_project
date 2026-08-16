from __future__ import annotations

import csv
from pathlib import Path

from scripts.assess_free_speech_machine_coverage_v10 import assess


CANDIDATES = [
    "clarity.shadow_candidate_score",
    "rhythm.shadow_candidate_score",
    "fluency.current_product_proxy",
    "intonation.shadow_candidate_score",
]


def _write_evidence(path: Path, *, learner_long: bool = True, intonation_long_available: bool = True) -> None:
    rows: list[dict] = []
    for i in range(40):
        learner = i < 30
        group = "learner" if learner else "native"
        if learner:
            duration = 2.0 if (i % 2 == 0 or not learner_long) else 10.0
        else:
            duration = 10.0
        task = "spontaneous" if i % 2 == 0 else "controlled_dialogue"
        for candidate in CANDIDATES:
            available = not (
                candidate == "intonation.shadow_candidate_score"
                and learner
                and duration >= 8.0
                and not intonation_long_available
            )
            rows.append(
                {
                    "sample_id": f"s{i:02d}",
                    "speaker_id": f"spk{i % 12:02d}",
                    "speaker_group": group,
                    "task": task,
                    "subset": "held",
                    "expected_language": "ja",
                    "speech_duration_sec": duration,
                    "candidate": candidate,
                    "available": "true" if available else "false",
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_structure_can_be_ready_while_one_candidate_lacks_usable_long_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.csv"
    _write_evidence(evidence, learner_long=True, intonation_long_available=False)
    report = assess(evidence, "data/research_eval/free_speech_v10_consumer_promotion_protocol.json")

    assert report["collection_structure_ready_for_final_listener_pack"] is True
    assert report["held_learner_slice_counts"]["short"] == 15
    assert report["held_learner_slice_counts"]["long"] == 15
    assert report["candidate_coverage"]["clarity.shadow_candidate_score"]["coverage_ready_for_v10_human_join"] is True
    assert report["candidate_coverage"]["intonation.shadow_candidate_score"]["coverage_ready_for_v10_human_join"] is False
    assert report["human_ratings_used"] is False
    assert report["product_score_changed"] is False


def test_native_long_audio_cannot_rescue_collection_with_no_learner_long_audio(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.csv"
    _write_evidence(evidence, learner_long=False)
    report = assess(evidence, "data/research_eval/free_speech_v10_consumer_promotion_protocol.json")

    assert report["held_learner_slice_counts"]["long"] == 0
    assert report["collection_structure_states"]["long"] == "insufficient"
    assert report["collection_structure_ready_for_final_listener_pack"] is False
    assert "long" in report["missing_or_undercovered_structure"]
