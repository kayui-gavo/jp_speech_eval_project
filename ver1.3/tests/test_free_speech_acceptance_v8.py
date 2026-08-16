from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.run_free_speech_acceptance_v8 import run_acceptance
from scripts.validate_free_speech_sample_manifest import REQUIRED_COLUMNS


def _row(sample_id: str, expected_language: str, *, speaker_group: str = "learner"):
    return {
        "sample_id": sample_id,
        "audio_path": f"audio/{sample_id}.wav",
        "speaker_id": f"spk_{sample_id}",
        "speaker_group": speaker_group,
        "l1": "zh" if speaker_group == "learner" else "control" if speaker_group == "negative_control" else "ja",
        "task_mode": "controlled_dialogue",
        "prompt_id": f"prompt_{sample_id}",
        "split": "held",
        "expected_language": expected_language,
        "channel_condition": "clean",
        "channel_pair_id": "",
        "source_recording_id": f"source_{sample_id}",
        "context_type": "prompt",
        "context_id": f"ctx_{sample_id}",
        "context_text": "最近の出来事について話してください。",
        "context_audio_path": "",
        "source_note": "v8 acceptance test",
    }


def _write_manifest(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def test_acceptance_runs_product_batch_analysis_and_routing_summary(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    out_dir = tmp_path / "out"
    _write_manifest(
        manifest,
        [
            _row("ja_ok", "ja"),
            _row("ja_no_score", "ja"),
            _row("en_control", "en", speaker_group="negative_control"),
        ],
    )

    def fake_evaluator(wav_path, transcript=None, **kwargs):
        sample_id = Path(wav_path).stem
        assert transcript is None
        return {
            "details": {
                "mode": "transcript_assisted_light",
                "language_gate": {"eligible": sample_id != "en_control"},
                "recording_quality": {"score": 0.9},
                "shadow": {},
            },
            "fluency_score": 80,
        }

    def fake_policy(raw, mode="transcript_assisted_light"):
        eligible = bool(raw["details"]["language_gate"]["eligible"])
        # Make one Japanese row a deliberate no-score acceptance case by
        # observing the call order through a tiny mutable counter.
        fake_policy.calls += 1
        score_available = eligible and fake_policy.calls == 1
        return {
            "score_available": score_available,
            "display_score": 74 if score_available else None,
            "component_scores": {
                "clarity": {"value": 70, "evidence_state": "neutral_prior"},
                "mora_timing": {"value": 70, "evidence_state": "neutral_prior"},
                "delivery_fluency": {"value": 82 if eligible else None, "evidence_state": "broad_proxy" if eligible else "unavailable"},
                "intonation": {"value": 70, "evidence_state": "neutral_prior"},
            },
        }

    fake_policy.calls = 0
    report = run_acceptance(
        manifest,
        out_dir,
        audio_root=tmp_path,
        evaluator=fake_evaluator,
        user_policy=fake_policy,
        resume=False,
    )

    assert report["schema"] == "free_speech_acceptance_v8"
    assert report["batch"]["written_ok"] == 3
    assert report["batch"]["failed"] == 0
    assert report["frozen_product_conditions"]["scoring_used_gold_transcript"] is False
    assert report["frozen_product_conditions"]["threshold_tuning_allowed_on_this_run"] is False
    assert report["decision"] == "collect_evidence_only_no_score_promotion"

    routing = report["routing"]
    assert routing["groups"]["expected_japanese"]["n"] == 2
    assert routing["groups"]["expected_non_japanese_speech"]["n"] == 1
    assert routing["valid_japanese_false_no_score_count"] == 1
    assert routing["non_japanese_speech_normal_score_count"] == 0
    assert routing["nonspeech_control_normal_score_count"] == 0

    assert (out_dir / "free_speech_validation_v8.jsonl").exists()
    assert (out_dir / "partial_evidence_analysis_v8.json").exists()
    assert (out_dir / "acceptance_summary_v8.json").exists()
    saved = json.loads((out_dir / "acceptance_summary_v8.json").read_text(encoding="utf-8"))
    assert saved["schema"] == "free_speech_acceptance_v8"
    assert saved["partial_evidence_analysis"]["decision"] == "telemetry_only_no_product_promotion"


def test_acceptance_keeps_nonspeech_controls_separate_and_candidate_unavailable(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    out_dir = tmp_path / "out"
    _write_manifest(manifest, [_row("noise_control", "non_speech", speaker_group="negative_control")])

    def fake_evaluator(wav_path, transcript=None, **kwargs):
        return {"details": {"mode": "transcript_assisted_light", "language_gate": {"eligible": False}, "recording_quality": {}, "shadow": {}}, "fluency_score": None}

    def fake_policy(raw, mode="transcript_assisted_light"):
        return {
            "score_available": False,
            "display_score": None,
            "component_scores": {
                "clarity": {"value": 70, "evidence_state": "neutral_prior"},
                "mora_timing": {"value": 70, "evidence_state": "neutral_prior"},
                "delivery_fluency": {"value": None, "evidence_state": "unavailable"},
                "intonation": {"value": 70, "evidence_state": "neutral_prior"},
            },
        }

    report = run_acceptance(
        manifest,
        out_dir,
        audio_root=tmp_path,
        evaluator=fake_evaluator,
        user_policy=fake_policy,
        resume=False,
    )
    analysis = report["partial_evidence_analysis"]
    assert analysis["candidate_available_rate"] == 0.0
    assert analysis["paired_score_count"] == 0
    routing = report["routing"]
    assert routing["groups"]["expected_nonspeech_control"]["product_score_available"] == 0
    assert routing["nonspeech_control_normal_score_count"] == 0
    assert "expected_non_japanese_speech" not in routing["groups"]
