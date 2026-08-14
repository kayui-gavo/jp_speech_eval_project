from __future__ import annotations

import numpy as np

from jp_speech_eval.japanese_phoneme_gop import add_sequence_level_evidence
from jp_speech_eval.japanese_target_evidence import JapaneseTargetEvidence
from jp_speech_eval.phone_gop_preflight import build_phone_gop_preflight_report
from jp_speech_eval.phoneme_gop import compute_phone_gop_evidence


def _target(text: str, phones: list[str], *, distribution: str = "pyopenjtalk-plus") -> JapaneseTargetEvidence:
    return JapaneseTargetEvidence(
        surface_text=text,
        frontend_input=text,
        reading_kana=text,
        reading_source="test",
        phones=phones,
        moras=[],
        fullcontext_labels=[],
        accent_source="test",
        accent_phrases=[],
        verified_target_used=False,
        frontend_distribution=distribution,
        frontend_version="0.4.1.post8" if distribution == "pyopenjtalk-plus" else "0.3-test",
        frontend_ambiguous=False,
        marine_available=False,
        marine_used=False,
        marine_fullcontext_labels=None,
        warnings=[],
    )


class _FakeBackend:
    def vocabulary(self):
        return {"PAD": 0, "a": 1, "b": 2, "pau": 3, "sil": 4}


def _logits() -> np.ndarray:
    values = np.full((8, 5), -6.0, dtype=float)
    values[:, 0] = 0.0
    values[0, 0] = 8.0
    values[1:3, 1] = 9.0
    values[3:5, 0] = 8.0
    values[5:7, 2] = 9.0
    values[7, 0] = 8.0
    return values


def _result(phones: list[str]):
    logits = _logits()
    vocab = {"PAD": 0, "a": 1, "b": 2, "pau": 3, "sil": 4}
    base = compute_phone_gop_evidence(
        logits,
        phones,
        vocab=vocab,
        blank_id=0,
        frame_stride_sec=0.02,
        competitor_token_ids=[1, 2],
    )
    return add_sequence_level_evidence(base, logits, phones, vocab=vocab, blank_id=0)


def _with_contract_flags(result):
    return type(result)(
        **{
            **result.__dict__,
            "summary": {
                **result.summary,
                "high_vowel_allophones_collapsed": True,
                "ctc_support_frames_are_not_physical_phone_boundaries": True,
            },
        }
    )


def test_backend_preflight_can_pass_without_spending_human_time() -> None:
    correct_target = _target("correct", ["a", "b"])
    wrong_target = _target("wrong", ["b", "a"])
    correct = _with_contract_flags(_result(["a", "b"]))
    wrong = _with_contract_flags(_result(["b", "a"]))

    report = build_phone_gop_preflight_report(
        backend=_FakeBackend(),
        correct_target=correct_target,
        wrong_target=wrong_target,
        extra_targets={},
        correct_result=correct,
        wrong_result=wrong,
        gain_results={"gain_0p80": correct, "gain_1p20": correct},
    )

    assert report.backend_preflight_passed is True
    # Passing one bundled-audio backend check is intentionally insufficient to
    # authorize the 38-clip human battery.
    assert report.human_gate_promoted is False
    assert report.human_recording_allowed is False
    statuses = {check.name: check.status for check in report.checks}
    assert statuses["target_frontend_distribution"] == "pass"
    assert statuses["correct_vs_wrong_target_separation"] == "pass"
    assert statuses["gain_stability"] == "pass"
    assert statuses["no_product_score_mapping"] == "pass"


def test_human_gate_requires_explicit_promotion_after_backend_pass() -> None:
    correct_target = _target("correct", ["a", "b"])
    wrong_target = _target("wrong", ["b", "a"])
    correct = _with_contract_flags(_result(["a", "b"]))
    wrong = _with_contract_flags(_result(["b", "a"]))
    report = build_phone_gop_preflight_report(
        backend=_FakeBackend(),
        correct_target=correct_target,
        wrong_target=wrong_target,
        extra_targets={},
        correct_result=correct,
        wrong_result=wrong,
        gain_results={"gain_0p80": correct, "gain_1p20": correct},
        human_gate_promoted=True,
    )
    assert report.backend_preflight_passed is True
    assert report.human_recording_allowed is True


def test_preflight_blocks_base_pyopenjtalk_even_when_acoustics_look_good() -> None:
    correct_target = _target("correct", ["a", "b"], distribution="pyopenjtalk")
    wrong_target = _target("wrong", ["b", "a"], distribution="pyopenjtalk")
    correct = _with_contract_flags(_result(["a", "b"]))
    wrong = _with_contract_flags(_result(["b", "a"]))

    report = build_phone_gop_preflight_report(
        backend=_FakeBackend(),
        correct_target=correct_target,
        wrong_target=wrong_target,
        extra_targets={},
        correct_result=correct,
        wrong_result=wrong,
        gain_results={"gain_0p80": correct, "gain_1p20": correct},
        human_gate_promoted=True,
    )

    assert report.backend_preflight_passed is False
    assert report.human_recording_allowed is False
    frontend = next(check for check in report.checks if check.name == "target_frontend_distribution")
    assert frontend.status == "block"
