from __future__ import annotations

import numpy as np

from jp_speech_eval.phoneme_gop import (
    HuggingFacePhoneCtcBackend,
    compute_phone_gop_evidence,
    ctc_viterbi_align,
)


def _clean_logits() -> np.ndarray:
    # Vocabulary: blank/PAD=0, a=1, b=2, c=3.
    logits = np.full((8, 4), -3.0, dtype=float)
    logits[:, 0] = 0.0
    logits[0, 0] = 6.0
    logits[1, 1] = 7.0
    logits[2, 1] = 6.5
    logits[3, 0] = 6.0
    logits[4, 2] = 7.0
    logits[5, 2] = 6.5
    logits[6, 0] = 6.0
    logits[7, 0] = 6.0
    return logits


def test_ctc_viterbi_align_supports_known_phone_sequence() -> None:
    logits = _clean_logits()
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    aligned = ctc_viterbi_align(log_probs, [1, 2], blank_id=0)

    assert len(aligned["state_path"]) == logits.shape[0]
    assert len(aligned["phone_frame_indices"]) == 2
    assert aligned["phone_frame_indices"][0]
    assert aligned["phone_frame_indices"][1]


def test_phone_gop_keeps_raw_posterior_logit_and_uncertainty_evidence() -> None:
    result = compute_phone_gop_evidence(
        _clean_logits(),
        ["a", "b"],
        vocab={"PAD": 0, "a": 1, "b": 2, "c": 3},
        blank_id=0,
        frame_stride_sec=0.02,
        backend="synthetic",
        model_id="synthetic-phone-ctc",
    )

    assert result.available is True
    assert result.score_mapped is False
    assert result.product_calibrated is False
    assert len(result.evidence) == 2
    assert all(item.posterior_gop_margin > 0 for item in result.evidence)
    assert all(item.mean_logit_margin > 0 for item in result.evidence)
    assert all(item.mean_entropy >= 0 for item in result.evidence)
    assert result.summary["posterior_gop_margin"]["median"] is not None


def test_local_competitor_can_make_target_phone_the_weakest_without_fake_score() -> None:
    logits = _clean_logits()
    # Make /c/ a strong competitor specifically where canonical /b/ must be
    # supported. Forced CTC still aligns /b/, while the raw GOP/logit evidence
    # records that the acoustics favour a competing phone.
    logits[4:6, 3] = 9.0
    logits[4:6, 2] = 2.0
    result = compute_phone_gop_evidence(
        logits,
        ["a", "b"],
        vocab={"PAD": 0, "a": 1, "b": 2, "c": 3},
        blank_id=0,
        frame_stride_sec=0.02,
    )

    by_phone = {item.canonical_phone: item for item in result.evidence}
    assert by_phone["b"].best_competitor_phone == "c"
    assert by_phone["b"].posterior_gop_margin < by_phone["a"].posterior_gop_margin
    assert result.summary["weakest_phone_by_posterior_margin"] == "b"
    assert result.score_mapped is False


def test_inventory_mismatch_is_explicit_not_silently_remapped() -> None:
    result = compute_phone_gop_evidence(
        _clean_logits(),
        ["a", "cl"],
        vocab={"PAD": 0, "a": 1, "b": 2, "c": 3},
        blank_id=0,
        frame_stride_sec=0.02,
    )
    assert result.available is False
    assert result.summary["reason"] == "canonical_phone_not_in_backend_vocabulary"
    assert result.summary["missing_phones"] == ["cl"]
    assert "phone_inventory_mismatch" in result.warnings


def test_huggingface_backend_is_lazy_and_local_only_by_default() -> None:
    backend = HuggingFacePhoneCtcBackend()
    assert backend.local_files_only is True
    assert backend.model is None
    assert backend.processor is None
