from __future__ import annotations

import numpy as np

from jp_speech_eval.japanese_phoneme_gop import (
    SPECIAL_MORA_TOKENS,
    sanitize_canonical_phones,
    segmental_competitor_ids,
)
from jp_speech_eval.segmentation_free_gop import compute_enumerated_fgop_sf_sd_features


def _ctc_friendly_logits() -> tuple[np.ndarray, dict[str, int]]:
    # PAD, a, k, N, cl. Canonical sequence a-N-k receives explicit support.
    vocab = {"PAD": 0, "a": 1, "k": 2, "N": 3, "cl": 4}
    logits = np.full((13, 5), -7.0, dtype=float)
    logits[:, 0] = 0.0
    logits[0, 0] = 9.0
    logits[1:3, 1] = 10.0
    logits[3:5, 0] = 9.0
    logits[5:7, 3] = 10.0
    logits[7:9, 0] = 9.0
    logits[9:11, 2] = 10.0
    logits[11:, 0] = 9.0
    return logits, vocab


def test_special_morae_are_targets_but_not_ordinary_clarity_competitors() -> None:
    vocab = {"PAD": 0, "a": 1, "k": 2, "N": 3, "cl": 4, "pau": 5, "sil": 6}
    competitors = segmental_competitor_ids(vocab, blank_id=0)
    assert set(competitors) == {vocab["a"], vocab["k"]}
    for phone in SPECIAL_MORA_TOKENS:
        assert vocab[phone] not in competitors

    kept, dropped = sanitize_canonical_phones(["a", "N", "cl", "pau", "sil"])
    assert kept == ["a", "N", "cl"]
    assert dropped == ["pau", "sil"]


def test_alignment_free_special_mora_row_is_canonical_plus_deletion_only() -> None:
    logits, vocab = _ctc_friendly_logits()
    result = compute_enumerated_fgop_sf_sd_features(
        logits,
        ["a", "N", "k"],
        vocab=vocab,
        blank_id=0,
        model_id="synthetic",
        revision="test",
    )
    assert result.available is True
    assert result.summary["alignment_free"] is True
    assert result.summary["requires_external_phone_boundaries"] is False
    assert result.summary["special_mora_policy"] == "canonical_plus_deletion_only"
    assert result.summary["individual_lpr_sign_is_pronunciation_error_rule"] is False

    ordinary_a = result.evidence[0]
    mora_n = result.evidence[1]
    ordinary_k = result.evidence[2]
    assert set(ordinary_a.substitution_log_posterior_ratios) == {"a", "k"}
    assert set(ordinary_k.substitution_log_posterior_ratios) == {"a", "k"}
    assert set(mora_n.substitution_log_posterior_ratios) == {"N"}
    assert np.isfinite(mora_n.deletion_log_posterior_ratio)
    assert mora_n.best_noncanonical_alternative_type == "deletion"
    assert mora_n.best_noncanonical_alternative_phone is None


def test_cl_has_same_special_mora_candidate_policy_as_n() -> None:
    logits, vocab = _ctc_friendly_logits()
    # Re-use the synthetic support path as a contract test: target cl is valid
    # even if the acoustics are intentionally N-like. Candidate policy, not
    # correctness, is the assertion here.
    result = compute_enumerated_fgop_sf_sd_features(
        logits,
        ["a", "cl", "k"],
        vocab=vocab,
        blank_id=0,
        model_id="synthetic",
        revision="test",
    )
    assert result.available is True
    cl_row = result.evidence[1]
    assert cl_row.canonical_phone == "cl"
    assert set(cl_row.substitution_log_posterior_ratios) == {"cl"}
    assert cl_row.best_noncanonical_alternative_type == "deletion"
    assert result.score_mapped is False
    assert result.product_calibrated is False
