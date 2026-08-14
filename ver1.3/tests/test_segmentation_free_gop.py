from __future__ import annotations

import numpy as np

from jp_speech_eval.segmentation_free_gop import compute_enumerated_fgop_sf_sd_features


def _logits_for_ab(*, make_b_sound_like_c: bool = False) -> np.ndarray:
    # PAD=0, a=1, b=2, c=3. CTC-friendly a [blank] b path.
    logits = np.full((9, 4), -7.0, dtype=float)
    logits[:, 0] = 0.0
    logits[0, 0] = 9.0
    logits[1:3, 1] = 10.0
    logits[3:5, 0] = 9.0
    logits[5:7, 2] = 10.0
    logits[7:, 0] = 9.0
    if make_b_sound_like_c:
        logits[5:7, 2] = 1.0
        logits[5:7, 3] = 10.0
    return logits


def test_fgop_sf_sd_feature_vector_contains_lpp_substitutions_and_deletion() -> None:
    vocab = {"PAD": 0, "a": 1, "b": 2, "c": 3}
    result = compute_enumerated_fgop_sf_sd_features(
        _logits_for_ab(),
        ["a", "b"],
        vocab=vocab,
        blank_id=0,
    )
    assert result.available is True
    assert result.score_mapped is False
    assert result.product_calibrated is False
    assert result.summary["includes_substitution"] is True
    assert result.summary["includes_deletion"] is True
    assert result.summary["includes_insertion"] is False
    assert result.summary["occ_activation_normalization_implemented"] is False
    assert len(result.evidence) == 2

    for row in result.evidence:
        # The SD feature set includes substituting the canonical phone with
        # itself, so that LPR dimension must be exactly zero up to FP error.
        assert abs(row.substitution_log_posterior_ratios[row.canonical_phone]) < 1e-8
        assert set(row.substitution_log_posterior_ratios) == {"a", "b", "c"}
        assert np.isfinite(row.deletion_log_posterior_ratio)
        # Denominator includes canonical + alternatives, so log posterior ratio
        # to the full SD set cannot be positive.
        assert row.gop_sf_sd <= 1e-8


def test_alignment_free_feature_identifies_strong_substitution_alternative() -> None:
    vocab = {"PAD": 0, "a": 1, "b": 2, "c": 3}
    clean = compute_enumerated_fgop_sf_sd_features(
        _logits_for_ab(), ["a", "b"], vocab=vocab, blank_id=0
    )
    error = compute_enumerated_fgop_sf_sd_features(
        _logits_for_ab(make_b_sound_like_c=True), ["a", "b"], vocab=vocab, blank_id=0
    )

    clean_b = clean.evidence[1]
    error_b = error.evidence[1]
    assert error_b.best_noncanonical_alternative_type == "substitution"
    assert error_b.best_noncanonical_alternative_phone == "c"
    assert error_b.best_noncanonical_log_posterior_ratio < 0
    assert error_b.substitution_log_posterior_ratios["c"] < clean_b.substitution_log_posterior_ratios["c"]
    assert "one_or_more_noncanonical_sd_alternatives_outscore_canonical_sequence" in error.warnings


def test_deletion_is_an_explicit_feature_not_inferred_from_forced_frames() -> None:
    vocab = {"PAD": 0, "a": 1, "b": 2, "c": 3}
    # Audio contains a -> c while canonical is a -> b -> c. The middle /b/
    # deletion alternative should be more probable than the canonical sequence.
    logits = np.full((9, 4), -7.0, dtype=float)
    logits[:, 0] = 0.0
    logits[0, 0] = 9.0
    logits[1:3, 1] = 10.0
    logits[3:5, 0] = 9.0
    logits[5:7, 3] = 10.0
    logits[7:, 0] = 9.0

    result = compute_enumerated_fgop_sf_sd_features(
        logits, ["a", "b", "c"], vocab=vocab, blank_id=0
    )
    middle = result.evidence[1]
    assert middle.canonical_phone == "b"
    assert middle.deletion_log_posterior_ratio < 0
    assert middle.best_noncanonical_alternative_type in {"deletion", "substitution"}


def test_repeated_phone_positions_remain_distinct_feature_rows() -> None:
    vocab = {"PAD": 0, "a": 1, "b": 2}
    logits = np.full((9, 3), -7.0, dtype=float)
    logits[:, 0] = 0.0
    logits[0, 0] = 9.0
    logits[1:3, 1] = 10.0
    logits[3:5, 0] = 9.0
    logits[5:7, 1] = 10.0
    logits[7:, 0] = 9.0
    result = compute_enumerated_fgop_sf_sd_features(
        logits, ["a", "a"], vocab=vocab, blank_id=0
    )
    assert result.available is True
    assert [row.phone_index for row in result.evidence] == [0, 1]
    assert [row.canonical_phone for row in result.evidence] == ["a", "a"]
    # Deleting either identical phone yields the same alternate transcription;
    # keep that ambiguity explicit rather than pretending the metric localizes
    # which long/repeated mora was omitted.
    assert np.isclose(
        result.evidence[0].deletion_log_posterior,
        result.evidence[1].deletion_log_posterior,
    )
