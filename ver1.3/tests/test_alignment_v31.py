from types import SimpleNamespace

import numpy as np

from jp_speech_eval.alignment import AlignmentResult, estimate_mora_boundaries_cached_dtw_result


def _cache() -> SimpleNamespace:
    return SimpleNamespace(
        mora_count=3,
        ref_mfcc=np.ones((13, 12), dtype=np.float32),
        ref_y=np.ones(1920, dtype=np.float32),
        meta=SimpleNamespace(ref_mora_boundaries=[(0.0, .04), (.04, .08), (.08, .12)]),
    )


def test_internal_short_audio_equal_fallback_is_explicit():
    result = estimate_mora_boundaries_cached_dtw_result(_cache(), np.ones(100, dtype=np.float32), 16000)
    assert result.used_equal_fallback
    assert not result.available
    assert result.failure_reason == "user_audio_too_short"
    assert result.method.startswith("cached_dtw_mfcc")


def test_insufficient_frames_keep_equal_provenance(monkeypatch):
    monkeypatch.setattr("jp_speech_eval.alignment._feature_pair", lambda *_args: (np.ones((13, 1)), np.ones((13, 1))))
    result = estimate_mora_boundaries_cached_dtw_result(_cache(), np.ones(3000, dtype=np.float32), 16000)
    assert result.used_equal_fallback
    assert result.failure_reason == "insufficient_feature_frames"


def test_dtw_exception_keeps_equal_provenance(monkeypatch):
    monkeypatch.setattr("jp_speech_eval.alignment._feature_pair", lambda *_args: (np.ones((13, 12)), np.ones((13, 14))))
    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic")
    monkeypatch.setattr("jp_speech_eval.alignment.librosa.sequence.dtw", fail)
    result = estimate_mora_boundaries_cached_dtw_result(_cache(), np.ones(3000, dtype=np.float32), 16000)
    assert result.used_equal_fallback
    assert result.failure_reason == "dtw_exception:RuntimeError"


def test_wider_band_runs_only_after_explicit_first_pass_failure(monkeypatch):
    calls = []

    def attempt(_cache, _audio, _sr, *, band_rad, feature_kind):
        calls.append(band_rad)
        if len(calls) == 1:
            return AlignmentResult([], "first", False, 0.0, True, "boundary_health_unstable")
        return AlignmentResult([(0.0, .12), (.12, .25), (.25, .4)], "wide", True, .8, False)

    monkeypatch.setattr("jp_speech_eval.alignment._cached_dtw_attempt", attempt)
    result = estimate_mora_boundaries_cached_dtw_result(_cache(), np.ones(3000, dtype=np.float32), 16000, second_pass_wider_band=True)
    assert calls == [.25, .45]
    assert result.available
    assert result.method == "wide_second_pass"
