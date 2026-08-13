from jp_speech_eval.content_match import ContentMatch, estimate_content_match_base_first_cascade


def _result(*, verified: bool, similarity: float, model: str) -> ContentMatch:
    return ContentMatch(
        status="pass" if verified else "fail", score=similarity, dtw_cost=1.0,
        duration_ratio=1.0, kana_similarity=similarity, transcript="x",
        transcript_kana="x", target_kana="x", asr_provider=model,
        content_verified=verified, acoustic_likely_match=True,
        verification_level="asr_verified" if verified else "mismatch", method="asr", note="ok",
    )


def test_base_first_accepts_verified_base_without_small(monkeypatch):
    calls = []
    def fake(*_args, **kwargs):
        calls.append(kwargs["asr_model"])
        return _result(verified=True, similarity=.95, model=kwargs["asr_model"])
    monkeypatch.setattr("jp_speech_eval.content_match.estimate_content_match", fake)
    result = estimate_content_match_base_first_cascade(None, None, 16000)  # type: ignore[arg-type]
    assert calls == ["base"]
    assert result.method == "base_first_cascade"


def test_base_first_near_mismatch_rescues_only_with_small(monkeypatch):
    calls = []
    def fake(*_args, **kwargs):
        calls.append(kwargs["asr_model"])
        return _result(verified=kwargs["asr_model"] == "small", similarity=.68, model=kwargs["asr_model"])
    monkeypatch.setattr("jp_speech_eval.content_match.estimate_content_match", fake)
    result = estimate_content_match_base_first_cascade(None, None, 16000, rescue_similarity_floor=.60)  # type: ignore[arg-type]
    assert calls == ["base", "small"]
    assert result.content_verified
    assert result.method == "base_first_selective_small_rescue"


def test_base_first_clear_mismatch_does_not_pay_for_small(monkeypatch):
    calls = []
    def fake(*_args, **kwargs):
        calls.append(kwargs["asr_model"])
        return _result(verified=False, similarity=.12, model=kwargs["asr_model"])
    monkeypatch.setattr("jp_speech_eval.content_match.estimate_content_match", fake)
    result = estimate_content_match_base_first_cascade(None, None, 16000, rescue_similarity_floor=.60)  # type: ignore[arg-type]
    assert calls == ["base"]
    assert not result.content_verified
    assert "direct_broad" in result.note
