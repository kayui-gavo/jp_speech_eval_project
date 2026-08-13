from types import SimpleNamespace

import numpy as np

from jp_speech_eval import content_match
from jp_speech_eval.asr import AsrTranscript


def _cache():
    return SimpleNamespace(
        meta=SimpleNamespace(ref_duration_sec=1.0, kana="テスト"),
        ref_mfcc=np.ones((2, 3), dtype=np.float32),
    )


def test_acoustic_only_pass_is_never_content_verified(monkeypatch):
    monkeypatch.setattr(content_match, "_user_mfcc", lambda *_args, **_kwargs: np.ones((2, 3), dtype=np.float32))
    monkeypatch.setattr(content_match.librosa.sequence, "dtw", lambda **_kwargs: (np.array([[3.6]]), [(0, 0)]))
    result = content_match.estimate_content_match(_cache(), np.ones(16000), 16000, use_asr=False)
    assert result.acoustic_likely_match is True
    assert result.content_verified is False
    assert result.verification_level == "acoustic_likely"


def test_asr_unavailable_does_not_restore_acoustic_verification(monkeypatch):
    monkeypatch.setattr(content_match, "transcribe_japanese", lambda *_args, **_kwargs: AsrTranscript(False, "fake", "tiny", "", "ja", "offline"))
    result = content_match._asr_gate(_cache(), np.ones(16000), 16000, "pass", 0.9, 3.5, 1.0, "tiny", "fake")
    assert result.status == "uncertain"
    assert result.content_verified is False
    assert result.verification_level == "acoustic_likely"


def test_asr_verified_requires_text_agreement(monkeypatch):
    monkeypatch.setattr(content_match, "transcribe_japanese", lambda *_args, **_kwargs: AsrTranscript(True, "fake", "tiny", "テスト", "ja", "ok"))
    monkeypatch.setattr(content_match, "text_to_kana", lambda text: text)
    result = content_match._asr_gate(_cache(), np.ones(16000), 16000, "pass", 0.9, 3.5, 1.0, "tiny", "fake")
    assert result.content_verified is True
    assert result.verification_level == "asr_verified"
