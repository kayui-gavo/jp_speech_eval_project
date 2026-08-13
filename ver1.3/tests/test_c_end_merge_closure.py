import json
from pathlib import Path

from jp_speech_eval.api import _fallback_language_eligibility
from jp_speech_eval.asr import AsrTranscript
from jp_speech_eval.config import DEFAULT_SCORING_CONFIG
from jp_speech_eval.prosody_shadows import compute_phrase_intonation_shadow
from jp_speech_eval.special_mora_shadow_v2 import compute_special_mora_v2_shadow


def _evidence(language: str, probability: float | None = 0.9, available: bool = True) -> AsrTranscript:
    return AsrTranscript(available, "faster-whisper", "small", "", language, "ok", probability)


def test_default_json_content_match_does_not_drift_from_default_config():
    config = Path(__file__).resolve().parents[1] / "configs" / "scoring_config.json"
    loaded = json.loads(config.read_text(encoding="utf-8"))["content_match"]
    default = DEFAULT_SCORING_CONFIG["content_match"]
    for field in (
        "enabled", "use_asr", "asr_policy", "asr_provider", "asr_model",
        "cascade_policy", "cascade_base_model", "cascade_rescue_model",
        "cascade_rescue_similarity_floor",
    ):
        assert loaded[field] == default[field]


def test_language_eligibility_accepts_short_kanji_and_katakana_japanese():
    for transcript in (
        "はい", "いいえ", "寿司", "東京", "ラーメン", "コーヒー", "ありがとうございます",
        "東京大学", "新宿駅東口", "日本語能力試験", "人工知能研究", "大学院入学試験",
    ):
        result = _fallback_language_eligibility(transcript, speech_detected=True, f0_coverage=.8, evidence=_evidence("ja"))
        assert result["ok"], transcript
        assert result["eligibility"] == "eligible"
    # Inconclusive language ID retains a conservative Japanese route rather
    # than rejecting a valid kanji-only utterance by script type.
    assert _fallback_language_eligibility("東京", speech_detected=True, f0_coverage=.8, evidence=_evidence("", None, False))["ok"]
    # This backend can call a one-word katakana utterance English at low
    # confidence.  That observation is not allowed to veto Japanese evidence.
    assert _fallback_language_eligibility("ラーメン", speech_detected=False, f0_coverage=.8, evidence=_evidence("en", .41))["ok"]


def test_language_eligibility_rejects_non_japanese_and_nonvoice_controls():
    assert not _fallback_language_eligibility("This is English", speech_detected=True, f0_coverage=.8, evidence=_evidence("en"))["ok"]
    assert not _fallback_language_eligibility("全地化普通クコ", speech_detected=True, f0_coverage=.8, evidence=_evidence("zh"))["ok"]
    # Small faster-whisper can label a short Mandarin control as ja and emit
    # this kana/kanji-looking hallucination.  The lexical check must still
    # keep it out of the broad Japanese scoring route.
    ambiguous = _fallback_language_eligibility("全地化普通クコ", speech_detected=True, f0_coverage=.8, evidence=_evidence("ja"))
    assert not ambiguous["ok"]
    assert ambiguous["reason"] == "joint_asr_hallucination_evidence"
    for name in ("silence", "white noise", "pink noise"):
        assert not _fallback_language_eligibility(name, speech_detected=False, f0_coverage=0.0, evidence=_evidence("ja"))["ok"]


def test_phrase_final_difference_requires_shared_final_transition():
    result = {
        "mora_table": [
            {"f0_hz": None}, {"f0_hz": 100.0}, {"f0_hz": 120.0}, {"f0_hz": None},
        ],
        "details": {"reference_f0_by_mora": [None, None, 100.0, 130.0]},
    }
    phrase = compute_phrase_intonation_shadow(result)
    assert phrase["final_movement_transition_index"] is None
    assert phrase["final_movement_difference"] is None


def test_long_vowel_uses_honest_periodicity_name():
    result = {
        "mora_table": [
            {"mora": "ラ", "start_sec": 0.0, "end_sec": .1},
            {"mora": "ー", "start_sec": .1, "end_sec": .3},
        ],
        "alignment_mode": "cached_dtw",
    }
    payload = compute_special_mora_v2_shadow(result, __import__("numpy").ones(8000) * .1, 16000)
    features = payload["evidence"][0]["features"]
    assert "periodicity_autocorrelation" in features
    assert "voicing_coverage" not in features
