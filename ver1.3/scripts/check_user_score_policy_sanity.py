from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.user_score_policy import apply_user_score_policy


def _base(**overrides):
    result = {
        "total_score": 90,
        "pronunciation_score": 88,
        "prosody_score": 86,
        "fluency_score": 87,
        "moras": ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ", "サ", "イ"],
        "alignment_mode": "cached_dtw",
        "details": {
            "mode": "reference_based",
            "reliability": {"level": "high", "overall": 0.95, "alignment": 0.9, "f0_coverage": 0.9},
            "recording_quality": {"score": 0.9},
            "content_match": {"status": "pass"},
            "alignment": {"mode": "cached_dtw"},
            "fluency": {"rhythm_timing_score": 86, "delivery_fluency_score": 87},
        },
    }
    for key, value in overrides.items():
        if key == "details":
            result["details"].update(value)
        else:
            result[key] = value
    return result


def _check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"ok - {name}")


def main() -> None:
    good = apply_user_score_policy(_base())
    _check("good fixed reference can show high practice reference score", good["display_score"] >= 80)

    bad = apply_user_score_policy(_base(pronunciation_score=55, prosody_score=95, fluency_score=95))
    _check("bad pronunciation under 60 is capped", bad["display_score"] <= 68)
    _check("bad pronunciation message is not hidden", bad["main_message_key"] == "clear_recording_but_pronunciation_needs_practice")

    fallback = apply_user_score_policy(_base(alignment_mode="cached_dtw_fallback_equal"))
    _check("fallback display score capped", fallback["display_score"] <= 70)
    _check("fallback pronunciation clarity capped", fallback["pronunciation_clarity_score"] <= 65)

    marginal = apply_user_score_policy(_base(details={"content_match": {"status": "marginal"}}))
    _check("marginal content capped", marginal["display_score"] <= 75)

    failed = apply_user_score_policy(_base(details={"content_match": {"status": "fail"}}))
    _check("content failed hides display score", failed["display_score"] is None)
    _check("content failed hides pronunciation clarity", failed["pronunciation_clarity_score"] is None)

    weak = apply_user_score_policy(_base(details={"weak_reference": True}), mode="asr_confirmed_weak_reference")
    _check("weak reference capped", weak["display_score"] <= 80)
    _check("weak reference confidence not high", weak["confidence_label"] != "high")

    short = apply_user_score_policy(_base(moras=["ア", "メ", "ガ"]))
    _check("short utterance capped", short["display_score"] <= 80)
    _check("short utterance disables detail feedback", short["detail_feedback_allowed"] is False)


if __name__ == "__main__":
    main()
