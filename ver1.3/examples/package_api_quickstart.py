from __future__ import annotations

from pathlib import Path

from jp_speech_eval import EvaluationRequest, SpeechEvalConfig, SpeechEvaluationClient


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    client = SpeechEvaluationClient(
        SpeechEvalConfig(
            cache_path=str(root / "cache" / "ramen_kudasai"),
            tts_backend="pyopenjtalk",
        )
    )
    response = client.evaluate(
        EvaluationRequest(
            audio_path=str(root / "data" / "ramen.wav"),
            mode="reference",
            target_text="ラーメンをください",
        )
    )
    payload = response.to_dict()
    if not payload["ok"]:
        raise SystemExit(payload["error"])

    user_facing = payload["user_facing"]
    print("status:", user_facing["status"])
    print("practice_score:", user_facing["practice_score"])
    print("summary:", user_facing["summary_text"])
    print("suggestion:", user_facing["primary_suggestion_text"])


if __name__ == "__main__":
    main()
