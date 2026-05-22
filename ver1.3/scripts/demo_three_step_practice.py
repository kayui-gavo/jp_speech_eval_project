from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jp_speech_eval.app_core.personalized_scorer import compare_to_profile
from jp_speech_eval.app_core.practice_modes import (
    compute_reference_dependency_gap,
    step_label,
    step_user_instruction,
)
from jp_speech_eval.app_core.progress_tracker import (
    append_progress_record,
    latest_record,
    load_progress_records,
    record_from_evaluation,
)
from jp_speech_eval.app_core.user_profile import load_user_profile
from jp_speech_eval.evaluator import evaluate_utterance


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one step of the minimal three-step practice MVP.")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--item-id", default="ramen_kudasai")
    parser.add_argument("--step", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--target-text", default="ラーメンをください")
    parser.add_argument("--cache", default=None, help="Optional prepared sentence cache.")
    parser.add_argument("--profile", default=None, help="Optional user voice profile JSON.")
    parser.add_argument("--progress-jsonl", default="outputs/progress/practice_progress.jsonl")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    records = load_progress_records(args.progress_jsonl, user_id=args.user_id)
    previous = latest_record(records, item_id=args.item_id, step=args.step)
    profile = load_user_profile(args.profile) if args.profile else None

    result = evaluate_utterance(
        text=None if args.cache else args.target_text,
        wav_path=args.wav,
        cache_path=args.cache,
        alignment_mode="cached_dtw" if args.cache else "equal",
        scoring_config_path=args.config,
        sample_rate=args.sr,
        use_content_match=bool(args.cache),
    )
    personalized = compare_to_profile(result, profile=profile, previous_record=previous)
    record = record_from_evaluation(
        user_id=args.user_id,
        item_id=args.item_id,
        step=args.step,
        audio_path=args.wav,
        result=result,
        extra_feedback=personalized.feedback,
    )
    append_progress_record(args.progress_jsonl, record)

    updated_records = load_progress_records(args.progress_jsonl, user_id=args.user_id, item_id=args.item_id)
    dependency = compute_reference_dependency_gap(updated_records, item_id=args.item_id)

    output = {
        "step": args.step,
        "step_label": step_label(args.step),
        "instruction": step_user_instruction(args.step),
        "scores": record.scores,
        "feedback": record.feedback[:6],
        "personalized": personalized.to_dict(),
        "reference_dependency": dependency.to_dict(),
        "saved_to": args.progress_jsonl,
        "note": "User-facing feedback is intentionally short; raw evaluator details remain in debug fields.",
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
