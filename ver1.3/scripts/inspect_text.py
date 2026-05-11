from __future__ import annotations

import argparse

from jp_speech_eval.text_frontend import build_text_info, run_frontend


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Japanese text frontend")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    info = build_text_info(args.text)
    print("Text          :", info.text)
    print("Kana          :", info.kana)
    print("Mora          :", "・".join(info.moras))
    print("Target pitch  :", " ".join(info.target_pitch))
    print("Pitch source  :", info.pitch_target_source)
    print("Is question   :", info.is_question)
    print("\nAccent phrases:")
    for phrase in info.accent_phrases:
        print(phrase)
    print("\nOpenJTalk frontend raw output:")
    for item in run_frontend(args.text):
        print(item)


if __name__ == "__main__":
    main()
