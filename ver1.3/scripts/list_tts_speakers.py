from __future__ import annotations

import argparse
import json
from urllib import request


def main() -> None:
    parser = argparse.ArgumentParser(description="List speakers/styles exposed by a VOICEVOX-compatible TTS engine.")
    parser.add_argument("--tts-url", default="http://127.0.0.1:10101")
    args = parser.parse_args()

    with request.urlopen(args.tts_url.rstrip("/") + "/speakers", timeout=20) as resp:
        speakers = json.loads(resp.read().decode("utf-8"))

    for speaker in speakers:
        name = speaker.get("name", "")
        speaker_uuid = speaker.get("speaker_uuid", "")
        for style in speaker.get("styles", []):
            print(f"{style.get('id')}\t{name}\t{style.get('name', '')}\t{speaker_uuid}")


if __name__ == "__main__":
    main()
