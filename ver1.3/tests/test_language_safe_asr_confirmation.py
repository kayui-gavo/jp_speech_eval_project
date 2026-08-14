from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from jp_speech_eval.asr import AsrTranscript, transcribe_language_aware
from jp_speech_eval.asr_confirmation import _language_eligibility, build_asr_confirmation_prompt


class LanguageSafeAsrConfirmationTest(unittest.TestCase):
    def test_language_aware_faster_whisper_does_not_force_japanese(self) -> None:
        observed = {}

        def fake_try(y, sr, model_name, language="ja"):
            observed["language"] = language
            return AsrTranscript(True, "faster-whisper", model_name, "hello", "en", "ok", 0.98)

        with patch("jp_speech_eval.asr._try_faster_whisper", side_effect=fake_try):
            out = transcribe_language_aware(np.zeros(1600, dtype=np.float32), 16000, provider="faster-whisper")
        self.assertIsNone(observed["language"])
        self.assertEqual(out.text, "hello")
        self.assertEqual(out.language, "en")

    def test_confident_english_is_ineligible(self) -> None:
        eligible, reason = _language_eligibility(
            AsrTranscript(True, "faster-whisper", "small", "I like coffee.", "en", "ok", 0.97)
        )
        self.assertFalse(eligible)
        self.assertEqual(reason, "detected_non_japanese")

    def test_japanese_is_eligible(self) -> None:
        eligible, reason = _language_eligibility(
            AsrTranscript(True, "faster-whisper", "small", "今日はいい天気です。", "ja", "ok", 0.93)
        )
        self.assertTrue(eligible)
        self.assertEqual(reason, "detected_japanese")

    def test_confirmation_never_turns_english_into_japanese_candidate(self) -> None:
        fake_audio = SimpleNamespace(y=np.zeros(3200, dtype=np.float32), sr=16000)
        english = AsrTranscript(True, "faster-whisper", "small", "I like coffee.", "en", "ok", 0.97)
        with patch("jp_speech_eval.asr_confirmation.load_audio", return_value=fake_audio), patch(
            "jp_speech_eval.asr_confirmation.trim_to_speech",
            return_value=(fake_audio.y, SimpleNamespace()),
        ), patch(
            "jp_speech_eval.asr_confirmation.transcribe_language_aware",
            return_value=english,
        ):
            prompt = build_asr_confirmation_prompt("dummy.wav")
        self.assertFalse(prompt.language_eligible)
        self.assertEqual(prompt.mode, "asr_language_reject")
        self.assertEqual(prompt.editable_text, "")
        self.assertEqual(prompt.asr_candidates, [])
        self.assertEqual(prompt.asr_raw["text"], "I like coffee.")
        self.assertEqual(prompt.asr_raw["language"], "en")


if __name__ == "__main__":
    unittest.main()
