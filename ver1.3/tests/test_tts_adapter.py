from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from jp_speech_eval.reference_store import (
    ReferenceStore,
    build_reference_config,
    build_reference_hash,
)
from jp_speech_eval.tts_adapter import TTSAdapter, TTSRequest, canonical_provider_name


class ReferenceStoreTests(unittest.TestCase):
    def test_hash_is_stable_and_changes_with_voice(self) -> None:
        base = build_reference_config(
            text="ラーメンをください",
            provider="local_aivis",
            model=None,
            voice="1",
            speed=1.0,
            style="neutral_teacher",
            prompt=None,
            language="ja-JP",
            sample_rate=16000,
        )
        same = dict(base)
        changed = dict(base)
        changed["voice"] = "2"
        self.assertEqual(build_reference_hash(base), build_reference_hash(same))
        self.assertNotEqual(build_reference_hash(base), build_reference_hash(changed))

    def test_store_roundtrip_marks_cache_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ReferenceStore(tmp)
            cfg = build_reference_config(
                text="雨です",
                provider="local_pyopenjtalk",
                model=None,
                voice=None,
                speed=None,
                style=None,
                prompt=None,
                language="ja-JP",
                sample_rate=16000,
            )
            config_hash = build_reference_hash(cfg)
            saved = store.save(
                text="雨です",
                target_id="ame",
                provider="local_pyopenjtalk",
                model=None,
                voice=None,
                speed=None,
                style=None,
                prompt=None,
                language="ja-JP",
                sample_rate=16000,
                config_hash=config_hash,
                y=np.zeros(320, dtype=np.float32),
            )
            cached = store.get_cached(
                target_id="ame",
                provider="local_pyopenjtalk",
                config_hash=config_hash,
            )
            self.assertIsNotNone(cached)
            assert cached is not None
            self.assertFalse(saved.cache_hit)
            self.assertTrue(cached.cache_hit)
            self.assertTrue(Path(cached.audio_path).exists())


class TTSAdapterTests(unittest.TestCase):
    def test_provider_aliases_are_canonical(self) -> None:
        self.assertEqual(canonical_provider_name("aivis_http"), "local_aivis")
        self.assertEqual(canonical_provider_name("voicevox"), "local_voicevox")
        self.assertEqual(canonical_provider_name("pyopenjtalk"), "local_pyopenjtalk")

    def test_reserved_cloud_provider_is_nonfatal_but_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = TTSAdapter(tmp)
            status = adapter.validate_provider_config("openai")
            self.assertEqual(status["provider"], "openai")
            self.assertFalse(status["available"])
            self.assertFalse(status["implemented"])

    def test_request_hash_uses_canonical_provider(self) -> None:
        req_a = TTSRequest(text="橋です", provider="aivis_http", voice="1")
        req_b = TTSRequest(text="橋です", provider="local_aivis", voice="1")
        self.assertEqual(req_a.config_hash, req_b.config_hash)


if __name__ == "__main__":
    unittest.main()
