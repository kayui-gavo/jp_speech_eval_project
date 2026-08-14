import importlib.util
import io
from pathlib import Path
import struct
import unittest
import wave


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_official_jvs_samples.py"
SPEC = importlib.util.spec_from_file_location("download_official_jvs_samples", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _pcm_wav(duration_sec: float, *, sr: int = 24000, channels: int = 1) -> bytes:
    frames = int(round(duration_sec * sr))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(struct.pack("<h", 0) * frames * channels)
    return buffer.getvalue()


class DownloadOfficialJvsSamplesTest(unittest.TestCase):
    def test_all_three_official_samples_have_source_and_duration_metadata(self) -> None:
        self.assertEqual(set(MODULE.SAMPLES), {"jvs001", "jvs002", "jvs003"})
        for speaker, spec in MODULE.SAMPLES.items():
            self.assertTrue(str(spec["file_id"]))
            self.assertGreater(float(spec["expected_duration_sec"]), 5.0)
            historical = dict(spec["historical_raw_variants"])
            self.assertTrue(historical, speaker)
            for digest, byte_count in historical.items():
                self.assertEqual(len(str(digest)), 64)
                self.assertGreater(int(byte_count), 100_000)

    def test_semantic_verifier_accepts_same_duration_with_new_container_hash(self) -> None:
        expected = float(MODULE.SAMPLES["jvs001"]["expected_duration_sec"])
        data = _pcm_wav(expected, sr=24000)
        result = MODULE.verify_official_sample("jvs001", data)
        self.assertTrue(result["semantic_duration_verified"])
        self.assertFalse(result["raw_transport_hash_is_acoustic_identity"])
        self.assertEqual(result["sample_rate"], 24000)
        self.assertEqual(result["channels"], 1)

    def test_semantic_verifier_rejects_duration_drift(self) -> None:
        expected = float(MODULE.SAMPLES["jvs001"]["expected_duration_sec"])
        data = _pcm_wav(expected + 0.5, sr=24000)
        with self.assertRaisesRegex(RuntimeError, "acoustic source drift"):
            MODULE.verify_official_sample("jvs001", data)

    def test_semantic_verifier_rejects_stereo(self) -> None:
        expected = float(MODULE.SAMPLES["jvs002"]["expected_duration_sec"])
        data = _pcm_wav(expected, sr=24000, channels=2)
        with self.assertRaisesRegex(RuntimeError, "must be mono"):
            MODULE.verify_official_sample("jvs002", data)

    def test_download_url_uses_reviewed_file_id(self) -> None:
        file_id = str(MODULE.SAMPLES["jvs002"]["file_id"])
        url = MODULE._download_url(file_id)
        self.assertIn(file_id, url)
        self.assertTrue(url.startswith("https://drive.usercontent.google.com/"))


if __name__ == "__main__":
    unittest.main()
