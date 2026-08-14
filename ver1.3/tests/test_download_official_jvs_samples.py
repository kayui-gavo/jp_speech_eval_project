import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_official_jvs_samples.py"
SPEC = importlib.util.spec_from_file_location("download_official_jvs_samples", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DownloadOfficialJvsSamplesTest(unittest.TestCase):
    def test_all_three_official_samples_have_frozen_metadata(self) -> None:
        self.assertEqual(set(MODULE.SAMPLES), {"jvs001", "jvs002", "jvs003"})
        for speaker, spec in MODULE.SAMPLES.items():
            self.assertTrue(str(spec["file_id"]))
            self.assertGreater(int(spec["bytes"]), 100_000)
            digest = str(spec["sha256"])
            self.assertEqual(len(digest), 64)
            self.assertTrue(all(ch in "0123456789abcdef" for ch in digest), speaker)

    def test_verifier_rejects_source_drift(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "frozen sample drift"):
            MODULE.verify_frozen_sample("jvs001", b"RIFF" + b"x" * 2048)

    def test_download_url_uses_frozen_file_id(self) -> None:
        file_id = str(MODULE.SAMPLES["jvs002"]["file_id"])
        url = MODULE._download_url(file_id)
        self.assertIn(file_id, url)
        self.assertTrue(url.startswith("https://drive.usercontent.google.com/"))


if __name__ == "__main__":
    unittest.main()
