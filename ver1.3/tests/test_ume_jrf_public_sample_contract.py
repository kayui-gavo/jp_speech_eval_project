from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from jp_speech_eval.japanese_target_evidence import build_japanese_target_evidence
from jp_speech_eval.japanese_phoneme_gop import sanitize_canonical_phones


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "download_official_ume_jrf_samples",
    ROOT / "scripts" / "download_official_ume_jrf_samples.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _phones(text: str) -> list[str]:
    evidence = build_japanese_target_evidence(text)
    phones, _dropped = sanitize_canonical_phones(evidence.phones)
    return list(phones)


class UmeJrfPublicSampleContractTest(unittest.TestCase):
    def test_official_public_sample_ids_and_categories_are_frozen(self) -> None:
        rows = {row["sample_id"]: row for row in MODULE.SAMPLES}
        self.assertEqual(set(rows), {"A1_001", "B1_001", "C1_001", "D1_001", "D1_002"})
        self.assertEqual(rows["A1_001"]["category"], "phonetically_balanced_sentence")
        self.assertEqual(rows["B1_001"]["category"], "prosody_sentence")
        self.assertEqual(rows["C1_001"]["category"], "difficult_sentence")
        self.assertEqual(rows["D1_001"]["category"], "minimal_pair_word")

    def test_source_annotated_hito_reading_is_made_explicit_for_g2p(self) -> None:
        rows = {row["sample_id"]: row for row in MODULE.SAMPLES}
        a1 = rows["A1_001"]
        self.assertIn("人 (ひと)いきれ", a1["source_display_text"])
        self.assertIn("ひといきれ", a1["target_text"])
        self.assertNotIn("人いきれ", a1["target_text"])
        self.assertEqual(a1["reading_disambiguation"], "人(ひと) -> ひと")

    def test_minimal_pair_is_n_presence_absence_contrast(self) -> None:
        rows = {row["sample_id"]: row for row in MODULE.SAMPLES}
        self.assertEqual(rows["D1_001"]["target_text"], "じぶつ")
        self.assertEqual(rows["D1_001"]["minimal_pair_partner"], "じんぶつ")
        self.assertEqual(rows["D1_002"]["target_text"], "じんぶつ")
        self.assertEqual(rows["D1_002"]["minimal_pair_partner"], "じぶつ")

        short = _phones("じぶつ")
        nasal = _phones("じんぶつ")
        self.assertEqual(len(nasal), len(short) + 1)
        candidates = [i for i in range(len(nasal)) if nasal[:i] + nasal[i + 1 :] == short]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(nasal[candidates[0]], "N")

    def test_urls_are_official_nii_src_paths(self) -> None:
        self.assertEqual(MODULE.OFFICIAL_PAGE, "https://research.nii.ac.jp/src/en/UME-JRF.html")
        self.assertEqual(MODULE.BASE, "https://research.nii.ac.jp/src/sample/UME-JRF")
        for row in MODULE.SAMPLES:
            self.assertTrue(row["filename"].endswith(".wav"))


if __name__ == "__main__":
    unittest.main()
