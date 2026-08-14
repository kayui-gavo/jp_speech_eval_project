from __future__ import annotations

import unittest

from jp_speech_eval.consumer_dimension_policy import build_consumer_score_dimensions


class ConsumerDimensionPolicyTest(unittest.TestCase):
    def _result(self) -> dict:
        return {
            "moras": ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ", "サ", "イ"],
            "pronunciation_score": 78,
            "fluency_score": 84,
            "prosody_score": 81,
            "details": {
                "pronunciation": {
                    "score_interpretation": "mora_timing_proxy_not_full_segmental_pronunciation",
                },
                "fluency": {
                    "delivery_fluency_score": 88,
                    "rhythm_timing_score": 82,
                },
                "prosody": {
                    "contour_valid_mora_count": 8,
                    "mora_count": 9,
                    "contour_corr": 0.65,
                },
                "reliability": {"f0_coverage": 0.88},
                "alignment": {
                    "available": True,
                    "used_equal_fallback": False,
                },
                "shadow": {},
            },
        }

    def test_legacy_pronunciation_proxy_is_relabelled_as_rhythm(self) -> None:
        dims = build_consumer_score_dimensions(
            self._result(), {"display_score": 80}, mode="reference"
        )
        by_key = {item["key"]: item for item in dims}
        self.assertNotIn("pronunciation", by_key)
        self.assertEqual(by_key["mora_timing"]["label"], "リズム")
        self.assertEqual(by_key["mora_timing"]["value"], 78)
        self.assertIn("not segmental pronunciation", by_key["mora_timing"]["note"])

    def test_fixed_reference_f0_contour_gets_intonation_score(self) -> None:
        dims = build_consumer_score_dimensions(
            self._result(), {"display_score": 80}, mode="reference"
        )
        by_key = {item["key"]: item for item in dims}
        self.assertEqual(by_key["intonation"]["label"], "抑揚")
        self.assertTrue(by_key["intonation"]["available"])
        self.assertEqual(by_key["intonation"]["value"], 81)
        self.assertEqual(
            by_key["intonation"]["construct"],
            "reference_relative_f0_contour_similarity",
        )
        self.assertIn("not lexical pitch-accent", by_key["intonation"]["note"])

    def test_intonation_is_hidden_when_f0_is_insufficient(self) -> None:
        result = self._result()
        result["details"]["reliability"]["f0_coverage"] = 0.2
        dims = build_consumer_score_dimensions(result, {"display_score": 80}, mode="reference")
        by_key = {item["key"]: item for item in dims}
        self.assertFalse(by_key["intonation"]["available"])
        self.assertIsNone(by_key["intonation"]["value"])

    def test_lexical_pitch_accent_is_not_a_top_level_dimension(self) -> None:
        dims = build_consumer_score_dimensions(
            self._result(), {"display_score": 80}, mode="reference"
        )
        self.assertNotIn("pitch_accent", {item["key"] for item in dims})

    def test_pronunciation_appears_only_when_mapped_ssl_exists(self) -> None:
        result = self._result()
        result["details"]["shadow"]["ssl_pronunciation"] = {
            "score_mapped": True,
            "mapped_score": 73,
        }
        dims = build_consumer_score_dimensions(result, {"display_score": 80}, mode="reference")
        by_key = {item["key"]: item for item in dims}
        self.assertEqual(by_key["pronunciation"]["label"], "発音")
        self.assertEqual(by_key["pronunciation"]["value"], 73)


if __name__ == "__main__":
    unittest.main()
