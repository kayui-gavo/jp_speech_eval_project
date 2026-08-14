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
                    "note": "ok",
                },
                "reliability": {"f0_coverage": 0.88},
                "alignment": {
                    "available": True,
                    "used_equal_fallback": False,
                },
                "shadow": {},
            },
        }

    def _dims(self, result: dict | None = None) -> list[dict]:
        return build_consumer_score_dimensions(
            result or self._result(), {"display_score": 80}, mode="reference"
        )

    def test_four_dimensions_are_stable_and_non_overlapping(self) -> None:
        dims = self._dims()
        self.assertEqual(
            [item["key"] for item in dims],
            ["delivery_fluency", "clarity", "mora_timing", "intonation"],
        )
        self.assertEqual(
            [item["label"] for item in dims],
            ["流暢さ", "明瞭さ", "リズム", "抑揚"],
        )
        self.assertNotIn("韻律", [item["label"] for item in dims])
        self.assertNotIn("pitch_accent", {item["key"] for item in dims})

    def test_legacy_pronunciation_proxy_is_rhythm_not_clarity(self) -> None:
        by_key = {item["key"]: item for item in self._dims()}
        self.assertEqual(by_key["mora_timing"]["value"], 78)
        self.assertIn("mora_timing", by_key["mora_timing"]["construct"])
        self.assertFalse(by_key["clarity"]["available"])
        self.assertIsNone(by_key["clarity"]["value"])
        self.assertNotEqual(by_key["clarity"]["source_field"], "pronunciation_score")
        self.assertIn("not recording quality", by_key["clarity"]["note"])

    def test_mapped_ssl_can_fill_clarity_without_reusing_timing_proxy(self) -> None:
        result = self._result()
        result["details"]["shadow"]["ssl_pronunciation"] = {
            "score_mapped": True,
            "mapped_score": 73,
        }
        by_key = {item["key"]: item for item in self._dims(result)}
        self.assertTrue(by_key["clarity"]["available"])
        self.assertEqual(by_key["clarity"]["value"], 73)
        self.assertIn("ssl", by_key["clarity"]["construct"])

    def test_fixed_reference_f0_contour_gets_intonation_score(self) -> None:
        by_key = {item["key"]: item for item in self._dims()}
        self.assertTrue(by_key["intonation"]["available"])
        self.assertEqual(by_key["intonation"]["value"], 81)
        self.assertEqual(
            by_key["intonation"]["construct"],
            "reference_relative_normalized_f0_contour_similarity",
        )
        self.assertIn("not strict lexical pitch-accent", by_key["intonation"]["note"])

    def test_intonation_survives_moderate_f0_coverage(self) -> None:
        result = self._result()
        result["details"]["reliability"]["f0_coverage"] = 0.42
        result["details"]["prosody"]["contour_valid_mora_count"] = 4
        by_key = {item["key"]: item for item in self._dims(result)}
        self.assertTrue(by_key["intonation"]["available"])
        self.assertEqual(by_key["intonation"]["confidence"], "medium")

    def test_true_f0_failure_is_not_turned_into_neutral_score(self) -> None:
        result = self._result()
        result["details"]["reliability"]["f0_coverage"] = 0.2
        result["details"]["prosody"]["contour_valid_mora_count"] = 2
        result["details"]["prosody"]["note"] = "insufficient_valid_mora_f0"
        by_key = {item["key"]: item for item in self._dims(result)}
        self.assertFalse(by_key["intonation"]["available"])
        self.assertIsNone(by_key["intonation"]["value"])


if __name__ == "__main__":
    unittest.main()
