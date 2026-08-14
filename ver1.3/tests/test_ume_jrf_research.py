from pathlib import Path
import tempfile
import unittest

from jp_speech_eval.ume_jrf_research import (
    D_RATED_WORDS,
    build_ume_jrf_criterion_label,
    probe_ume_jrf_layout,
    research_guard_metadata,
)


class UmeJrfResearchTest(unittest.TestCase):
    def test_research_guard_blocks_product_ingestion(self) -> None:
        guard = research_guard_metadata()
        self.assertTrue(guard["research_only_license"])
        self.assertFalse(guard["commercial_product_use_allowed"])
        self.assertFalse(guard["product_runtime_ingestion_allowed"])
        self.assertFalse(guard["product_model_training_allowed_without_separate_permission"])
        self.assertFalse(guard["normalize_raw_expert_labels_to_100_on_import"])
        self.assertIn("pseudonymous", guard["speaker_identifier_policy"])
        self.assertEqual(len(D_RATED_WORDS), 10)
        self.assertIn("酸っぱい", D_RATED_WORDS)

    def test_set_specific_label_scales_and_constructs_remain_separate(self) -> None:
        broad = build_ume_jrf_criterion_label(
            set_id="A", speaker_id="learner001", item_id="A1_001", rater_id="r1", raw_label=5
        )
        phone_binary = build_ume_jrf_criterion_label(
            set_id="B", speaker_id="learner001", item_id="B1_001", rater_id="r2", raw_label=1
        )
        prosody = build_ume_jrf_criterion_label(
            set_id="C", speaker_id="learner001", item_id="C_001", rater_id="r3", raw_label=4
        )
        phone_ordinal = build_ume_jrf_criterion_label(
            set_id="D",
            speaker_id="learner001",
            item_id="D_suppai",
            rater_id="r4",
            raw_label=3,
            target_description="促音になっているか",
        )
        self.assertEqual(broad.scale, "ordinal_1_5")
        self.assertEqual(phone_binary.scale, "binary_correctness")
        self.assertEqual(prosody.scale, "ordinal_1_5")
        self.assertEqual(phone_ordinal.scale, "ordinal_1_5")
        self.assertNotEqual(broad.construct, phone_binary.construct)
        self.assertNotEqual(prosody.construct, phone_ordinal.construct)
        for label in (broad, phone_binary, prosody, phone_ordinal):
            self.assertEqual(label.speaker_id, "learner001")
            self.assertIsNone(label.normalized_100)
            self.assertTrue(label.research_only_license)
            self.assertFalse(label.commercial_product_use_allowed)
            self.assertFalse(label.product_score_mapped)

    def test_invalid_scale_values_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            build_ume_jrf_criterion_label(
                set_id="B", speaker_id="learner001", item_id="B1", rater_id="r1", raw_label=4
            )
        with self.assertRaises(ValueError):
            build_ume_jrf_criterion_label(
                set_id="D", speaker_id="learner001", item_id="D1", rater_id="r1", raw_label=0
            )

    def test_missing_speaker_id_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "speaker_id"):
            build_ume_jrf_criterion_label(
                set_id="D", speaker_id="", item_id="D1", rater_id="r1", raw_label=3
            )

    def test_missing_corpus_probe_does_not_download_or_enable_parser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "UME-JRF-not-present"
            probe = probe_ume_jrf_layout(missing)
        self.assertFalse(probe.available)
        self.assertEqual(probe.grading_schema_status, "corpus_root_missing")
        self.assertFalse(probe.automatic_label_parsing_allowed)
        self.assertTrue(probe.research_only_license)
        self.assertFalse(probe.commercial_product_use_allowed)

    def test_present_label_description_still_requires_schema_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            label_doc = root / "Vol1" / "doc" / "FJlabel" / "description.txt"
            content_doc = root / "Vol1" / "doc" / "FJcontent" / "description.txt"
            label_doc.parent.mkdir(parents=True)
            content_doc.parent.mkdir(parents=True)
            label_doc.write_text("dummy grading format documentation", encoding="utf-8")
            content_doc.write_text("dummy content documentation", encoding="utf-8")
            (root / "mystery_labels.csv").write_text("x,y\n1,5\n", encoding="utf-8")
            probe = probe_ume_jrf_layout(root)
        self.assertTrue(probe.available)
        self.assertIn("Vol1/doc/FJlabel/description.txt", probe.known_documents)
        self.assertEqual(
            probe.grading_schema_status,
            "label_documentation_present_requires_human_schema_inspection",
        )
        self.assertFalse(probe.automatic_label_parsing_allowed)
        self.assertTrue(any("mystery_labels.csv" in path for path in probe.candidate_label_files))


if __name__ == "__main__":
    unittest.main()
