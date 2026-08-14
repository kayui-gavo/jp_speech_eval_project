import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_official_jvs_phone_ctc_anchor_preflight.py"
SPEC = importlib.util.spec_from_file_location("jvs_anchor_preflight", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OfficialJvsPhoneCtcAnchorPreflightTest(unittest.TestCase):
    def test_rotation_preserves_phone_count_and_changes_order(self) -> None:
        source = [1, 2, 3, 4, 5, 6]
        rotated = MODULE._rotate_control(source)
        self.assertEqual(len(rotated), len(source))
        self.assertCountEqual(rotated, source)
        self.assertNotEqual(rotated, source)

    def test_sequence_metrics_favor_matching_ctc_path(self) -> None:
        # vocab: blank=0, a=1, b=2. A five-frame CTC path strongly supports
        # a,b while the same-length rotated control b,a is much less likely.
        logits = np.asarray(
            [
                [7.0, 0.0, 0.0],
                [0.0, 7.0, 0.0],
                [7.0, 0.0, 0.0],
                [0.0, 0.0, 7.0],
                [7.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        result = MODULE._sequence_metrics(
            logits,
            ["a", "b"],
            {"<blank>": 0, "a": 1, "b": 2},
            0,
        )
        self.assertTrue(result["available"])
        self.assertGreater(result["canonical_minus_permuted"], 0.0)
        self.assertGreater(result["canonical_minus_permuted_per_frame"], 0.0)
        self.assertEqual(result["phone_count"], 2)
        self.assertFalse(result["control_is_pronunciation_error_label"])

    def test_sequence_metrics_reports_phone_inventory_mismatch(self) -> None:
        result = MODULE._sequence_metrics(
            np.zeros((5, 3), dtype=np.float64),
            ["a", "missing"],
            {"<blank>": 0, "a": 1, "b": 2},
            0,
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "phone_inventory_mismatch")
        self.assertEqual(result["missing_phones"], ["missing"])

    def test_current_semantic_manifest_keeps_raw_hash_as_provenance_only(self) -> None:
        result = MODULE._source_provenance(
            {
                "speaker": "jvs001",
                "raw_sha256": "a" * 64,
                "raw_bytes": 413854,
                "sample_rate": 24000,
                "sample_width_bytes": 2,
                "duration_sec": 8.621,
                "semantic_duration_verified": True,
                "raw_transport_hash_is_acoustic_identity": False,
                "google_drive_file_id": "reviewed-id",
            }
        )
        self.assertEqual(result["raw_sha256"], "a" * 64)
        self.assertEqual(result["sample_rate"], 24000)
        self.assertTrue(result["semantic_duration_verified"])
        self.assertFalse(result["raw_transport_hash_is_acoustic_identity"])

    def test_legacy_manifest_hash_is_read_only_for_old_artifacts(self) -> None:
        result = MODULE._source_provenance(
            {"speaker": "jvs001", "sha256": "b" * 64, "bytes": 778284}
        )
        self.assertEqual(result["raw_sha256"], "b" * 64)
        self.assertEqual(result["bytes"], 778284)
        self.assertFalse(result["raw_transport_hash_is_acoustic_identity"])

    def test_missing_manifest_hash_provenance_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing raw hash provenance"):
            MODULE._source_provenance({"speaker": "jvs001", "duration_sec": 8.621})


if __name__ == "__main__":
    unittest.main()
