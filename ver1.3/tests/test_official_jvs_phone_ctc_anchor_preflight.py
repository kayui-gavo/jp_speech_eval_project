import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_official_jvs_phone_ctc_anchor_preflight.py"
SPEC = importlib.util.spec_from_file_location("jvs_anchor_preflight", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

DOWNLOAD_SCRIPT = ROOT / "scripts" / "download_official_jvs_samples.py"
DOWNLOAD_SPEC = importlib.util.spec_from_file_location("jvs_download_contract", DOWNLOAD_SCRIPT)
assert DOWNLOAD_SPEC and DOWNLOAD_SPEC.loader
DOWNLOAD = importlib.util.module_from_spec(DOWNLOAD_SPEC)
DOWNLOAD_SPEC.loader.exec_module(DOWNLOAD)


def _find_subsequence(sequence, pattern):
    return [
        index
        for index in range(len(sequence) - len(pattern) + 1)
        if list(sequence[index : index + len(pattern)]) == list(pattern)
    ]


class OfficialJvsPhoneCtcAnchorPreflightTest(unittest.TestCase):
    def test_rotation_preserves_phone_count_and_changes_order(self) -> None:
        source = [1, 2, 3, 4, 5, 6]
        rotated = MODULE._rotate_control(source)
        self.assertEqual(len(rotated), len(source))
        self.assertCountEqual(rotated, source)
        self.assertNotEqual(rotated, source)

    def test_sequence_metrics_favor_matching_ctc_path(self) -> None:
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

    def test_reviewed_phone_target_repeats_identical_meiou_block(self) -> None:
        phones = list(DOWNLOAD.TARGET_PHONES)
        meiou = ["my", "o", "o", "o", "o"]
        contaminated = ["m", "i", "y", "o", "u", "o", "u"]
        starts = _find_subsequence(phones, meiou)
        self.assertEqual(len(phones), 75)
        self.assertEqual(len(starts), 2)
        self.assertEqual(_find_subsequence(phones, contaminated), [])
        self.assertEqual(DOWNLOAD.PHONE_PROVENANCE["policy"], "reviewed_logical_phone_override_v1")

    def test_reviewed_segments_flatten_exactly_to_target_phones(self) -> None:
        flattened = [
            str(phone)
            for segment in DOWNLOAD.TARGET_PHONE_SEGMENTS
            for phone in segment["phones"]
        ]
        self.assertEqual(flattened, list(DOWNLOAD.TARGET_PHONES))
        meiou_segments = [
            segment for segment in DOWNLOAD.TARGET_PHONE_SEGMENTS
            if segment["surface"] == "明王"
        ]
        self.assertEqual(len(meiou_segments), 2)
        self.assertEqual(
            [tuple(segment["phones"]) for segment in meiou_segments],
            [("my", "o", "o", "o", "o"), ("my", "o", "o", "o", "o")],
        )

    def test_reviewed_phone_index_metadata_is_contiguous_and_segment_traceable(self) -> None:
        metadata = list(DOWNLOAD.TARGET_PHONE_INDEX_METADATA)
        phones = list(DOWNLOAD.TARGET_PHONES)
        self.assertEqual(len(metadata), len(phones))
        self.assertEqual([int(row["phone_index"]) for row in metadata], list(range(len(phones))))
        self.assertEqual([str(row["phone"]) for row in metadata], phones)
        meiou_rows = [row for row in metadata if row["segment_surface"] == "明王"]
        self.assertEqual(len(meiou_rows), 10)
        self.assertEqual(
            [str(row["phone"]) for row in meiou_rows[:5]],
            ["my", "o", "o", "o", "o"],
        )
        self.assertEqual(
            [str(row["phone"]) for row in meiou_rows[5:]],
            ["my", "o", "o", "o", "o"],
        )

    def test_manifest_v5_requires_same_reviewed_phone_target_for_all_speakers(self) -> None:
        rows = []
        for speaker in ("jvs001", "jvs002", "jvs003"):
            rows.append(
                {
                    "speaker": speaker,
                    "target_text": MODULE.TARGET_TEXT,
                    "target_reading": DOWNLOAD.TARGET_READING,
                    "target_phones": list(DOWNLOAD.TARGET_PHONES),
                    "target_phone_source": "reviewed_logical_phone_override_v1",
                    "automatic_surface_g2p_is_safe_for_anchor": False,
                    "automatic_kana_g2p_is_phone_exact_for_anchor": False,
                }
            )
        payload = {
            "schema": "jvs_official_samples_manifest_v5",
            "target_phone_override_required": True,
            "samples": rows,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            loaded = MODULE._load_manifest(path)
        reading, phones = MODULE._reviewed_target(loaded)
        self.assertEqual(reading, DOWNLOAD.TARGET_READING)
        self.assertEqual(phones, list(DOWNLOAD.TARGET_PHONES))

    def test_old_manifest_schema_is_rejected_for_phone_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(
                json.dumps({"schema": "jvs_official_samples_manifest_v4", "samples": []}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "requires reviewed-phone manifest v5"):
                MODULE._load_manifest(path)

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
                "target_phone_source": "reviewed_logical_phone_override_v1",
                "automatic_kana_g2p_is_phone_exact_for_anchor": False,
            }
        )
        self.assertEqual(result["raw_sha256"], "a" * 64)
        self.assertEqual(result["sample_rate"], 24000)
        self.assertTrue(result["semantic_duration_verified"])
        self.assertFalse(result["raw_transport_hash_is_acoustic_identity"])
        self.assertEqual(result["target_phone_source"], "reviewed_logical_phone_override_v1")
        self.assertFalse(result["automatic_kana_g2p_is_phone_exact_for_anchor"])

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
