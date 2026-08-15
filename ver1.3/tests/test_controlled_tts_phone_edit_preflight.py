from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_controlled_tts_phone_edit_preflight",
    ROOT / "scripts" / "run_controlled_tts_phone_edit_preflight.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ControlledPhoneEditDefinitionTest(unittest.TestCase):
    def test_single_substitution_is_localized(self) -> None:
        result = MODULE._single_edit(["b", "a", "s", "u"], ["p", "a", "s", "u"], "substitution")
        self.assertTrue(result["available"])
        self.assertEqual(result["target_index"], 0)
        self.assertEqual(result["target_phone"], "b")
        self.assertEqual(result["error_phone"], "p")

    def test_single_deletion_is_localized(self) -> None:
        result = MODULE._single_edit(["k", "a", "cl", "k", "o"], ["k", "a", "k", "o"], "deletion")
        self.assertTrue(result["available"])
        self.assertEqual(result["target_index"], 2)
        self.assertEqual(result["target_phone"], "cl")

    def test_ambiguous_or_multi_edit_fails_closed(self) -> None:
        result = MODULE._single_edit(["b", "a"], ["p", "o"], "substitution")
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "expected_exactly_one_substitution")


if __name__ == "__main__":
    unittest.main()
