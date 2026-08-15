from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_controlled_tts_phone_edit_preflight.py"
SPEC = importlib.util.spec_from_file_location("controlled_tts_contract", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ControlledTtsConstructContractTest(unittest.TestCase):
    def test_every_control_edit_hits_declared_construct(self) -> None:
        seen = set()
        for case in MODULE.CASES:
            with self.subTest(case_id=case["case_id"]):
                target_phones, target_roles = MODULE._target_contract(case["target"])
                spoken_phones = MODULE._phones(case["spoken_error"])
                edit = MODULE._single_edit(target_phones, spoken_phones, case["edit_type"])
                self.assertTrue(target_roles["available"], target_roles.get("summary"))
                self.assertTrue(edit["available"], edit)
                index = int(edit["target_index"])
                role = target_roles["roles"][index]
                self.assertEqual(role, case["expected_construct_role"])
                seen.add(role)
        self.assertEqual(
            seen,
            {MODULE.ORDINARY_ROLE, MODULE.SPECIAL_MORA_ROLE, MODULE.LONG_VOWEL_ROLE},
        )

    def test_long_vowel_control_is_one_phone_deletion(self) -> None:
        case = next(row for row in MODULE.CASES if row["case_id"] == "long_vowel_deletion")
        target_phones, target_roles = MODULE._target_contract(case["target"])
        spoken_phones = MODULE._phones(case["spoken_error"])
        edit = MODULE._single_edit(target_phones, spoken_phones, "deletion")
        self.assertTrue(edit["available"], edit)
        index = int(edit["target_index"])
        self.assertEqual(target_roles["roles"][index], MODULE.LONG_VOWEL_ROLE)
        self.assertIn(str(edit["target_phone"]), {"a", "i", "u", "e", "o"})


if __name__ == "__main__":
    unittest.main()
