import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_phone_gop_existing_data.py"
SPEC = importlib.util.spec_from_file_location("phone_existing_benchmark", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExistingPhoneGopBenchmarkDiscoveryTest(unittest.TestCase):
    def test_janon_isolated_suffix_is_speaker_independent(self) -> None:
        self.assertEqual(MODULE._janon_isolated_suffix("jpf1_i63.wav"), "i63.wav")
        self.assertEqual(MODULE._janon_isolated_suffix("chf2_i63.wav"), "i63.wav")
        self.assertEqual(MODULE._janon_isolated_suffix("jpm2_i5.wav"), "i5.wav")
        self.assertIsNone(MODULE._janon_isolated_suffix("VOICEACTRESS100_001.wav"))

    def test_janon_group_does_not_infer_learner_quality(self) -> None:
        native = {"jpf1", "jpf2", "jpm1", "jpm2"}
        self.assertEqual(MODULE._janon_group("jpf1", native), "janon_native_isolated")
        self.assertEqual(
            MODULE._janon_group("chf1", native),
            "janon_non_native_isolated_unlabeled_quality",
        )
        self.assertNotIn("bad", MODULE._janon_group("chf1", native))
        self.assertNotIn("error", MODULE._janon_group("chf1", native))

    def test_corpus_root_search_checks_repo_adjacent_locations(self) -> None:
        roots = MODULE._janon_root_candidates()
        self.assertGreaterEqual(len(roots), 2)
        self.assertEqual(len({str(path) for path in roots}), len(roots))
        self.assertTrue(all(path.name == "JANON" for path in roots))


if __name__ == "__main__":
    unittest.main()
