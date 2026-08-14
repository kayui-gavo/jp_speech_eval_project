from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PRODUCT_PATHS = (
    ROOT / "src" / "jp_speech_eval" / "api.py",
    ROOT / "src" / "jp_speech_eval" / "evaluator.py",
    ROOT / "src" / "jp_speech_eval" / "consumer_dimension_policy.py",
    ROOT / "src" / "jp_speech_eval" / "product_score_v3.py",
    ROOT / "scripts" / "debug_ui.py",
    ROOT / "debug_ui" / "consumer_v2.html",
)
FORBIDDEN_MARKERS = (
    "ume_jrf_research",
    "UME-JRF",
    "ume-jrf",
)


class UmeJrfProductIsolationTest(unittest.TestCase):
    def test_research_only_corpus_is_not_imported_or_named_by_product_runtime(self) -> None:
        missing = [str(path.relative_to(ROOT)) for path in PRODUCT_PATHS if not path.is_file()]
        self.assertEqual(missing, [], f"expected product-path contract changed: {missing}")
        violations = []
        for path in PRODUCT_PATHS:
            text = path.read_text(encoding="utf-8")
            lowered = text.lower()
            for marker in FORBIDDEN_MARKERS:
                if marker.lower() in lowered:
                    violations.append(f"{path.relative_to(ROOT)} contains {marker!r}")
        self.assertEqual(
            violations,
            [],
            "research-only UME-JRF must not become a product runtime dependency: " + "; ".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
