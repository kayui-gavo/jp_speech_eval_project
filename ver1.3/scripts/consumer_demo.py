from __future__ import annotations

import sys
from pathlib import Path

import debug_ui


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "assets" / "reference_cache" / "ramen_kudasai_aivis"
DEFAULT_WAV = DEFAULT_CACHE.with_suffix(".ref.wav")


def _has_option(args: list[str], name: str) -> bool:
    return name in args or any(arg.startswith(f"{name}=") for arg in args)


def _inject_defaults(args: list[str]) -> list[str]:
    injected: list[str] = []
    if not _has_option(args, "--cache"):
        injected.extend(["--cache", str(DEFAULT_CACHE)])
    if not _has_option(args, "--wav"):
        injected.extend(["--wav", str(DEFAULT_WAV)])
    if "--public-demo" not in args:
        injected.append("--public-demo")
    return [*injected, *args]


class ConsumerUiHandler(debug_ui.DebugUiHandler):
    """Keep the research UI intact while making `/` product-first for this launcher."""

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path in {"", "/"}:
            self.send_response(302)
            self.send_header("Location", "/consumer_v2.html")
            self.end_headers()
            return
        super().do_GET()


def main() -> None:
    if not DEFAULT_CACHE.with_suffix(".json").exists() or not DEFAULT_CACHE.with_suffix(".npz").exists():
        raise FileNotFoundError(
            "Bundled demo reference cache is missing: "
            f"{DEFAULT_CACHE}.json / {DEFAULT_CACHE}.npz"
        )
    if not DEFAULT_WAV.exists():
        raise FileNotFoundError(f"Bundled demo reference wav is missing: {DEFAULT_WAV}")

    debug_ui.DebugUiHandler = ConsumerUiHandler
    sys.argv = [sys.argv[0], *_inject_defaults(sys.argv[1:])]
    debug_ui.main()


if __name__ == "__main__":
    main()
