from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts import debug_ui  # noqa: E402
from jp_speech_eval.consumer_dimension_policy import build_consumer_score_dimensions  # noqa: E402
from jp_speech_eval.feedback_renderer import render_user_facing_result as _base_render_user_facing_result  # noqa: E402


DEFAULT_CACHE = ROOT / "assets" / "reference_cache" / "ramen_kudasai_aivis"
DEFAULT_WAV = DEFAULT_CACHE.with_suffix(".ref.wav")
CONSUMER_HTML = ROOT / "debug_ui" / "consumer_v2.html"


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


def _render_consumer_user_facing(result: Any, *, mode: str | None = None, **kwargs: Any) -> dict[str, Any]:
    payload = _base_render_user_facing_result(result, mode=mode, **kwargs)
    payload["score_dimensions"] = build_consumer_score_dimensions(
        result,
        payload,
        mode=str(mode or result.get("details", {}).get("mode") or "reference"),
    )
    payload.setdefault("dimension_policy", {})
    payload["dimension_policy"].update({
        "version": "consumer_semantics_v2",
        "top_level_dimensions": ["delivery_fluency", "clarity", "mora_timing", "intonation"],
        "prosody_is_not_a_peer_label_to_intonation": True,
        "lexical_pitch_accent_is_not_top_level_intonation": True,
        "recording_quality_is_not_clarity": True,
        "legacy_pronunciation_timing_proxy_is_not_clarity": True,
    })
    return payload


def _consumer_html_bytes() -> bytes:
    html = CONSUMER_HTML.read_text(encoding="utf-8")
    patch = r"""
<script>
/* Preview-only semantic patch: backend fields remain backward compatible. */
dimensionLabel = function(k){
  const d={
    delivery_fluency:{"zh-CN":"流畅度","zh-TW":"流暢度",ja:"流暢さ",en:"Fluency"},
    clarity:{"zh-CN":"清晰度","zh-TW":"清晰度",ja:"明瞭さ",en:"Clarity"},
    mora_timing:{"zh-CN":"节奏","zh-TW":"節奏",ja:"リズム",en:"Rhythm"},
    intonation:{"zh-CN":"抑扬","zh-TW":"抑揚",ja:"抑揚",en:"Intonation"}
  };
  return d[k]?.[locale]||k;
};
copy["zh-CN"].hero="从流畅度、清晰度、节奏和抑扬四个方向看这次发话。录音质量只决定结果可信度，不会冒充清晰度；词汇重音放到详细反馈里。";
copy["zh-TW"].hero="從流暢度、清晰度、節奏和抑揚四個方向看這次發話。錄音品質只決定結果可信度，不會冒充清晰度；詞彙重音放到詳細回饋裡。";
copy.ja.hero="流暢さ・明瞭さ・リズム・抑揚の4方向から今回の発話を確認します。録音品質は信頼度として扱い、明瞭さの点数にはしません。語彙アクセントは詳細フィードバックで扱います。";
copy.en.hero="Review each attempt through four separate dimensions: fluency, clarity, rhythm, and intonation. Recording quality affects confidence rather than clarity; lexical pitch accent stays in detailed feedback.";
</script>
"""
    return html.replace("</body>", f"{patch}</body>").encode("utf-8")


class ConsumerUiHandler(debug_ui.DebugUiHandler):
    """Keep the research UI intact while making `/` product-first for this launcher."""

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path in {"", "/"}:
            self.send_response(302)
            self.send_header("Location", "/consumer_v2.html")
            self.end_headers()
            return
        if self.path.split("?", 1)[0] == "/consumer_v2.html":
            body = _consumer_html_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def _asr_confirmation_response(
        self,
        prompt: Any,
        wav_path: Path,
        mode: str,
        *,
        requires_user_confirmation: bool = True,
    ) -> None:
        """Reject clear non-Japanese speech before a pseudo-reference is created."""
        if getattr(prompt, "language_eligible", True) is False:
            debug_ui._json_response(
                self,
                {
                    "ok": False,
                    "mode": "asr_language_reject",
                    "error": str(getattr(prompt, "message", "这段录音没有可靠识别为日语。请用日语重新录制。")),
                    "language_eligible": False,
                    "language_reason": str(getattr(prompt, "language_reason", "detected_non_japanese")),
                    "asr_raw": dict(getattr(prompt, "asr_raw", {}) or {}),
                },
                status=422,
            )
            if not self.server.retain_uploads:  # type: ignore[attr-defined]
                wav_path.unlink(missing_ok=True)
            return
        super()._asr_confirmation_response(
            prompt,
            wav_path,
            mode,
            requires_user_confirmation=requires_user_confirmation,
        )


def main() -> None:
    if not DEFAULT_CACHE.with_suffix(".json").exists() or not DEFAULT_CACHE.with_suffix(".npz").exists():
        raise FileNotFoundError(
            "Bundled demo reference cache is missing: "
            f"{DEFAULT_CACHE}.json / {DEFAULT_CACHE}.npz"
        )
    if not DEFAULT_WAV.exists():
        raise FileNotFoundError(f"Bundled demo reference wav is missing: {DEFAULT_WAV}")

    debug_ui.DebugUiHandler = ConsumerUiHandler
    debug_ui.render_user_facing_result = _render_consumer_user_facing
    sys.argv = [sys.argv[0], *_inject_defaults(sys.argv[1:])]
    debug_ui.main()


if __name__ == "__main__":
    main()
