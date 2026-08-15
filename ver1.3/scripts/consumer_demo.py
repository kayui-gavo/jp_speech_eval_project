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


def _is_consumer_entry_path(path: str) -> bool:
    """Route ordinary demo entry points away from the legacy research renderer."""
    clean = str(path or "").split("?", 1)[0]
    return clean in {"", "/", "/index.html"}


def _render_consumer_user_facing(result: Any, *, mode: str | None = None, **kwargs: Any) -> dict[str, Any]:
    payload = _base_render_user_facing_result(result, mode=mode, **kwargs)
    resolved_mode = str(mode or result.get("details", {}).get("mode") or "reference")
    dimensions = build_consumer_score_dimensions(
        result,
        payload,
        mode=resolved_mode,
    )
    payload["score_dimensions"] = dimensions

    # The overall score is now computed by user_score_policy from the same four
    # semantic components shown here. Do not replace it with a second launcher-
    # only average: that previously made the visible total disagree with the
    # actual product scoring policy and caused research/debug and consumer
    # surfaces to report different totals for the same utterance.
    payload.setdefault("dimension_policy", {})
    payload["dimension_policy"].update({
        "version": "consumer_semantics_v4_shared_total",
        "top_level_dimensions": ["delivery_fluency", "clarity", "mora_timing", "intonation"],
        "always_show_four_scores_after_japanese_acceptance": True,
        "evidence_degrades_before_score_disappears": True,
        "consumer_total_policy": "shared_semantic_four_component_product_heuristic_v1",
        "consumer_total_weights": {
            "clarity": 0.30,
            "mora_timing": 0.25,
            "delivery_fluency": 0.25,
            "intonation": 0.20,
        },
        "consumer_total_product_calibrated": False,
        "prosody_is_not_a_peer_label_to_intonation": True,
        "lexical_pitch_accent_is_not_top_level_intonation": True,
        "recording_quality_is_not_clarity": True,
        "legacy_pronunciation_timing_proxy_is_not_clarity": True,
        "target_mismatch_disables_target_relative_dimension_evidence": True,
        "non_japanese_or_unusable_audio_can_still_be_no_score": True,
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
copy["zh-CN"].hero="只要确认是在说可评价的日语，就会从流畅度、清晰度、节奏和抑扬四个方向给出练习分。证据不足时会降低判断可信度，而不是让某一维消失。";
copy["zh-TW"].hero="只要確認是在說可評價的日語，就會從流暢度、清晰度、節奏和抑揚四個方向給出練習分。證據不足時會降低判斷可信度，而不是讓某一維消失。";
copy.ja.hero="評価可能な日本語発話として確認できた場合は、流暢さ・明瞭さ・リズム・抑揚の4項目を必ず練習スコアとして表示します。根拠が弱い場合は信頼度を下げ、項目自体は消しません。";
copy.en.hero="Once the utterance is accepted as scoreable Japanese, the demo always returns four practice scores: fluency, clarity, rhythm, and intonation. Weak evidence lowers confidence instead of making a dimension disappear.";
</script>
"""
    return html.replace("</body>", f"{patch}</body>").encode("utf-8")


class ConsumerUiHandler(debug_ui.DebugUiHandler):
    """Keep the research UI intact while making product routes consumer-first."""

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if _is_consumer_entry_path(self.path):
            self.send_response(302)
            self.send_header("Location", "/consumer_v2.html")
            self.end_headers()
            return
        if self.path.split("?", 1)[0] == "/consumer_v2.html":
            body = _consumer_html_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.split("?", 1)[0] == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
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
