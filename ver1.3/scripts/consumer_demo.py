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
        "version": "consumer_semantics_v5_confidence_surface",
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
        "low_confidence_dimension_marker": "reference_estimate",
        "target_mismatch_copy_is_nonpunitive": True,
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
<style>
/* Consumer semantic layer: keep uncertainty visible without turning the page
   into a research dashboard. */
.dim{grid-template-columns:145px 58px 52px minmax(80px,1fr)}
.dim .dim-confidence{font-family:var(--sans);font-size:9.8px;line-height:1;letter-spacing:.04em;color:var(--muted-soft);white-space:nowrap}
.dim .dim-confidence.low{color:var(--rose-deep)}
@media(max-width:760px){.dim{grid-template-columns:minmax(92px,1fr) 48px 44px minmax(58px,.9fr);gap:10px}.dim .dim-confidence{font-size:9px}}
</style>
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

const consumerConfidenceMarker={
  "zh-CN":{low:"参考"},"zh-TW":{low:"參考"},ja:{low:"参考"},en:{low:"Est."}
};
const consumerResultCopy={
  "zh-CN":{
    mismatchTitle:"句子不同，但这段日语仍然可以评分",
    mismatchCopy:"没有把你和固定例句硬比较。下面保留的是对这段日语本身可用的练习分；依赖目标句的局部纠错会暂时收起。",
    normalCopy:"四项都是练习分。标记“参考”的项目表示当前证据较弱，适合看方向，不适合当作精密测量。"
  },
  "zh-TW":{
    mismatchTitle:"句子不同，但這段日語仍然可以評分",
    mismatchCopy:"沒有把你和固定例句硬比較。下面保留的是對這段日語本身可用的練習分；依賴目標句的局部糾錯會暫時收起。",
    normalCopy:"四項都是練習分。標記「參考」的項目表示目前證據較弱，適合看方向，不適合當作精密測量。"
  },
  ja:{
    mismatchTitle:"お題とは違いますが、日本語として評価できます",
    mismatchCopy:"固定例文との一致度では減点していません。この発話そのものから出せる練習スコアを表示し、お題依存の細かな指摘だけを控えています。",
    normalCopy:"4項目はいずれも練習用の目安です。「参考」は根拠が弱い項目で、傾向を見るための推定値です。"
  },
  en:{
    mismatchTitle:"Different sentence, still scoreable as Japanese",
    mismatchCopy:"You are not being penalized for missing the fixed prompt. The scores below use evidence available from this Japanese utterance itself; prompt-specific corrections are withheld.",
    normalCopy:"All four values are practice scores. “Est.” marks a lower-confidence estimate that is useful for direction, not precise measurement."
  }
};
function consumerIsTargetMismatch(payload){
  const u=payload?.user_facing||{};
  const key=u?.debug?.user_score_policy?.main_message_key||"";
  return key==="content_mismatch_general_score"||payload?.mode==="reference_mismatch_general_japanese"||u?.dimension_policy?.target_mismatch===true;
}
function consumerDecorateResult(payload){
  const u=payload?.user_facing||{};
  const dims=Array.isArray(u.score_dimensions)?u.score_dimensions:[];
  const rows=[...document.querySelectorAll("#dimensions .dim")];
  rows.forEach((row,i)=>{
    row.querySelector(".dim-confidence")?.remove();
    const confidence=String(dims[i]?.confidence||"").toLowerCase();
    const tag=document.createElement("small");
    tag.className=`dim-confidence ${confidence}`;
    tag.textContent=consumerConfidenceMarker[locale]?.[confidence]||"";
    tag.setAttribute("aria-label",confidence?`confidence: ${confidence}`:"");
    row.appendChild(tag);
  });
  const text=consumerResultCopy[locale]||consumerResultCopy["zh-CN"];
  if(consumerIsTargetMismatch(payload)){
    $("resultTitle").textContent=text.mismatchTitle;
    $("resultCopy").textContent=text.mismatchCopy;
  }else{
    $("resultCopy").textContent=text.normalCopy;
  }
}
const _consumerBaseRenderResult=renderResult;
renderResult=function(payload){
  _consumerBaseRenderResult(payload);
  consumerDecorateResult(payload);
};
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
