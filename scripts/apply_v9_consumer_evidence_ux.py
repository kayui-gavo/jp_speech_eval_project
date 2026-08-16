from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"expected one match in {path}: {old[:80]!r}; got {text.count(old)}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Evidence semantics change without a ProductScore formula change.
replace_once(
    "ver1.3/src/jp_speech_eval/score_contract.py",
    'EVIDENCE_SCHEMA_VERSION = "consumer_evidence_v4"',
    'EVIDENCE_SCHEMA_VERSION = "consumer_evidence_v5"',
)
replace_once(
    "ver1.3/tests/test_transcript_assisted_language_gate.py",
    'assert product["evidence_schema_version"] == "consumer_evidence_v4"',
    'assert product["evidence_schema_version"] == "consumer_evidence_v5"',
)

# Expose measurement context at the public user-facing contract.
p = Path("ver1.3/src/jp_speech_eval/feedback_renderer.py")
text = p.read_text(encoding="utf-8")
text = text.replace(
    "from .consumer_dimension_policy import build_consumer_score_dimensions\n",
    "from .consumer_dimension_policy import build_consumer_score_dimensions\nfrom .measurement_context import build_measurement_context\n",
    1,
)
old = '''    return UserFacingResult(
        mode=policy.mode,
'''
new = '''    score_dimensions = _score_dimensions(result, gate, user_score, mode=policy.mode)
    measurement_context = build_measurement_context(
        result,
        score_dimensions,
        reliability_gate=gate.to_dict(),
        scoring_policy=policy.to_dict(),
    )
    payload = UserFacingResult(
        mode=policy.mode,
'''
if text.count(old) != 1:
    raise SystemExit("feedback_renderer return block not found")
text = text.replace(old, new, 1)
text = text.replace(
    '        score_dimensions=_score_dimensions(result, gate, user_score, mode=policy.mode),\n',
    '        score_dimensions=score_dimensions,\n',
    1,
)
old_tail = '''        evidence_schema_version=str(user_score.get("evidence_schema_version") or ""),
    ).to_dict()
'''
new_tail = '''        evidence_schema_version=str(user_score.get("evidence_schema_version") or ""),
    ).to_dict()
    payload["measurement_context"] = measurement_context
    return payload
'''
if text.count(old_tail) != 1:
    raise SystemExit("feedback_renderer tail not found")
text = text.replace(old_tail, new_tail, 1)
p.write_text(text, encoding="utf-8")

# Formal HF Space UI: facts instead of a fake unified confidence percentage.
p = Path("ver1.3/debug_ui/index.html")
html = p.read_text(encoding="utf-8")
html = html.replace(
    '<h2 data-i18n="reliability">本次结果可信度</h2>\n          <span class="pill" id="reliabilityLevel">暂无数据</span>',
    '<h2 data-consumer-copy="evidenceBasis">本次判断依据</h2>\n          <span class="pill" id="reliabilityLevel">等待结果</span>',
    1,
)

# Add consumer copy to each locale immediately before the dimensions map.
copy_inserts = {
    'modeGroup:"练习方式",\n        dimensions:{delivery_fluency:"流畅度", clarity:"明瞭度", mora_timing:"节奏", intonation:"抑扬"}':
    'modeGroup:"练习方式",\n        evidenceBasis:"本次判断依据", evidenceWaiting:"等待结果", recordingStatus:"录音状态", recordingGood:"良好", recordingUsable:"可用", recordingAffected:"录音条件有影响", recordingRetry:"建议重录",\n        scoreEvidence:"评分依据", evidenceBadge:"有依据 {active}/{total}", evidenceComposition:"音声确认 {measured} · 大致参考 {broad} · 参考值 {neutral} · 未测定 {unavailable}",\n        localDetail:"局部细节", detailAvailable:"可看局部细节", detailLimited:"细节判断有限", detailBroadOnly:"只做整体评价", detailUnavailable:"暂不可判断",\n        resultLimitation:"本次限制", noResultLimitation:"无额外限制", evidenceShownBelow:"判断依据见下方",\n        dimensions:{delivery_fluency:"流畅度", clarity:"明瞭度", mora_timing:"节奏", intonation:"抑扬"}',
    'modeGroup:"練習方式",\n        dimensions:{delivery_fluency:"流暢度", clarity:"明瞭度", mora_timing:"節奏", intonation:"抑揚"}':
    'modeGroup:"練習方式",\n        evidenceBasis:"本次判斷依據", evidenceWaiting:"等待結果", recordingStatus:"錄音狀態", recordingGood:"良好", recordingUsable:"可用", recordingAffected:"錄音條件有影響", recordingRetry:"建議重錄",\n        scoreEvidence:"評分依據", evidenceBadge:"有依據 {active}/{total}", evidenceComposition:"音聲確認 {measured} · 大致參考 {broad} · 參考值 {neutral} · 未測定 {unavailable}",\n        localDetail:"局部細節", detailAvailable:"可看局部細節", detailLimited:"細節判斷有限", detailBroadOnly:"只做整體評價", detailUnavailable:"暫不可判斷",\n        resultLimitation:"本次限制", noResultLimitation:"無額外限制", evidenceShownBelow:"判斷依據見下方",\n        dimensions:{delivery_fluency:"流暢度", clarity:"明瞭度", mora_timing:"節奏", intonation:"抑揚"}',
    'modeGroup:"練習方法",\n        dimensions:{delivery_fluency:"流暢さ", clarity:"明瞭さ", mora_timing:"リズム", intonation:"抑揚"}':
    'modeGroup:"練習方法",\n        evidenceBasis:"今回の判断根拠", evidenceWaiting:"結果待ち", recordingStatus:"録音状態", recordingGood:"良好", recordingUsable:"利用可能", recordingAffected:"録音条件の影響あり", recordingRetry:"録り直し推奨",\n        scoreEvidence:"評価の根拠", evidenceBadge:"根拠あり {active}/{total}", evidenceComposition:"音声から確認 {measured} · 大まかな目安 {broad} · 参考値 {neutral} · 未測定 {unavailable}",\n        localDetail:"細部の判定", detailAvailable:"細部も確認可能", detailLimited:"細部の判定は限定的", detailBroadOnly:"全体評価のみ", detailUnavailable:"今回は判定不可",\n        resultLimitation:"今回の制限", noResultLimitation:"追加の制限なし", evidenceShownBelow:"判断根拠は下に表示",\n        dimensions:{delivery_fluency:"流暢さ", clarity:"明瞭さ", mora_timing:"リズム", intonation:"抑揚"}',
    'modeGroup:"Practice mode",\n        dimensions:{delivery_fluency:"Fluency", clarity:"Clarity", mora_timing:"Rhythm", intonation:"Intonation"}':
    'modeGroup:"Practice mode",\n        evidenceBasis:"Evidence used this time", evidenceWaiting:"Waiting for result", recordingStatus:"Recording status", recordingGood:"Good", recordingUsable:"Usable", recordingAffected:"Recording conditions affected analysis", recordingRetry:"Record again",\n        scoreEvidence:"Score evidence", evidenceBadge:"Evidence {active}/{total}", evidenceComposition:"Audio evidence {measured} · Broad estimate {broad} · Reference value {neutral} · Unavailable {unavailable}",\n        localDetail:"Local detail", detailAvailable:"Local detail available", detailLimited:"Local detail is limited", detailBroadOnly:"Overall evaluation only", detailUnavailable:"Unavailable this time",\n        resultLimitation:"Current limitation", noResultLimitation:"No additional limitation", evidenceShownBelow:"Evidence details shown below",\n        dimensions:{delivery_fluency:"Fluency", clarity:"Clarity", mora_timing:"Rhythm", intonation:"Intonation"}',
}
for old_copy, new_copy in copy_inserts.items():
    if html.count(old_copy) != 1:
        raise SystemExit(f"consumer copy insertion not unique: {old_copy[:50]}")
    html = html.replace(old_copy, new_copy, 1)

# Consumer formatted copy helper.
old_cc = '''    function cc(key) {
      const pack = consumerCopy[state.locale] || consumerCopy["zh-CN"];
      return pack[key] ?? consumerCopy["zh-CN"][key] ?? key;
    }

    function ccDimension(key, fallback="") {
'''
new_cc = '''    function cc(key) {
      const pack = consumerCopy[state.locale] || consumerCopy["zh-CN"];
      return pack[key] ?? consumerCopy["zh-CN"][key] ?? key;
    }

    function ccf(key, vars = {}) {
      return String(cc(key)).replace(/\\{(\\w+)\\}/g, (_, name) => vars[name] ?? "");
    }

    function ccDimension(key, fallback="") {
'''
if html.count(old_cc) != 1:
    raise SystemExit("cc helper block not found")
html = html.replace(old_cc, new_cc, 1)

# Practice status is learner performance; remove the pseudo unified confidence label.
old_card = 'cards.push(`<div class="score"><span>${t("practiceResult")}</span><strong>${practiceResultLabel(status)}</strong><span>${t("reliabilitySimple")}：${reliabilityDisplay(reliability)}</span></div>`);'
new_card = 'cards.push(`<div class="score"><span>${t("practiceResult")}</span><strong>${practiceResultLabel(status)}</strong><span>${cc("evidenceShownBelow")}</span></div>`);'
if html.count(old_card) != 1:
    raise SystemExit("practice result confidence card not found")
html = html.replace(old_card, new_card, 1)

start = html.index('    function renderReliability(result, userFacing = null) {')
end = html.index('\n    function renderEndpointing(endpointing) {', start)
new_reliability = r'''    function renderReliability(result, userFacing = null) {
      const reliability = result.details?.reliability || {};
      const recording = result.details?.recording_quality || {};
      const mode = result.details?.mode || state.lastPayload?.mode || "reference";
      const gate = userFacing?.debug?.reliability_gate || {};
      const policy = userFacing?.debug?.scoring_policy || {};
      const context = userFacing?.measurement_context || {};
      const dimensions = Array.isArray(userFacing?.score_dimensions) ? userFacing.score_dimensions : [];

      const fallbackCounts = { measured_proxy:0, broad_proxy:0, neutral_prior:0, unavailable:0 };
      dimensions.forEach((item) => {
        const stateName = Object.prototype.hasOwnProperty.call(fallbackCounts, item.evidence_state) ? item.evidence_state : "unavailable";
        fallbackCounts[stateName] += 1;
      });
      const evidence = context.dimension_evidence || {
        dimension_count: dimensions.length,
        evidence_backed_count: fallbackCounts.measured_proxy + fallbackCounts.broad_proxy,
        measured_proxy_count: fallbackCounts.measured_proxy,
        broad_proxy_count: fallbackCounts.broad_proxy,
        neutral_prior_count: fallbackCounts.neutral_prior,
        unavailable_count: fallbackCounts.unavailable
      };
      const total = Number(evidence.dimension_count || dimensions.length || 4);
      const active = Number(evidence.evidence_backed_count || 0);
      $("reliabilityLevel").textContent = ccf("evidenceBadge", { active, total });
      $("reliabilityLevel").className = active >= total && total > 0 ? "pill good" : active > 0 ? "pill warn" : "pill";

      let recordingState = context.recording_state;
      if (!recordingState) {
        const recordingScore = Number(recording.score ?? reliability.recording_quality ?? 1);
        recordingState = recordingScore < .20 ? "retry" : recordingScore < .55 ? "affected" : recordingScore < .75 ? "usable" : "good";
      }
      const recordingText = ({
        good: cc("recordingGood"), usable: cc("recordingUsable"),
        affected: cc("recordingAffected"), retry: cc("recordingRetry")
      })[recordingState] || cc("recordingUsable");

      const evidenceText = ccf("evidenceComposition", {
        measured: Number(evidence.measured_proxy_count || 0),
        broad: Number(evidence.broad_proxy_count || 0),
        neutral: Number(evidence.neutral_prior_count || 0),
        unavailable: Number(evidence.unavailable_count || 0)
      });

      let detailState = context.local_detail_state;
      if (!detailState) {
        const blocked = new Set(gate.blocked_categories || []);
        const localBlocked = ["pitch", "special_mora", "pronunciation", "pronunciation_detail"].some((key) => blocked.has(key));
        detailState = policy.broad_mode || policy.weak_reference ? "broad_only" : localBlocked ? "limited" : "available";
      }
      const detailText = ({
        available: cc("detailAvailable"), limited: cc("detailLimited"),
        broad_only: cc("detailBroadOnly"), unavailable: cc("detailUnavailable")
      })[detailState] || cc("detailLimited");

      const limitation = context.primary_limitation || (gate.messages || [])[0] || cc("noResultLimitation");
      const rows = [
        [cc("recordingStatus"), recordingText],
        [cc("scoreEvidence"), evidenceText],
        [cc("localDetail"), detailText],
        [cc("resultLimitation"), limitation]
      ];
      $("reliabilityGrid").innerHTML = rows
        .map(([label, value]) => `<div class="reliability-item"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
        .join("");

      const content = result.details?.content_match || {};
      const contentWarnings = [];
      if (content.transcript) {
        contentWarnings.unshift(`${t("asrPrefix")}：${content.transcript} / ${content.transcript_kana || ""}`);
      }
      if (mode.includes("fallback_acoustic")) {
        contentWarnings.unshift(activeModeText().fallbackAcoustic || modeText["zh-CN"].fallbackAcoustic);
      }
      if (mode === "reference_free_acoustic" || mode === "acoustic" || mode.includes("fallback_acoustic")) {
        contentWarnings.push(t("acousticModeWarning"));
      }
      if (mode === "transcript_assisted_light") {
        contentWarnings.push(t("transcriptModeWarning"));
      }
      $("reliabilityWarnings").innerHTML = contentWarnings
        .map((warning) => `<li>${warning}</li>`)
        .join("");
    }
'''
html = html[:start] + new_reliability + html[end:]
p.write_text(html, encoding="utf-8")

# Add v9 branch/test to permanent light CI.
p = Path(".github/workflows/baseline-evolution-light-tests.yml")
yml = p.read_text(encoding="utf-8")
yml = yml.replace("      - scoring-integrity-v8\n", "      - scoring-integrity-v8\n      - consumer-evidence-ux-v9\n", 1)
yml = yml.replace(
    '      - "ver1.3/src/jp_speech_eval/partial_evidence_aggregate.py"\n',
    '      - "ver1.3/src/jp_speech_eval/partial_evidence_aggregate.py"\n      - "ver1.3/src/jp_speech_eval/measurement_context.py"\n',
    1,
)
yml = yml.replace(
    '      - "ver1.3/tests/test_scoring_integrity_v8.py"\n',
    '      - "ver1.3/tests/test_scoring_integrity_v8.py"\n      - "ver1.3/tests/test_measurement_context_v9.py"\n      - "ver1.3/tests/test_consumer_evidence_ux_v9.py"\n',
    1,
)
yml = yml.replace(
    '            tests/test_scoring_integrity_v8.py \\\n',
    '            tests/test_scoring_integrity_v8.py \\\n            tests/test_measurement_context_v9.py \\\n            tests/test_consumer_evidence_ux_v9.py \\\n',
    1,
)
p.write_text(yml, encoding="utf-8")

# v9 integration/UI contract tests.
Path("ver1.3/tests/test_consumer_evidence_ux_v9.py").write_text(r'''from __future__ import annotations

from pathlib import Path

from jp_speech_eval.feedback_renderer import render_user_facing_result

ROOT = Path(__file__).resolve().parents[1]


def _fixed_result():
    moras = ["ラ", "ー", "メ", "ン", "ヲ", "ク", "ダ", "サ", "イ"]
    return {
        "target_text": "ラーメンをください",
        "moras": moras,
        "mora_table": [{"mora": m, "start_sec": i * .18, "end_sec": (i + 1) * .18} for i, m in enumerate(moras)],
        "total_score": 84, "pronunciation_score": 80, "prosody_score": 82, "fluency_score": 88, "tone_score": 80,
        "feedback": [], "alignment_mode": "cached_dtw",
        "details": {
            "mode": "reference", "verified_level": "human_checked", "pitch_target_source": "human_checked",
            "recording_quality": {"score": .92}, "content_match": {"status": "pass"},
            "alignment": {"mode": "cached_dtw", "available": True, "confidence": .90},
            "reliability": {"level": "high", "overall": .95, "endpointing": .95, "alignment": .90, "mora_evidence": .90, "f0_coverage": .90, "recording_quality": .92},
            "fluency": {"rate_score": 88, "pause_score": 90, "speech_rate_mora_per_sec": 5.2},
            "pronunciation": {"mora_duration_cv": .12, "special_mora_diagnostics": []},
            "prosody": {"contour_corr": .70, "contour_valid_mora_count": 9, "transition_agreement": .75, "pitch_target_source": "human_checked", "note": "ok"},
            "tone": {"pitch_range_log": .35, "pitch_score": 82},
            "mora_evidence": [{"judgement_available": True, "boundary_confidence": .85, "energy_coverage": .8} for _ in moras],
        },
    }


def test_user_facing_payload_exposes_measurement_context_not_one_confidence_probability():
    rendered = render_user_facing_result(_fixed_result(), mode="reference")
    context = rendered["measurement_context"]
    assert context["schema"] == "consumer_measurement_context_v1"
    assert context["recording_state"] == "good"
    assert context["dimension_evidence"]["dimension_count"] == 4
    assert context["single_confidence_percentage_allowed"] is False
    assert context["raw_reliability_numeric_user_facing"] is False
    assert rendered["evidence_schema_version"] == "consumer_evidence_v5"
    assert rendered["debug"]["reliability_gate"]["reliability"] == "high"


def test_space_ui_uses_measurement_facts_not_raw_reliability_percent():
    html = (ROOT / "debug_ui" / "index.html").read_text(encoding="utf-8")
    assert 'data-consumer-copy="evidenceBasis"' in html
    assert 'cc("recordingStatus")' in html
    assert 'cc("scoreEvidence")' in html
    assert 'cc("localDetail")' in html
    assert 'cc("resultLimitation")' in html
    assert 'measurement_context' in html
    assert 'Number(reliability.overall || 0) * 100' not in html
    assert '${t("reliabilitySimple")}：${reliabilityDisplay(reliability)}' not in html


def test_space_ui_has_evidence_context_copy_for_all_locales():
    html = (ROOT / "debug_ui" / "index.html").read_text(encoding="utf-8")
    for text in (
        'evidenceBasis:"本次判断依据"', 'evidenceBasis:"本次判斷依據"',
        'evidenceBasis:"今回の判断根拠"', 'evidenceBasis:"Evidence used this time"',
    ):
        assert text in html
''', encoding="utf-8")
