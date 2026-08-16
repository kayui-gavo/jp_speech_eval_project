from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "debug_ui" / "index.html"


CONSUMER_COPY_REPLACEMENTS = {
    'evidenceMeasured:"音声证据", evidenceBroad:"大致参考", evidenceNeutral:"参考值", evidenceUnavailable:"未测定",': 'evidenceMeasured:"音声证据", evidenceBroad:"大致参考", evidenceNeutral:"参考值", evidenceUnavailable:"未测定", resultBasis:"结果依据", recordingStatus:"录音状态", scoreEvidence:"评分依据", recordingGood:"良好", recordingUsable:"可用", recordingCare:"需注意", evidenceStrong:"较充分", evidencePartial:"部分依据", evidenceLimited:"有限",',
    'evidenceMeasured:"音聲證據", evidenceBroad:"大致參考", evidenceNeutral:"參考值", evidenceUnavailable:"未測定",': 'evidenceMeasured:"音聲證據", evidenceBroad:"大致參考", evidenceNeutral:"參考值", evidenceUnavailable:"未測定", resultBasis:"結果依據", recordingStatus:"錄音狀態", scoreEvidence:"評分依據", recordingGood:"良好", recordingUsable:"可用", recordingCare:"需注意", evidenceStrong:"較充分", evidencePartial:"部分依據", evidenceLimited:"有限",',
    'evidenceMeasured:"音声から確認", evidenceBroad:"大まかな目安", evidenceNeutral:"参考値", evidenceUnavailable:"未測定",': 'evidenceMeasured:"音声から確認", evidenceBroad:"大まかな目安", evidenceNeutral:"参考値", evidenceUnavailable:"未測定", resultBasis:"結果の根拠", recordingStatus:"録音状態", scoreEvidence:"評価の根拠", recordingGood:"良好", recordingUsable:"利用可能", recordingCare:"要確認", evidenceStrong:"十分", evidencePartial:"一部参考", evidenceLimited:"限定的",',
    'evidenceMeasured:"Audio evidence", evidenceBroad:"Broad estimate", evidenceNeutral:"Reference value", evidenceUnavailable:"Unavailable",': 'evidenceMeasured:"Audio evidence", evidenceBroad:"Broad estimate", evidenceNeutral:"Reference value", evidenceUnavailable:"Unavailable", resultBasis:"Result basis", recordingStatus:"Recording status", scoreEvidence:"Score evidence", recordingGood:"Good", recordingUsable:"Usable", recordingCare:"Needs care", evidenceStrong:"Strong", evidencePartial:"Partial", evidenceLimited:"Limited",',
}


NEW_RENDER_RELIABILITY = '''    function renderReliability(result, userFacing = null) {
      const reliability = result.details?.reliability || {};
      const content = result.details?.content_match || {};
      const recording = result.details?.recording_quality || {};
      const evidence = result.details?.mora_evidence_summary || {};
      const mode = result.details?.mode || state.lastPayload?.mode || "reference";
      const gate = userFacing?.debug?.reliability_gate || {};
      const policy = userFacing?.debug?.scoring_policy || {};
      const analyzability = userFacing?.recording_analyzability || {};
      const scoreEvidence = userFacing?.score_evidence || {};
      const legacyLevel = userFacing?.reliability || gate.reliability || reliability.level || "unknown";

      const recordingValue = Number(recording.score ?? reliability.recording_quality ?? 0);
      const recordingLevel = analyzability.level || (recordingValue >= 0.80 ? "high" : recordingValue >= 0.50 ? "medium" : "low");
      const evidenceLevel = scoreEvidence.level || legacyLevel;
      const recordingLabel = recordingLevel === "high" ? cc("recordingGood") : recordingLevel === "medium" ? cc("recordingUsable") : cc("recordingCare");
      const evidenceLabel = evidenceLevel === "high" ? cc("evidenceStrong") : evidenceLevel === "medium" ? cc("evidencePartial") : cc("evidenceLimited");

      $("reliabilityLevel").textContent = `${cc("scoreEvidence")} · ${evidenceLabel}`;
      $("reliabilityLevel").className = evidenceLevel === "high" ? "pill good" : evidenceLevel === "medium" ? "pill warn" : "pill bad";

      const reliable = [];
      if ((recording.score ?? reliability.recording_quality ?? 0) >= 0.75) reliable.push(t("recordingClear"));
      if (content.status === "pass" || mode !== "reference") reliable.push(t("contentUsable"));
      if ((reliability.mora_evidence ?? 0) >= 0.65 || (evidence.judgement_available_count || 0) >= 3) reliable.push(t("timingEvidenceUsable"));

      const blocked = gate.blocked_categories || [];
      const caution = [];
      if (blocked.includes("pitch") || (reliability.f0_coverage ?? 1) < 0.5) caution.push(t("pitchLimited"));
      if (blocked.includes("pronunciation") || blocked.includes("special_mora")) caution.push(t("moraDetailLimited"));
      if (policy.weak_reference) caution.push(t("weakReferenceNote"));

      const rows = [
        [cc("recordingStatus"), recordingLabel],
        [cc("scoreEvidence"), evidenceLabel],
        [t("reliableParts"), reliable.length ? reliable.join(" / ") : t("noData")],
        [t("cautionParts"), caution.length ? caution.join(" / ") : t("noMajorLimit")]
      ];
      $("reliabilityGrid").innerHTML = rows
        .map(([label, value]) => `<div class="reliability-item"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
        .join("");

      const contentWarnings = [];
      const mainIssue = (gate.messages || [])[0];
      if (mainIssue) contentWarnings.push(mainIssue);
      if (content.transcript) contentWarnings.push(`${t("asrPrefix")}：${content.transcript} / ${content.transcript_kana || ""}`);
      if (mode.includes("fallback_acoustic")) contentWarnings.push(activeModeText().fallbackAcoustic || modeText["zh-CN"].fallbackAcoustic);
      if (mode === "reference_free_acoustic" || mode === "acoustic" || mode.includes("fallback_acoustic")) contentWarnings.push(t("acousticModeWarning"));
      if (mode === "transcript_assisted_light") contentWarnings.push(t("transcriptModeWarning"));
      $("reliabilityWarnings").innerHTML = [...new Set(contentWarnings)]
        .map((warning) => `<li>${escapeHtml(warning)}</li>`)
        .join("");
    }

'''


def patch_text(text: str) -> str:
    old_heading = '<h2 data-i18n="reliability">本次结果可信度</h2>'
    new_heading = '<h2 data-consumer-copy="resultBasis">结果依据</h2>'
    if old_heading not in text:
        if new_heading in text:
            return text
        raise RuntimeError("reliability heading marker not found")
    text = text.replace(old_heading, new_heading, 1)

    for old, new in CONSUMER_COPY_REPLACEMENTS.items():
        if old not in text:
            raise RuntimeError(f"consumer copy marker not found: {old[:48]}")
        text = text.replace(old, new, 1)

    start_marker = "    function renderReliability(result, userFacing = null) {"
    end_marker = "    function renderEndpointing(endpointing) {"
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError("renderReliability block markers not found")
    text = text[:start] + NEW_RENDER_RELIABILITY + text[end:]

    if "userFacing?.recording_analyzability" not in text:
        raise RuntimeError("recording analyzability was not wired into UI")
    if "userFacing?.score_evidence" not in text:
        raise RuntimeError("score evidence was not wired into UI")
    block_start = text.index("function renderReliability(result, userFacing = null)")
    block_end = text.index("function renderEndpointing", block_start)
    if "* 100).toFixed(0)}%" in text[block_start:block_end]:
        raise RuntimeError("probability-like reliability percentage remains in public renderer")
    return text


def main() -> None:
    original = INDEX.read_text(encoding="utf-8")
    patched = patch_text(original)
    INDEX.write_text(patched, encoding="utf-8")


if __name__ == "__main__":
    main()
