from pathlib import Path

p = Path('ver1.3/debug_ui/index.html')
s = p.read_text(encoding='utf-8')


def replace_once(old: str, new: str, label: str) -> None:
    global s
    if old not in s:
        raise SystemExit(f'missing anchor: {label}')
    s = s.replace(old, new, 1)


def replace_between(start: str, end: str, replacement: str, label: str) -> None:
    global s
    a = s.find(start)
    if a < 0:
        raise SystemExit(f'missing start: {label}')
    b = s.find(end, a)
    if b < 0:
        raise SystemExit(f'missing end: {label}')
    s = s[:a] + replacement + s[b:]

# 1) Keep the established visual system and add only the new practice/replay layer.
css = r'''
    .karaoke-section { border-top: 3px solid var(--accent); }
    .karaoke-intro { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:10px; }
    .karaoke-intro p { margin:0; color:var(--muted); font-size:12px; }
    .karaoke-lyrics {
      min-height: 82px;
      padding: 15px 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      font-size: clamp(21px, 2vw, 30px);
      font-weight: 720;
      line-height: 1.75;
      letter-spacing: .01em;
      color: #98a2b3;
      word-break: break-word;
    }
    .karaoke-lyrics.static { color: var(--ink); font-size:18px; font-weight:650; }
    .karaoke-token {
      appearance:none;
      border:0;
      min-height:0;
      padding:2px 4px;
      margin:0 1px;
      background:transparent;
      color:inherit;
      font:inherit;
      font-weight:inherit;
      line-height:inherit;
      border-radius:5px;
      cursor:pointer;
      overflow:visible;
      white-space:normal;
    }
    .karaoke-token.past { color:#667085; }
    .karaoke-token.active { color:var(--ink); background:var(--accent-soft); }
    .karaoke-token.low-asr { text-decoration:underline dotted #d7a34b; text-underline-offset:5px; }
    .karaoke-visual { position:relative; height:160px; margin-top:10px; border:1px solid var(--line); border-radius:8px; background:#fbfcfd; overflow:hidden; }
    .karaoke-visual canvas { width:100%; height:100%; display:block; }
    .karaoke-empty { position:absolute; inset:0; display:none; align-items:center; justify-content:center; padding:20px; text-align:center; color:var(--muted); font-size:12px; }
    .karaoke-empty.show { display:flex; }
    .karaoke-player { display:grid; grid-template-columns:42px minmax(0,1fr) 86px; gap:10px; align-items:center; margin-top:10px; }
    .karaoke-player button { min-height:36px; padding:0; }
    .karaoke-player input[type="range"] { width:100%; accent-color:var(--accent); }
    .karaoke-time { text-align:right; color:var(--muted); font:11px ui-monospace,SFMono-Regular,Menlo,monospace; }
    .karaoke-layers { display:flex; flex-wrap:wrap; gap:14px; margin-top:8px; color:var(--muted); font-size:12px; }
    .karaoke-layers label { display:inline-flex; align-items:center; gap:6px; cursor:pointer; }
    .karaoke-layers input { accent-color:var(--accent); }
    .score-evidence { display:inline-flex!important; width:max-content; margin-top:5px; padding:2px 7px; border:1px solid var(--line); border-radius:999px; background:#fff; font-size:10px!important; }
    .score-evidence.measured { color:var(--good); border-color:#abefc6; background:#ecfdf3; }
    .score-evidence.broad { color:var(--warn); border-color:#fedf89; background:#fffaeb; }
    .score-evidence.neutral { color:#6941c6; border-color:#d9d6fe; background:#f4f3ff; }
    .public-mode .debug-only-panel { display:none!important; }
'''
replace_once('    @media (max-width: 920px) {', css + '\n    @media (max-width: 920px) {', 'karaoke css')

# 2) Insert the karaoke replay directly after the existing score panel.
score_panel = '''      <section class="score-section">\n        <div class="panel-head">\n          <h2 data-i18n="scores">评分</h2>\n          <span class="pill" id="wavPath">等待录音</span>\n        </div>\n        <div class="panel-body">\n          <div class="score-grid" id="scoreGrid"></div>\n        </div>\n      </section>\n'''
karaoke_panel = score_panel + r'''

      <section class="karaoke-section" id="karaokeSection">
        <div class="panel-head">
          <h2 data-consumer-copy="karaokeReplay">跟读回放</h2>
          <span class="pill" id="karaokeSyncBadge">等待录音</span>
        </div>
        <div class="panel-body">
          <div class="karaoke-intro"><p id="karaokeSyncNote" data-consumer-copy="karaokeHint">播放录音时，字幕和声音走势会同步。</p></div>
          <div class="karaoke-lyrics static" id="karaokeLyrics" data-consumer-copy="karaokePlaceholder">完成一次评价后，这里会显示你的发话。</div>
          <div class="karaoke-visual">
            <canvas id="karaokeCanvas"></canvas>
            <div class="karaoke-empty" id="karaokeEmpty" data-consumer-copy="pitchUnavailable">这次没有足够的音高信息可供可视化；这不代表抑扬表现差。</div>
          </div>
          <div class="karaoke-player">
            <button id="karaokePlayBtn" type="button" aria-label="play">▶</button>
            <input id="karaokeScrub" type="range" min="0" max="1" step="0.01" value="0" aria-label="playback position">
            <span class="karaoke-time" id="karaokeTime">0:00 / 0:00</span>
          </div>
          <div class="karaoke-layers">
            <label><input id="karaokePitchLayer" type="checkbox" checked><span data-consumer-copy="voiceMovement">声音走势</span></label>
            <label><input id="karaokePauseLayer" type="checkbox" checked><span data-consumer-copy="pauseLayer">停顿</span></label>
          </div>
        </div>
      </section>
'''
replace_once(score_panel, karaoke_panel, 'karaoke panel')

# 3) Public demo should stay product-facing; keep engineering replay/debug locally.
replace_once('      <section>\n        <div class="panel-head">\n          <h2 data-i18n="realtimeReplay">实时状态回放</h2>', '      <section class="debug-only-panel">\n        <div class="panel-head">\n          <h2 data-i18n="realtimeReplay">实时状态回放</h2>', 'debug realtime panel')
replace_once('<details class="advanced-panel">', '<details class="advanced-panel debug-only-panel">', 'debug advanced panel')

# 4) Consumer copy lives alongside, but does not replace, the original four-language system.
consumer_copy = r'''
    const consumerCopy = {
      "zh-CN": {
        karaokeReplay:"跟读回放", karaokeHint:"播放录音时，字幕、停顿和声音走势会同步。", karaokePlaceholder:"完成一次评价后，这里会显示你的发话。",
        pitchUnavailable:"这次没有足够的音高信息可供可视化；这不代表抑扬表现差。", voiceMovement:"声音走势", pauseLayer:"停顿",
        syncWord:"逐词同步", syncMora:"逐拍同步", syncMoraApprox:"逐拍同步·概算", syncSentence:"全文显示", syncUnavailable:"暂无同步",
        evidenceMeasured:"音声证据", evidenceBroad:"大致参考", evidenceNeutral:"参考值", evidenceUnavailable:"未测定",
        modeFixed:"范例跟读", modeFree:"自由说话", fixedHint:"按目标句练习", freeHint:"自然说日语即可",
        fixedDesc:"跟着范例练习同一句话。结果会结合目标句和参考音。", freeDesc:"直接说自然日语，不要求和某个目标句一致。",
        modeGroup:"练习方式",
        dimensions:{delivery_fluency:"流畅度", clarity:"明瞭度", mora_timing:"节奏", intonation:"抑扬"}
      },
      "zh-TW": {
        karaokeReplay:"跟讀回放", karaokeHint:"播放錄音時，字幕、停頓與聲音走勢會同步。", karaokePlaceholder:"完成一次檢查後，這裡會顯示你的發話。",
        pitchUnavailable:"這次沒有足夠的音高資訊可供視覺化；這不代表抑揚表現差。", voiceMovement:"聲音走勢", pauseLayer:"停頓",
        syncWord:"逐詞同步", syncMora:"逐拍同步", syncMoraApprox:"逐拍同步・概算", syncSentence:"全文顯示", syncUnavailable:"暫無同步",
        evidenceMeasured:"音聲證據", evidenceBroad:"大致參考", evidenceNeutral:"參考值", evidenceUnavailable:"未測定",
        modeFixed:"範例跟讀", modeFree:"自由說話", fixedHint:"依目標句練習", freeHint:"自然說日語即可",
        fixedDesc:"跟著範例練習同一句話。結果會結合目標句與參考音。", freeDesc:"直接說自然日語，不要求和某個目標句一致。",
        modeGroup:"練習方式",
        dimensions:{delivery_fluency:"流暢度", clarity:"明瞭度", mora_timing:"節奏", intonation:"抑揚"}
      },
      ja: {
        karaokeReplay:"発話リプレイ", karaokeHint:"録音を再生すると、字幕・間・声の動きが時間に合わせて表示されます。", karaokePlaceholder:"評価後、ここにあなたの発話を表示します。",
        pitchUnavailable:"今回は声の動きを表示できる十分な F0 情報がありません。低い抑揚評価を意味しません。", voiceMovement:"声の動き", pauseLayer:"間",
        syncWord:"単語同期", syncMora:"拍同期", syncMoraApprox:"拍同期・概算", syncSentence:"全文表示", syncUnavailable:"同期なし",
        evidenceMeasured:"音声から確認", evidenceBroad:"大まかな目安", evidenceNeutral:"参考値", evidenceUnavailable:"未測定",
        modeFixed:"お手本練習", modeFree:"自由に話す", fixedHint:"目標文で練習", freeHint:"自然な日本語をそのまま",
        fixedDesc:"お手本と同じ文を練習します。目標文と参照音声を使って確認します。", freeDesc:"目標文に合わせず、自然な日本語をそのまま話します。",
        modeGroup:"練習方法",
        dimensions:{delivery_fluency:"流暢さ", clarity:"明瞭さ", mora_timing:"リズム", intonation:"抑揚"}
      },
      en: {
        karaokeReplay:"Speech replay", karaokeHint:"Replay your recording to see transcript timing, pauses, and voice movement on one timeline.", karaokePlaceholder:"Your utterance will appear here after an evaluation.",
        pitchUnavailable:"There is not enough F0 evidence to visualize voice movement this time. This does not mean poor intonation.", voiceMovement:"Voice movement", pauseLayer:"Pauses",
        syncWord:"Word sync", syncMora:"Mora sync", syncMoraApprox:"Mora sync · approximate", syncSentence:"Full text", syncUnavailable:"No sync",
        evidenceMeasured:"Audio evidence", evidenceBroad:"Broad estimate", evidenceNeutral:"Reference value", evidenceUnavailable:"Unavailable",
        modeFixed:"Model practice", modeFree:"Free speaking", fixedHint:"Practice the target sentence", freeHint:"Speak natural Japanese",
        fixedDesc:"Practice the same sentence as the model. The target and reference audio are used for feedback.", freeDesc:"Speak natural Japanese without matching a fixed target sentence.",
        modeGroup:"Practice mode",
        dimensions:{delivery_fluency:"Fluency", clarity:"Clarity", mora_timing:"Rhythm", intonation:"Intonation"}
      }
    };

    function cc(key) {
      const pack = consumerCopy[state.locale] || consumerCopy["zh-CN"];
      return pack[key] ?? consumerCopy["zh-CN"][key] ?? key;
    }

    function ccDimension(key, fallback="") {
      const pack = consumerCopy[state.locale] || consumerCopy["zh-CN"];
      return pack.dimensions?.[key] || consumerCopy["zh-CN"].dimensions?.[key] || fallback || key;
    }

    function isPublicDemo() {
      return state.config?.server_label === "Public demo";
    }
'''
replace_once('    const modeCatalog = {', consumer_copy + '\n    const modeCatalog = {', 'consumer copy')

# 5) Public mode surface is now fixed-reference + direct broad free speech.
replace_once('    const publicUiModes = ["reference", "asr_pseudo_reference", "kanade_asr_voice_reference"];', '    const publicUiModes = ["reference", "transcript_assisted_light"];', 'public modes')

# 6) Locale switching also refreshes the incremental consumer copy.
replace_once('      document.querySelectorAll("[data-i18n]").forEach((node) => {\n        node.textContent = t(node.dataset.i18n);\n      });', '      document.querySelectorAll("[data-i18n]").forEach((node) => {\n        node.textContent = t(node.dataset.i18n);\n      });\n      document.querySelectorAll("[data-consumer-copy]").forEach((node) => {\n        node.textContent = cc(node.dataset.consumerCopy);\n      });', 'consumer locale refresh')

# 7) Keep local/debug mode breadth, but simplify the public dropdown and copy.
old_render_modes = '''      const configuredModes = Array.isArray(state.config?.available_modes) && state.config.available_modes.length\n        ? state.config.available_modes\n        : publicUiModes;\n      const modes = configuredModes.filter((mode) => publicUiModes.includes(mode));\n      const groups = ["core", "free", "diagnostic", "experimental"];\n      select.innerHTML = groups.map((group) => {\n        const groupModes = modes.filter((mode) => (modeCatalog[mode]?.group || "diagnostic") === group);\n        if (!groupModes.length) return "";\n        const options = groupModes\n          .map((mode) => `<option value="${mode}">${tMode(mode)}</option>`)\n          .join("");\n        return `<optgroup label="${tModeGroup(group)}">${options}</optgroup>`;\n      }).join("");\n      if (modes.includes(selected)) select.value = selected;'''
new_render_modes = '''      const configuredModes = Array.isArray(state.config?.available_modes) && state.config.available_modes.length\n        ? state.config.available_modes\n        : publicUiModes;\n      const modes = isPublicDemo()\n        ? configuredModes.filter((mode) => publicUiModes.includes(mode))\n        : configuredModes;\n      if (isPublicDemo()) {\n        select.innerHTML = modes.map((mode) => `<option value="${mode}">${mode === "reference" ? cc("modeFixed") : cc("modeFree")}</option>`).join("");\n      } else {\n        const groups = ["core", "free", "diagnostic", "experimental"];\n        select.innerHTML = groups.map((group) => {\n          const groupModes = modes.filter((mode) => (modeCatalog[mode]?.group || "diagnostic") === group);\n          if (!groupModes.length) return "";\n          const options = groupModes.map((mode) => `<option value="${mode}">${tMode(mode)}</option>`).join("");\n          return `<optgroup label="${tModeGroup(group)}">${options}</optgroup>`;\n        }).join("");\n      }\n      if (modes.includes(selected)) select.value = selected;\n      else if (modes.length) select.value = modes[0];'''
replace_once(old_render_modes, new_render_modes, 'renderModeOptions')

old_mode_desc = '''    function renderModeDescription(mode = currentMode()) {\n      const desc = tModeDescription(mode);\n      const group = modeCatalog[mode]?.group || "diagnostic";\n      const tone = modeCatalog[mode]?.tone || "";\n      $("modeDescription").innerHTML = `\n        <div class="mode-description-head">\n          <strong>${desc.title}</strong>\n          <span class="mode-tag ${tone}">${tModeGroup(group)}</span>\n        </div>\n        <div>${desc.body}</div>\n        <div class="mode-tags"><span class="mode-tag ${tone}">${desc.score}</span></div>\n      `;\n    }'''
new_mode_desc = '''    function renderModeDescription(mode = currentMode()) {\n      if (isPublicDemo() && publicUiModes.includes(mode)) {\n        const fixed = mode === "reference";\n        $("modeDescription").innerHTML = `\n          <div class="mode-description-head"><strong>${fixed ? cc("modeFixed") : cc("modeFree")}</strong><span class="mode-tag good">${cc("modeGroup")}</span></div>\n          <div>${fixed ? cc("fixedDesc") : cc("freeDesc")}</div>\n        `;\n        return;\n      }\n      const desc = tModeDescription(mode);\n      const group = modeCatalog[mode]?.group || "diagnostic";\n      const tone = modeCatalog[mode]?.tone || "";\n      $("modeDescription").innerHTML = `\n        <div class="mode-description-head">\n          <strong>${desc.title}</strong>\n          <span class="mode-tag ${tone}">${tModeGroup(group)}</span>\n        </div>\n        <div>${desc.body}</div>\n        <div class="mode-tags"><span class="mode-tag ${tone}">${desc.score}</span></div>\n      `;\n    }'''
replace_once(old_mode_desc, new_mode_desc, 'public mode description')

# 8) Public page should visibly be product-oriented; debug-only panels remain available locally.
replace_once('      state.config = config;\n      renderModeOptions();', '      state.config = config;\n      document.body.classList.toggle("public-mode", config.server_label === "Public demo");\n      renderModeOptions();', 'public body class')

# 9) Do not expose unverified H/L labels as a C-end mora truth display.
old_mora_html = '''      $("moraList").innerHTML = moras.length\n        ? moras.map((m, i) => `<span class="mora">${m}<span class="pitch">${pitch[i] || "?"}</span></span>`).join("")\n        : `<span class="pill">${t("noMoraTarget")}</span>`;'''
new_mora_html = '''      $("moraList").innerHTML = moras.length\n        ? moras.map((m, i) => `<span class="mora">${escapeHtml(m)}${isPublicDemo() ? "" : `<span class="pitch">${escapeHtml(pitch[i] || "?")}</span>`}</span>`).join("")\n        : `<span class="pill">${t("noMoraTarget")}</span>`;'''
replace_once(old_mora_html, new_mora_html, 'public mora H/L hide')

# 10) Public mode hint should not say "light fallback" or other implementation jargon.
replace_once('      $("modeHint").textContent = hints[mode] || mode;', '      $("modeHint").textContent = isPublicDemo() && publicUiModes.includes(mode)\n        ? (mode === "reference" ? cc("fixedHint") : cc("freeHint"))\n        : (hints[mode] || mode);', 'public mode hint')

# 11) Add the playback logic and render the canonical four dimensions with evidence state.
start = '    function renderScores(result, userFacing = null) {'
end = '    function renderReliability(result, userFacing = null) {'
replacement = r'''    function evidenceCopy(stateName) {
      if (stateName === "measured_proxy") return [cc("evidenceMeasured"), "measured"];
      if (stateName === "broad_proxy") return [cc("evidenceBroad"), "broad"];
      if (stateName === "neutral_prior") return [cc("evidenceNeutral"), "neutral"];
      return [cc("evidenceUnavailable"), ""];
    }

    function renderScores(result, userFacing = null, timeline = null) {
      const cards = [];
      const hasUserFacingDisplay = userFacing && userFacing.display_score !== undefined && userFacing.display_score !== null;
      const total = Number(hasUserFacingDisplay ? userFacing.display_score : (userFacing ? NaN : result.total_score));
      if (Number.isFinite(total) && result.details?.mode !== "reference_free_acoustic") {
        cards.push(`<div class="score total"><span>${t("overallPerformance")}</span><strong>${Math.round(total)}</strong><span>/ 100 · ${t("notStrictScore")}</span></div>`);
      } else if (userFacing) {
        cards.push(`<div class="score total"><span>${t("overallPerformance")}</span><strong>--</strong><span>${t("notStrictScore")}</span></div>`);
      }
      if (userFacing) {
        const status = userFacing.practice_check_result || "needs_attention";
        const reliability = userFacing.confidence_label || userFacing.reliability || "unknown";
        const timelineDimensions = Array.isArray(timeline?.dimensions) ? timeline.dimensions : [];
        const dimensions = timelineDimensions.length
          ? timelineDimensions
          : (Array.isArray(userFacing.score_dimensions) && userFacing.score_dimensions.length
            ? userFacing.score_dimensions
            : [
                { key:"delivery_fluency", value:userFacing.practice_completion_score, evidence_state:"broad_proxy" },
                { key:"clarity", value:userFacing.pronunciation_clarity_score, evidence_state:"broad_proxy" },
                { key:"mora_timing", value:userFacing.rhythm_fluency_score, evidence_state:"broad_proxy" },
                { key:"intonation", value:null, evidence_state:"unavailable" }
              ]);
        cards.push(...dimensions.map((item) => {
          const rawValue = item.score ?? item.value;
          const value = item.available === false ? null : Number(rawValue);
          const [evidenceText, evidenceClass] = evidenceCopy(item.evidence_state);
          const label = ccDimension(item.key, item.label || item.key);
          return `<div class="score"><span>${escapeHtml(label)}</span><strong>${Number.isFinite(value) ? Math.round(value) : "--"}</strong><span>${Number.isFinite(value) ? "/ 100" : t("notStrictScore")}</span><span class="score-evidence ${evidenceClass}">${escapeHtml(evidenceText)}</span></div>`;
        }));
        cards.push(`<div class="score"><span>${t("practiceResult")}</span><strong>${practiceResultLabel(status)}</strong><span>${t("reliabilitySimple")}：${reliabilityDisplay(reliability)}</span></div>`);
      } else {
        const keys = scoreKeys;
        cards.push(...keys.map((key) => {
          const label = tGroup("scoreLabels", key);
          const value = result[key] ?? 0;
          const className = key === "total_score" ? "score total" : "score";
          return `<div class="${className}"><span>${label}</span><strong>${value}</strong><span>/ 100</span></div>`;
        }));
      }
      $("scoreGrid").innerHTML = cards.join("");
    }

    function fmtPlaybackTime(sec) {
      sec = Math.max(0, Number(sec) || 0);
      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60);
      return `${m}:${String(s).padStart(2, "0")}`;
    }

    function karaokeSyncLabel(mode) {
      return ({word_timestamps:cc("syncWord"), mora_alignment:cc("syncMora"), mora_alignment_approximate:cc("syncMoraApprox"), sentence_progress_only:cc("syncSentence")}[mode] || cc("syncUnavailable"));
    }

    function renderKaraoke(payload) {
      const timeline = payload?.karaoke_timeline || {};
      const lyrics = $("karaokeLyrics");
      lyrics.innerHTML = "";
      lyrics.classList.remove("static");
      $("karaokeSyncBadge").textContent = karaokeSyncLabel(timeline.sync_mode);
      $("karaokeSyncNote").textContent = timeline.sync_note || cc("karaokeHint");
      const words = Array.isArray(timeline.words) ? timeline.words : [];
      const moras = Array.isArray(timeline.moras) ? timeline.moras : [];
      const rows = words.length ? words : moras;
      if (rows.length) {
        rows.forEach((row) => {
          const button = document.createElement("button");
          button.type = "button";
          button.className = `karaoke-token${row.asr_confidence === "low" ? " low-asr" : ""}`;
          button.textContent = row.text || "";
          button.dataset.start = row.start_sec;
          button.dataset.end = row.end_sec;
          button.title = words.length ? "ASR timestamp: playback alignment, not pronunciation correctness" : "Mora playback alignment, not pronunciation correctness";
          button.onclick = () => {
            const audio = $("lastRecordingAudio");
            if (!audio.src) return;
            audio.currentTime = Math.max(0, Number(row.start_sec) || 0);
            updateKaraokePlayback();
          };
          lyrics.appendChild(button);
        });
      } else {
        lyrics.classList.add("static");
        lyrics.textContent = timeline.transcript || cc("karaokePlaceholder");
      }
      const duration = Number(timeline.duration_sec) || Number($("lastRecordingAudio").duration) || 1;
      $("karaokeScrub").max = Math.max(0.01, duration);
      const pitchAvailable = Boolean(timeline.pitch?.available);
      $("karaokeEmpty").classList.toggle("show", !pitchAvailable);
      drawKaraokeTimeline();
      updateKaraokePlayback();
    }

    function drawKaraokeTimeline() {
      const canvas = $("karaokeCanvas");
      if (!canvas) return;
      const box = canvas.parentElement;
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, box.clientWidth);
      const h = Math.max(1, box.clientHeight);
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const timeline = state.lastPayload?.karaoke_timeline || {};
      const duration = Math.max(0.1, Number(timeline.duration_sec) || Number($("lastRecordingAudio").duration) || 1);
      const x = (sec) => Math.max(0, Math.min(w, (Number(sec) || 0) / duration * w));
      ctx.strokeStyle = "#e4e7ec";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
      if ($("karaokePauseLayer").checked) {
        ctx.fillStyle = "rgba(181, 71, 8, .08)";
        (timeline.pauses || []).forEach((p) => ctx.fillRect(x(p.start_sec), 0, Math.max(2, x(p.end_sec) - x(p.start_sec)), h));
      }
      if ($("karaokePitchLayer").checked && timeline.pitch?.available) {
        const userPoints = timeline.pitch.points || [];
        const refPoints = timeline.pitch.reference_points || [];
        const vals = userPoints.concat(refPoints).map((p) => Number(p.relative_semitone)).filter(Number.isFinite);
        const maxAbs = Math.max(3, ...vals.map((v) => Math.abs(v)));
        const draw = (points, color, dashed=false) => {
          let active = false;
          ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = dashed ? 1.6 : 2.3; ctx.setLineDash(dashed ? [6, 5] : []);
          points.forEach((p) => {
            const value = Number(p.relative_semitone);
            if (!Number.isFinite(value)) { active = false; return; }
            const px = x(p.t_sec), py = h / 2 - (value / maxAbs) * h * .34;
            if (!active) { ctx.moveTo(px, py); active = true; } else ctx.lineTo(px, py);
          });
          ctx.stroke(); ctx.setLineDash([]);
        };
        draw(userPoints, "#b45309", false);
        if (refPoints.length) draw(refPoints, "#087f8c", true);
      }
      const cursor = x($("lastRecordingAudio").currentTime || 0);
      ctx.strokeStyle = "rgba(24,34,48,.65)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cursor, 0); ctx.lineTo(cursor, h); ctx.stroke();
    }

    function updateKaraokePlayback() {
      const audio = $("lastRecordingAudio");
      const now = Number(audio.currentTime) || 0;
      const duration = Number(audio.duration) || Number(state.lastPayload?.karaoke_timeline?.duration_sec) || 0;
      $("karaokeScrub").value = Math.min(Number($("karaokeScrub").max) || duration || 1, now);
      $("karaokeTime").textContent = `${fmtPlaybackTime(now)} / ${fmtPlaybackTime(duration)}`;
      $("karaokePlayBtn").textContent = audio.paused ? "▶" : "Ⅱ";
      $("karaokeLyrics").querySelectorAll(".karaoke-token").forEach((token) => {
        const start = Number(token.dataset.start), end = Number(token.dataset.end);
        token.classList.toggle("active", now >= start && now < end);
        token.classList.toggle("past", now >= end);
      });
      drawKaraokeTimeline();
    }

    function karaokeAnimationLoop() {
      cancelAnimationFrame(state.karaokeRaf || 0);
      const tick = () => {
        updateKaraokePlayback();
        if (!$("lastRecordingAudio").paused) state.karaokeRaf = requestAnimationFrame(tick);
      };
      tick();
    }

    function bindKaraokeControls() {
      const audio = $("lastRecordingAudio");
      $("karaokePlayBtn").onclick = async () => {
        if (!audio.src) return;
        if (audio.paused) await audio.play(); else audio.pause();
      };
      $("karaokeScrub").oninput = () => {
        if (!audio.src) return;
        audio.currentTime = Number($("karaokeScrub").value) || 0;
        updateKaraokePlayback();
      };
      $("karaokePitchLayer").onchange = drawKaraokeTimeline;
      $("karaokePauseLayer").onchange = drawKaraokeTimeline;
      audio.addEventListener("play", karaokeAnimationLoop);
      audio.addEventListener("pause", updateKaraokePlayback);
      audio.addEventListener("ended", updateKaraokePlayback);
      audio.addEventListener("timeupdate", updateKaraokePlayback);
      audio.addEventListener("loadedmetadata", updateKaraokePlayback);
      window.addEventListener("resize", drawKaraokeTimeline);
    }

'''
replace_between(start, end, replacement, 'renderScores + karaoke')

# 12) Feed the new additive timeline into both score and playback UI.
replace_once('      renderScores(payload.result, payload.user_facing);\n      renderReliability(payload.result, payload.user_facing);\n      renderEndpointing(payload.result.endpointing || {});', '      renderScores(payload.result, payload.user_facing, payload.karaoke_timeline);\n      renderReliability(payload.result, payload.user_facing);\n      renderEndpointing(payload.result.endpointing || {});\n      renderKaraoke(payload);', 'render payload karaoke')

# 13) Bind playback controls after config exists.
replace_once('      renderConfig(config);\n      applyLocale();\n      $("languageSwitch").onclick', '      renderConfig(config);\n      applyLocale();\n      bindKaraokeControls();\n      $("languageSwitch").onclick', 'bind karaoke controls')

# 14) Public pitch chart becomes voice-movement comparison, not red/green H/L correctness.
replace_once('      const pad = { left: 42, right: 18, top: 18, bottom: 62 };', '      const showLexicalDebug = !isPublicDemo();\n      const pad = { left: 42, right: 18, top: 18, bottom: showLexicalDebug ? 62 : 34 };', 'pitch debug gate')
old_mora_draw = '''      moras.forEach((mora, i) => {\n        const x = xFor(i);\n        ctx.strokeStyle = "#e4e8ee";\n        ctx.beginPath();\n        ctx.moveTo(x, pad.top);\n        ctx.lineTo(x, pad.top + h);\n        ctx.stroke();\n        ctx.fillStyle = "#18202a";\n        ctx.textAlign = "center";\n        ctx.fillText(mora, x, pad.top + h + 22);\n        ctx.fillStyle = "#667085";\n        ctx.fillText(`${t("targetAbbr")}:${targetPitch[i] || "?"}`, x, pad.top + h + 38);\n        if (result) {\n          const obs = observed[i] || "?";\n          ctx.fillStyle = obs === "?" ? "#98a2b3" : obs === targetPitch[i] ? "#047857" : "#b42318";\n          ctx.fillText(`${t("observedAbbr")}:${obs}`, x, pad.top + h + 54);\n        }\n      });\n      ctx.textAlign = "left";'''
new_mora_draw = '''      moras.forEach((mora, i) => {\n        const x = xFor(i);\n        ctx.strokeStyle = "#e4e8ee";\n        ctx.beginPath();\n        ctx.moveTo(x, pad.top);\n        ctx.lineTo(x, pad.top + h);\n        ctx.stroke();\n        ctx.fillStyle = "#18202a";\n        ctx.textAlign = "center";\n        ctx.fillText(mora, x, pad.top + h + 22);\n        if (showLexicalDebug) {\n          ctx.fillStyle = "#667085";\n          ctx.fillText(`${t("targetAbbr")}:${targetPitch[i] || "?"}`, x, pad.top + h + 38);\n          if (result) {\n            const obs = observed[i] || "?";\n            ctx.fillStyle = obs === "?" ? "#98a2b3" : obs === targetPitch[i] ? "#047857" : "#b42318";\n            ctx.fillText(`${t("observedAbbr")}:${obs}`, x, pad.top + h + 54);\n          }\n        }\n      });\n      ctx.textAlign = "left";'''
replace_once(old_mora_draw, new_mora_draw, 'pitch lexical debug gate')

# 15) State owns animation id so replay does not leak rAF loops.
replace_once('      pendingAsrConfirmation: null,\n      kanadePollTimer: null', '      pendingAsrConfirmation: null,\n      karaokeRaf: 0,\n      kanadePollTimer: null', 'karaoke state')

p.write_text(s, encoding='utf-8')

# CI test list gets the v7 guard.
wf = Path('.github/workflows/baseline-evolution-light-tests.yml')
ws = wf.read_text(encoding='utf-8')
needle = '          tests/test_consumer_karaoke_ui.py \\\n'
if needle not in ws:
    raise SystemExit('missing CI consumer test anchor')
ws = ws.replace(needle, needle + '          tests/test_hf_space_karaoke_v7.py \\\n', 1)
wf.write_text(ws, encoding='utf-8')
