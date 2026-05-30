# 日语口语评价 core 工程审计

本报告记录当前仓库的真实入口、模块边界和下一步校准重点。目标是把 demo 收束为 native-calibrated、evidence-gated、reference-aware 的日语口语评价 core。

## 1. 真实入口与主要模块

### Demo / UI

- `ver1.3/scripts/debug_ui.py`
  - 本地和 Space demo 的主要 HTTP 入口。
  - 提供 reference、录音上传、ASR 确认、confirmed-ASR evaluation、reference wav 等 API。
  - ASR 模式和 Kanade-ASR 模式已经要求用户确认文本。

- `ver1.3/debug_ui/index.html`
  - 前端页面。
  - 负责模式选择、录音上传、ASR 候选确认、综合表现、可信度和反馈展示。

### fixed-reference pipeline

- `ver1.3/src/jp_speech_eval/evaluator.py`
  - 当前核心评价函数 `evaluate_utterance(...)`。
  - 做 audio loading、VAD、F0、content match、alignment、pronunciation/prosody/fluency/tone scoring、reliability metrics。
  - 目前 pronunciation 是 mora timing / special mora / alignment proxy，不是 phoneme-level correctness。

- `ver1.3/src/jp_speech_eval/sentence_cache.py`
  - 生成和读取 sentence cache。
  - 包含 target text、kana、mora、reference audio、reference feature。

### ASR / weak reference / Kanade

- `ver1.3/src/jp_speech_eval/asr.py`
  - ASR wrapper。
  - 优先 `faster-whisper`，默认 `small`，CPU int8。
  - 可 fallback 到 `openai-whisper`。

- `ver1.3/src/jp_speech_eval/asr_confirmation.py`
  - ASR candidate -> user confirmation / edit -> weak target。
  - ASR raw text 不直接进入 scoring。

- `ver1.3/src/jp_speech_eval/eval_modes.py`
  - `evaluate_asr_confirmed_weak_reference(...)`
  - `evaluate_kanade_asr_confirmed_voice_reference(...)`
  - Kanade 只生成 playback reference，不参与 pronunciation correctness。

- `ver1.3/src/jp_speech_eval/kanade_reference.py`
  - Kanade voice-conditioned playback reference。
  - 当前应保持 demo-only。

### TTS reference

- `ver1.3/src/jp_speech_eval/tts_backends.py`
  - `pyopenjtalk`
  - `google`
  - `aivis_http`
  - `voicevox_http`

- `ver1.3/src/jp_speech_eval/tts_adapter.py`
  - provider-agnostic TTS adapter 和 provider config validation。

TTS reference 全部应称为 pseudo-reference，除非未来加入人工验证过的 native reference。

### Text frontend / phonology

- `ver1.3/src/jp_speech_eval/text_frontend.py`
  - pyopenjtalk text frontend。
  - 生成 kana、mora、pitch target、accent phrase。

- `ver1.3/src/jp_speech_eval/phonology.py`
  - mora sequence 和特殊拍分类。

- `ver1.3/src/jp_speech_eval/target_specs.py`
  - TargetSpec schema。

- `ver1.3/src/jp_speech_eval/verified_targets.py`
  - verified target loading / lookup。

### Alignment / acoustic features / special mora

- `ver1.3/src/jp_speech_eval/alignment.py`
  - MFCC-DTW / cached-DTW / fallback alignment。
  - 当前没有真正 MFA backend。

- `ver1.3/src/jp_speech_eval/audio_features.py`
  - audio loading、F0、energy、MFCC 等基础特征。

- `ver1.3/src/jp_speech_eval/mora_evidence.py`
  - per-mora evidence，包括 boundary confidence、energy coverage、F0 coverage。

- `ver1.3/src/jp_speech_eval/special_mora_scorer.py`
  - 当前 special mora feedback。
  - 阈值仍是 heuristic，需要 JVS native calibration。

### Scoring / feedback / gate

- `ver1.3/src/jp_speech_eval/scoring.py`
  - pronunciation/prosody/fluency/tone 评分逻辑。

- `ver1.3/src/jp_speech_eval/scoring_policy.py`
  - 根据 mode、verified_level、weak_reference、Kanade demo-only 等决定允许哪些反馈。

- `ver1.3/src/jp_speech_eval/reliability_gate.py`
  - 用户反馈总闸门。
  - 控制 recording bad、content mismatch、alignment low、F0 low、short utterance、weak reference 等情况。

- `ver1.3/src/jp_speech_eval/feedback_renderer.py`
  - 将内部指标转换为 user-facing result。
  - display score 和 debug raw score 分开。

### 数据和诊断

- `JVS/`
  - JVS native corpus，100 speakers，约 14997 wav。
  - 用于 native false alarm / threshold sanity check。

- `JANON/`
  - L2 Japanese speech，约 8853 audio files。
  - 用于 learner trend audit，不是 ground truth。

- `ver1.3/data/prosody_minimal_pairs.json`
  - pitch accent minimal pair set。
  - 当前主要用于 Prosodic ABX diagnostic。

- `ver1.3/data/eval_test_cases.json`
  - 本次新增的最小工程测试目标集合。

- `ver1.3/scripts/run_calibration_snapshot.py`
  - 本次新增的轻量 JVS/JANON 校准快照脚本。

## 2. 正式评价维度建议

### D0 Semantic / Task Success

判断用户是否完成剧情 local goal。它不属于 pronunciation score。

当前状态：只在 ASR / confirmed text 层面有基础能力，还没有完整 story goal judge。

### D1 Mora Completeness / Mora Clarity

判断目标 mora 是否大体被说出来。

当前可用指标：

- mora count
- alignment confidence
- mora duration
- per-mora evidence
- MFCC-DTW / cached-DTW

建议命名：`mora_clarity_score`，不要叫 phoneme correctness。

### D2 Special Mora Accuracy

判断長音・促音・撥音・拗音是否成立。

当前状态：

- 已有特殊拍分类和 feedback。
- 阈值仍是 heuristic。
- 本维度最适合优先做 native-calibrated threshold。

### D3 Rhythm / Tempo

判断节奏、语速、停顿是否自然。

当前状态：

- 已有 speech rate、pause、mora duration CV。
- 小样本 JVS 显示 fluency/rhythm proxy 对 native 偏低，需要校准。

### D4 Pitch Accent

判断 accent phrase 内 H/L pattern 和下降位置。

当前状态：

- 有 F0、HL pattern、OpenJTalk target、TTS reference。
- 但可信度不足，必须强 gate。
- 只有 verified target + alignment high + F0 coverage high 时才适合用户端强反馈。

### D5 Phrase / Final Intonation

判断句尾语调和短语整体抑扬。

当前状态：

- 比 D4 更适合先做 C 端轻量反馈。
- 应从 prosody_score 中拆出来，避免和 pitch accent 混在一起。

### D6 Fluency

判断说话是否顺。

当前状态：

- 已有 speech rate / pause score。
- 需要用 JVS 校准自然范围，不能只靠工程阈值。

### D7 Reliability / Confidence

判断本次结果能不能信。

当前状态：

- 已有 reliability gate。
- 还应补充 reference trust level / target verified level / ASR confirmed status 的更细标记。

## 3. 当前最重要的缺陷

1. `prosody_score` 仍然混合了 pitch accent、phrase intonation、F0 debug，解释性不足。
2. `fluency_score` 小样本 JVS native mean 只有 79.9，说明语速/停顿阈值可能误伤自然朗读。
3. `special_mora_scorer.py` 使用 hand-tuned ratio，需要 JVS native distribution 改成分位数阈值。
4. 当前没有 MFA / phoneme-level forced alignment，无法做真正 phone correctness。
5. equal fallback 时必须继续禁止细粒度 mora/special mora/pitch feedback。
6. TTS reference 仍然是 pseudo-reference；Google/Aivis/pyopenjtalk 都不能叫 ground truth。
7. JANON 只能做 L2 trend audit，不能证明评分正确。
8. 缺少 teacher/native listener rating，因此 display score 仍是 product proxy。

## 4. 本次新增校准快照结果

命令：

```bash
python scripts/run_calibration_snapshot.py \
  --jvs-speakers 2 \
  --jvs-utterances-per-speaker 5 \
  --janon-limit 10
```

输出：

- `ver1.3/results/calibration/jvs_native_metrics.csv`
- `ver1.3/results/calibration/jvs_score_distribution.csv`
- `ver1.3/results/calibration/jvs_false_alarm_by_feature.csv`
- `ver1.3/results/calibration/janon_l2_metrics.csv`
- `ver1.3/reports/jvs_native_sanity_check.md`
- `ver1.3/reports/janon_l2_trend_report.md`

JVS 小样本结果：

- n = 10
- display mean = 89.7
- pronunciation mean = 100.0
- fluency/rhythm proxy mean = 79.9
- retry rate = 0.0
- unscorable rate = 0.0
- alignment fallback rate = 0.0
- special mora false alarm proxy = 0.0

解释：

- 当前不会明显误伤 JVS native 的 pronunciation。
- 但 rhythm/fluency proxy 没过 85 验收线，应优先校准。

JANON 小样本结果：

- n = 10
- display mean = 83.0
- pronunciation mean = 84.0
- fluency/rhythm proxy mean = 90.7
- retry rate = 0.0
- unscorable rate = 0.0
- alignment fallback rate = 0.0
- special mora false alarm proxy = 0.25

解释：

- JANON 结果只能说明 L2 trend，不能当正确标签。
- 当前 special mora proxy 对 L2 样本有明显差异信号，但是否正确需要人工/teacher rating。

## 5. 下一步最小建议

1. 先扩大 JVS calibration 到 100-300 条，按 sentence / isolated 分开。
2. 用 JVS 分位数重新定义 speech rate、avg mora duration、mora duration CV 的 native range。
3. 把 `fluency_score` 拆成 rhythm/tempo 和 delivery fluency，避免自然朗读被慢速惩罚。
4. 对 special mora 做 native false alarm 约束，目标是 native false alarm <= 5%。
5. 把 `prosody_score` 拆成：
   - `pitch_accent_score`：verified target + high evidence 才用户可见。
   - `phrase_intonation_score`：更早进入轻量反馈。
   - `f0_debug_metrics`：debug only。
6. 暂时不要做 phoneme-level correctness，除非接入 MFA 或 PPG。
7. 剧情模式 route decision 应主要看 D0 semantic + D7 reliability + D6 fluency，发音细节只作为轻量反馈，不应轻易 hard fail。
