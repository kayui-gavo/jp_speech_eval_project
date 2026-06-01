# User-facing policy examples

## fixed-reference normal pass

- raw condition: `{}`
- status: pass
- practice_score: {'value': 93, 'label': '良好', 'explanation': 'このスコアは，今回の録音について，内容・リズム・流暢さなどをもとにした練習用の目安です。発音能力そのものを厳密に評価するものではありません。 fixed-reference mode: verified target に基づく練習確認です。'}
- summary_text: 全体としてよくできています。
- primary_suggestion_text: None
- suppressed_reasons: ['debug_only_by_profile']

## near-boundary long vowel accepted

- raw condition: `{'special_mora_threshold_profile': 'v2_limited_candidate', 'enable_user_facing_calibrated_special_mora': True}`
- status: pass
- practice_score: {'value': 93, 'label': '良好', 'explanation': 'このスコアは，今回の録音について，内容・リズム・流暢さなどをもとにした練習用の目安です。発音能力そのものを厳密に評価するものではありません。 fixed-reference mode: verified target に基づく練習確認です。'}
- summary_text: 全体としてよくできています。
- primary_suggestion_text: None
- suppressed_reasons: ['no_correction_needed']

## clear long vowel too_short flag off

- raw condition: `{'special_mora_threshold_profile': 'v2_limited_candidate'}`
- status: pass
- practice_score: {'value': 93, 'label': '良好', 'explanation': 'このスコアは，今回の録音について，内容・リズム・流暢さなどをもとにした練習用の目安です。発音能力そのものを厳密に評価するものではありません。 fixed-reference mode: verified target に基づく練習確認です。'}
- summary_text: 全体としてよくできています。
- primary_suggestion_text: None
- suppressed_reasons: ['shadow_mode_user_facing_disabled']

## clear long vowel too_short flag on gentle only

- raw condition: `{'special_mora_threshold_profile': 'v2_limited_candidate', 'enable_user_facing_calibrated_special_mora': True}`
- status: practice_suggestion
- practice_score: {'value': 93, 'label': '良好', 'explanation': 'このスコアは，今回の録音について，内容・リズム・流暢さなどをもとにした練習用の目安です。発音能力そのものを厳密に評価するものではありません。 fixed-reference mode: verified target に基づく練習確認です。'}
- summary_text: 全体としては問題ありません。より自然にするための練習ポイントがあります。
- primary_suggestion_text: 全体としては問題ありません。より自然にするなら，「ー」を少し長めに意識するとよいです。
- suppressed_reasons: ['no_correction_needed']

## weak-reference ASR text not confirmed

- raw condition: `{'mode': 'asr_pseudo_reference'}`
- status: debug_only
- practice_score: {'value': None, 'label': '判定できません', 'explanation': 'このスコアは，今回の録音について，内容・リズム・流暢さなどをもとにした練習用の目安です。発音能力そのものを厳密に評価するものではありません。 認識された文をもとにした参考判定です。厳密な発音評価ではありません。'}
- summary_text: このモードでは，厳密な発音判定は行わず，練習用の参考として表示しています。
- primary_suggestion_text: None
- suppressed_reasons: ['debug_only_by_profile']

## ASR+Kanade playback only

- raw condition: `{'mode': 'kanade_asr_voice_reference'}`
- status: debug_only
- practice_score: {'value': None, 'label': '判定できません', 'explanation': 'このスコアは，今回の録音について，内容・リズム・流暢さなどをもとにした練習用の目安です。発音能力そのものを厳密に評価するものではありません。 これはあなたの声に近い参考音です。声の似ている度合いは採点していません。'}
- summary_text: このモードでは，厳密な発音判定は行わず，練習用の参考として表示しています。
- primary_suggestion_text: None
- suppressed_reasons: ['debug_only_by_profile']

## poor recording quality retry

- raw condition: `{}`
- status: retry
- practice_score: {'value': None, 'label': '録音を確認', 'explanation': 'このスコアは，今回の録音について，内容・リズム・流暢さなどをもとにした練習用の目安です。発音能力そのものを厳密に評価するものではありません。 fixed-reference mode: 信頼できる内容・リズム・流暢さを中心に確認します。'}
- summary_text: 録音が短すぎるか，音声がはっきり取れていません。もう一度録音してください。
- primary_suggestion_text: None
- suppressed_reasons: ['recording_quality_bad', 'debug_only_by_profile']

## low F0 coverage suppresses prosody

- raw condition: `{}`
- status: pass
- practice_score: {'value': 94, 'label': '良好', 'explanation': 'このスコアは，今回の録音について，内容・リズム・流暢さなどをもとにした練習用の目安です。発音能力そのものを厳密に評価するものではありません。 fixed-reference mode: 信頼できる内容・リズム・流暢さを中心に確認します。'}
- summary_text: 全体としてよくできています。
- primary_suggestion_text: None
- suppressed_reasons: ['low_f0_coverage', 'debug_only_by_profile']

## sokuon issue blocked

- raw condition: `{'special_mora_threshold_profile': 'v2_limited_candidate', 'enable_user_facing_calibrated_special_mora': True}`
- status: pass
- practice_score: {'value': 94, 'label': '良好', 'explanation': 'このスコアは，今回の録音について，内容・リズム・流暢さなどをもとにした練習用の目安です。発音能力そのものを厳密に評価するものではありません。 fixed-reference mode: 信頼できる内容・リズム・流暢さを中心に確認します。'}
- summary_text: 全体としてよくできています。
- primary_suggestion_text: None
- suppressed_reasons: ['short_utterance', 'blocked_by_profile']

## yoon duration debug-only

- raw condition: `{'special_mora_threshold_profile': 'v2_limited_candidate', 'enable_user_facing_calibrated_special_mora': True}`
- status: pass
- practice_score: {'value': 94, 'label': '良好', 'explanation': 'このスコアは，今回の録音について，内容・リズム・流暢さなどをもとにした練習用の目安です。発音能力そのものを厳密に評価するものではありません。 fixed-reference mode: 信頼できる内容・リズム・流暢さを中心に確認します。'}
- summary_text: 全体としてよくできています。
- primary_suggestion_text: None
- suppressed_reasons: ['short_utterance', 'debug_only_by_profile']
