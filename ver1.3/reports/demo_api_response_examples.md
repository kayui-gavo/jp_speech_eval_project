# Demo API response examples

## fixed_reference_pass

- input condition: `{'mode': 'reference', 'target_id': 'ramen_kudasai'}`
- user_facing.status: `pass`
- user_facing.practice_score: `{'value': 93, 'label': '良好', 'explanation': 'このスコアは，今回の録音について，内容・リズム・流暢さなどをもとにした練習用の目安です。発音能力そのものを厳密に評価するものではありません。 fixed-reference mode: verified target に基づく練習確認です。'}`
- user_facing.summary_text: 全体としてよくできています。
- user_facing.primary_suggestion_text: None
- debug summary: `{'weak_reference': False, 'demo_only': False, 'alignment_confidence': 0.9, 'f0_voiced_coverage': 0.9, 'special_mora_decision_count': 2, 'allow_pitch_feedback': True, 'exclude_from_pronunciation_score': False}`
- why safe: Learner fields contain only status, practice score, summary, and mode notice.

## fixed_reference_practice_suggestion

- input condition: `{'mode': 'reference', 'special_mora_flag': 'on'}`
- user_facing.status: `practice_suggestion`
- user_facing.practice_score: `{'value': 93, 'label': '良好', 'explanation': 'このスコアは，今回の録音について，内容・リズム・流暢さなどをもとにした練習用の目安です。発音能力そのものを厳密に評価するものではありません。 fixed-reference mode: verified target に基づく練習確認です。'}`
- user_facing.summary_text: 全体としては問題ありません。より自然にするための練習ポイントがあります。
- user_facing.primary_suggestion_text: 全体としては問題ありません。より自然にするなら，「ー」を少し長めに意識するとよいです。
- debug summary: `{'weak_reference': False, 'demo_only': False, 'alignment_confidence': 0.9, 'f0_voiced_coverage': 0.9, 'special_mora_decision_count': 2, 'allow_pitch_feedback': True, 'exclude_from_pronunciation_score': False}`
- why safe: Special mora appears only as a gentle practice point, not a correctness penalty.

## fixed_reference_retry_poor_recording

- input condition: `{'mode': 'reference', 'recording_quality': 'poor'}`
- user_facing.status: `retry`
- user_facing.practice_score: `{'value': None, 'label': '録音を確認', 'explanation': 'このスコアは，今回の録音について，内容・リズム・流暢さなどをもとにした練習用の目安です。発音能力そのものを厳密に評価するものではありません。 fixed-reference mode: 信頼できる内容・リズム・流暢さを中心に確認します。'}`
- user_facing.summary_text: 録音が短すぎるか，音声がはっきり取れていません。もう一度録音してください。
- user_facing.primary_suggestion_text: None
- debug summary: `{'weak_reference': False, 'demo_only': False, 'alignment_confidence': 0.9, 'f0_voiced_coverage': 0.9, 'special_mora_decision_count': 2, 'allow_pitch_feedback': True, 'exclude_from_pronunciation_score': False}`
- why safe: Retry is framed as recording quality, not pronunciation failure.

## weak_reference_unconfirmed_asr

- input condition: `{'mode': 'asr_pseudo_reference', 'confirmed': False}`
- user_facing.status: `debug_only`
- user_facing.practice_score: `{'value': None, 'label': '判定できません', 'explanation': 'このスコアは，今回の録音について，内容・リズム・流暢さなどをもとにした練習用の目安です。発音能力そのものを厳密に評価するものではありません。 認識された文をもとにした参考判定です。厳密な発音評価ではありません。'}`
- user_facing.summary_text: このモードでは，厳密な発音判定は行わず，練習用の参考として表示しています。
- user_facing.primary_suggestion_text: None
- debug summary: `{'weak_reference': True, 'demo_only': False, 'alignment_confidence': 0.9, 'f0_voiced_coverage': 0.9, 'special_mora_decision_count': 2, 'allow_pitch_feedback': False, 'exclude_from_pronunciation_score': False}`
- why safe: Unconfirmed ASR stays debug-only and has no practice score value.

## weak_reference_confirmed

- input condition: `{'mode': 'asr_confirmed_weak_reference', 'confirmed': True}`
- user_facing.status: `pass`
- user_facing.practice_score: `{'value': 94, 'label': '良好', 'explanation': 'このスコアは，今回の録音について，内容・リズム・流暢さなどをもとにした練習用の目安です。発音能力そのものを厳密に評価するものではありません。 認識された文をもとにした参考判定です。厳密な発音評価ではありません。'}`
- user_facing.summary_text: 全体としてよくできています。
- user_facing.primary_suggestion_text: None
- debug summary: `{'weak_reference': True, 'demo_only': False, 'alignment_confidence': 0.9, 'f0_voiced_coverage': 0.9, 'special_mora_decision_count': 2, 'allow_pitch_feedback': False, 'exclude_from_pronunciation_score': False}`
- why safe: Confirmed weak-reference keeps a weak-reference notice visible.

## asr_kanade_playback_notice

- input condition: `{'mode': 'kanade_asr_voice_reference', 'kanade_audio': 'mocked'}`
- user_facing.status: `debug_only`
- user_facing.practice_score: `{'value': None, 'label': '判定できません', 'explanation': 'このスコアは，今回の録音について，内容・リズム・流暢さなどをもとにした練習用の目安です。発音能力そのものを厳密に評価するものではありません。 これはあなたの声に近い参考音です。声の似ている度合いは採点していません。'}`
- user_facing.summary_text: このモードでは，厳密な発音判定は行わず，練習用の参考として表示しています。
- user_facing.primary_suggestion_text: None
- debug summary: `{'weak_reference': True, 'demo_only': True, 'alignment_confidence': 0.9, 'f0_voiced_coverage': 0.9, 'special_mora_decision_count': 2, 'allow_pitch_feedback': False, 'exclude_from_pronunciation_score': True}`
- why safe: Kanade notice says voice similarity is not scored; correctness scoring is excluded.

## special_mora_suppressed

- input condition: `{'mode': 'reference', 'special_mora_flag': 'off'}`
- user_facing.status: `pass`
- user_facing.practice_score: `{'value': 93, 'label': '良好', 'explanation': 'このスコアは，今回の録音について，内容・リズム・流暢さなどをもとにした練習用の目安です。発音能力そのものを厳密に評価するものではありません。 fixed-reference mode: verified target に基づく練習確認です。'}`
- user_facing.summary_text: 全体としてよくできています。
- user_facing.primary_suggestion_text: None
- debug summary: `{'weak_reference': False, 'demo_only': False, 'alignment_confidence': 0.9, 'f0_voiced_coverage': 0.9, 'special_mora_decision_count': 2, 'allow_pitch_feedback': True, 'exclude_from_pronunciation_score': False}`
- why safe: Near-boundary special-mora evidence remains hidden from learner fields.

## pitch_suppressed_unverified_target

- input condition: `{'mode': 'reference', 'target_id': 'coffee_kudasai', 'verified_level': 'auto_pyopenjtalk'}`
- user_facing.status: `pass`
- user_facing.practice_score: `{'value': 94, 'label': '良好', 'explanation': 'このスコアは，今回の録音について，内容・リズム・流暢さなどをもとにした練習用の目安です。発音能力そのものを厳密に評価するものではありません。 fixed-reference mode: 信頼できる内容・リズム・流暢さを中心に確認します。'}`
- user_facing.summary_text: 全体としてよくできています。
- user_facing.primary_suggestion_text: None
- debug summary: `{'weak_reference': False, 'demo_only': False, 'alignment_confidence': 0.9, 'f0_voiced_coverage': 0.9, 'special_mora_decision_count': 2, 'allow_pitch_feedback': False, 'exclude_from_pronunciation_score': False}`
- why safe: Pitch feedback is suppressed because the target is not OJAD/manual verified.
