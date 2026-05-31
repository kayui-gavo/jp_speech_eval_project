# 特殊拍人工复核指南 v2

目标：区分“听得出长短变化”和“真的应该给学习者反馈的问题”。

## 核心原则
- 听得出变化，不等于发音错误。
- 母语者语流中的长音、拨音会自然伸缩。
- near-boundary 和 mild variation 默认通过，不扣分，不强提示。
- too_long 默认 debug-only，不进入用户端纠错。
- JANON 只看学习者趋势，不当 ground truth。
- counterfactual 只测规则敏感度，不是真人验证。

## 字段怎么填
- intelligibility: 是否听得懂。clear / mostly_clear / hard_to_understand / unsure。
- naturalness: 听起来是否自然。natural / slightly_unnatural / unnatural / unsure。
- communication_impact: 是否影响交流。none / minor / clear / severe / unsure。
- variation_type: natural_variation / acceptable_variation / possible_issue / likely_error / alignment_uncertain / unsure。
- audible_variation: 是否听得出长短变化。yes / no / unsure。
- should_feedback: 是否值得给用户提示。yes / no / unsure。
- feedback_strength: none / gentle_tip / practice_focus / correction。
- wording_ok: 当前候选文案是否可以接受。
- alignment_issue: 对齐是否疑似有问题。
- audio_quality_issue: 原音质量是否影响判断。

## 上线规则
- natural_variation / acceptable_variation 不应扣分。
- communication_impact none/minor 不应强反馈。
- 只有 clear/severe + should_feedback=yes 才可能进入用户端提示候选。
- C 端当前最多使用 gentle_tip / practice_focus，不使用 correction。
