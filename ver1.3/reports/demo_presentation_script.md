# Demo presentation script

## 一句话介绍

这是一个面向 C 端日语学习者的口语/发音练习 demo，重点是 fixed-reference 朗读评分和 ASR+Kanade 个性化参考音。

日文版：

> 日本語学習者向けに，固定文の朗読練習と，自分の声に近い参考音を使った会話練習を試せる音声評価デモです。

## 演示流程 1: fixed-reference

推荐说法：

> まず一番信頼しやすい fixed-reference mode を見せます。  
> 目標文は先に決まっていて，ユーザーがそれを読みます。  
> 画面には raw score ではなく，user-facing の練習結果だけを出します。

流程：

1. 选择「ラーメンをください」。
2. 播放或确认 reference。
3. 用户朗读。
4. 显示 `user_facing.summary_text` 和最多一个练习建议。

强调：

- fixed-reference 是当前最可靠的模式。
- `practice_score` 是练习参考，不是科学发音能力评分。
- debug 指标可以给开发者看，但不直接给普通用户。

## 演示流程 2: ASR-confirmed weak-reference

推荐说法：

> 次は自由に話した時の流れです。  
> ASR の結果をそのまま正解にせず，まずユーザーに確認してもらいます。  
> 確認された文だけを弱い reference として使います。

流程：

1. 用户自由说一句。
2. ASR 给候选文本。
3. 用户确认或修改。
4. 系统生成 pseudo-reference。
5. 显示 conservative weak-reference feedback。

强调：

- 未确认 ASR 不进入严肃评分。
- weak-reference 不能说成严格发音评价。
- pitch / 特殊拍强纠错默认压制。

## 演示流程 3: ASR+Kanade

推荐说法：

> 最後は产品上比较有趣的 ASR+Kanade mode です。  
> ユーザーが確認した文から参考音を作り，Kanade でユーザーの声に近い音色に変換します。  
> ただし，Kanade は再生用で，採点には使いません。

流程：

1. 用户说话。
2. ASR 后用户确认文本。
3. 生成 TTS pseudo-reference。
4. Kanade 转成接近用户声线的参考音。
5. 播放 personalized reference。
6. 评价仍基于非 Kanade reference 和可靠 features。

强调：

- Kanade 只用于播放参考音。
- 不计算“像 Kanade 音声”的分数。
- 声音相似度不是发音正确性。

## 不要这样说

- これは科学的に発音能力を測れます。
- Kanade に似ているほど発音が正しいです。
- ASR が出した文をそのまま正解として採点します。
- ネイティブ度を測っています。

## 推荐收尾

> 今の段階では，正確な能力測定よりも，安全に練習フィードバックを返すことを優先しています。  
> 次の課題は，母語話者・教師による小規模確認を入れて，どのフィードバックを本当にユーザーに出してよいかを決めることです。
