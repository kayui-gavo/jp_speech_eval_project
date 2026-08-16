from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one match in {path}, got {count}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Preserve the legacy note phrase for compatibility while extending its meaning
# to low-precision reference timing.
replace_once(
    "ver1.3/src/jp_speech_eval/consumer_dimension_policy.py",
    'alignment_note = " local alignment or reference timing is approximate, so target-local timing/F0 evidence was not used as if it were precise." if alignment_fallback else ""',
    'alignment_note = " local alignment fell back or reference timing is approximate, so target-local timing/F0 evidence was not used as if it were precise." if alignment_fallback else ""',
)

# The previous long-turn fixture was intentionally repetitive and correctly
# triggered the repetition guard. Use a genuinely varied natural Japanese turn
# that exceeds the old pseudo-reference 80-character ceiling.
replace_once(
    "ver1.3/tests/test_scoring_integrity_v8.py",
    '    assert check_free_speech_transcript_sanity("今日はとてもいい天気ですね" * 8).ok is True\n',
    '    long_turn = "今日は大学で研究の打ち合わせをして、そのあと友達と昼ご飯を食べました。午後は図書館で資料を探してから、研究室に戻って実験の結果を整理しました。帰る前に先生へメールも送り、明日の発表で説明する内容をもう一度確認しました。"\n    assert len(long_turn) > 80\n    assert check_free_speech_transcript_sanity(long_turn).ok is True\n',
)
