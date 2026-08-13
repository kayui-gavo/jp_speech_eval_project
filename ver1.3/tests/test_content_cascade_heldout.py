from jp_speech_eval.content_cascade import split_rows_by_target_sentence


def test_target_sentence_split_is_disjoint_and_preserves_all_rows():
    rows = [
        {"target_sentence_id": "1", "sample_id": "a"},
        {"target_sentence_id": "2", "sample_id": "b"},
        {"target_sentence_id": "11", "sample_id": "c"},
        {"target_sentence_id": "12", "sample_id": "d"},
    ]
    development, held_out = split_rows_by_target_sentence(rows, {"1", "2"})
    assert [row["sample_id"] for row in development] == ["a", "b"]
    assert [row["sample_id"] for row in held_out] == ["c", "d"]
    assert {row["target_sentence_id"] for row in development}.isdisjoint(
        {row["target_sentence_id"] for row in held_out}
    )
