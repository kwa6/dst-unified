from dst.analysis import eval_audit


def test_split_context_turns_uppercase_tags():
    ctx = "Turn 0 [USER]: hi\nTurn 1 [SYSTEM]: hello"
    turns = eval_audit.split_context_turns(ctx)
    assert len(turns) == 2
    assert turns[0]["speaker"] == "user"
    assert turns[1]["speaker"] == "system"


def test_split_context_turns_lowercase_tags():
    ctx = "Turn 0 [user]: hi\nTurn 1 [system]: hello"
    turns = eval_audit.split_context_turns(ctx)
    assert len(turns) == 2
    assert turns[0]["speaker"] == "user"
    assert turns[1]["speaker"] == "system"


def test_split_context_turns_mixed_case_tags():
    ctx = "Turn 2 [UsEr]: hi"
    turns = eval_audit.split_context_turns(ctx)
    assert len(turns) == 1
    assert turns[0]["speaker"] == "user"


def test_split_context_turns_legacy_format():
    ctx = "Turn 3: legacy line"
    turns = eval_audit.split_context_turns(ctx)
    assert len(turns) == 1
    assert turns[0]["speaker"] is None
    assert turns[0]["text"] == "legacy line"


def test_extract_user_context_parity_fallback():
    ctx = "Turn 0: user line\nTurn 1: system line\nTurn 2: user again"
    user_ctx = eval_audit.extract_user_context(ctx)
    assert user_ctx == "user line\nuser again"


def test_extract_user_context_ignores_parity_when_tags_exist():
    ctx = "Turn 0 [user]: tagged user\nTurn 1: legacy system"
    user_ctx = eval_audit.extract_user_context(ctx)
    assert user_ctx == "tagged user"
