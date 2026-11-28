from rag_utils import correct_query


def test_correct_query_corrects_typo():
    result = correct_query("documnt", ["document", "chatbot", "python"])
    assert result == "document"


def test_correct_query_keeps_short_words():
    result = correct_query("AI is good", ["artificial", "intelligence"])
    assert result == "AI is good"


def test_correct_query_with_empty_document_words():
    result = correct_query("chatbot project", [])
    assert result == "chatbot project"


def test_correct_query_keeps_unrelated_word():
    result = correct_query("elephant", ["document", "chatbot"])
    assert result == "elephant"
