from io import BytesIO
from unittest.mock import patch, MagicMock

from rag_utils import create_chunks, extract_text


def test_create_chunks_splits_long_document():
    text = "A" * 2500
    chunks = create_chunks(text, chunk_size=1000, overlap=200)

    assert len(chunks) == 4
    assert len(chunks[0]) == 1000
    assert len(chunks[1]) == 1000
    assert len(chunks[2]) == 900
    assert len(chunks[3]) == 100


def test_create_chunks_empty_text():
    assert create_chunks("") == []


def test_create_chunks_whitespace():
    assert create_chunks("     ") == []


def test_create_chunks_rejects_invalid_overlap():
    import pytest

    with pytest.raises(ValueError):
        create_chunks("hello", chunk_size=10, overlap=10)


def test_extract_text_txt():
    uploaded = BytesIO(b"Real chatbot document")
    uploaded.name = "sample.txt"

    assert extract_text(uploaded) == "Real chatbot document"


def test_extract_text_unsupported_file():
    uploaded = BytesIO(b"some data")
    uploaded.name = "sample.docx"

    assert extract_text(uploaded) == ""


@patch("rag_utils.PdfReader")
def test_extract_text_pdf(mock_reader):
    page1 = MagicMock()
    page1.extract_text.return_value = "First PDF page"

    page2 = MagicMock()
    page2.extract_text.return_value = "Second PDF page"

    mock_reader.return_value.pages = [page1, page2]

    uploaded = BytesIO(b"fake pdf")
    uploaded.name = "sample.pdf"

    result = extract_text(uploaded)

    assert "First PDF page" in result
    assert "Second PDF page" in result
