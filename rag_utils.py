from pypdf import PdfReader
from rapidfuzz import process, fuzz

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
FUZZY_THRESHOLD = 85


def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


def extract_text(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    elif file_name.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"

        return text

    return ""


def correct_query(query, pdf_words):
    words = query.split()
    corrected_words = []

    for word in words:
        clean_word = word.strip(".,!?;:()[]{}\"'")

        if len(clean_word) < 4:
            corrected_words.append(word)
            continue

        result = process.extractOne(
            clean_word, pdf_words, scorer=fuzz.ratio
        )

        if result:
            match, score, _ = result
            if score >= FUZZY_THRESHOLD:
                corrected_words.append(match)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)
