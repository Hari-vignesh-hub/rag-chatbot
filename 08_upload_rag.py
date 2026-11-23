import streamlit as st
import ollama
import chromadb

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from rapidfuzz import process, fuzz


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG AI Chatbot",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# 2. BASIC SETTINGS
# ============================================================

MODEL_NAME = "llama3.2"

CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "documents"

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

TOP_K = 5

FUZZY_THRESHOLD = 85

DISTANCE_THRESHOLD = 1.2


# ============================================================
# 3. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a helpful AI assistant.

You can answer two types of questions:

1. Questions about the user's uploaded documents.
2. General questions and normal conversation.

When relevant information from an uploaded document
is provided, prefer that information.

If the retrieved document context is not relevant
to the user's question, answer using your general
knowledge.

Never invent information from the document.

Never claim that information came from the document
unless it is actually supported by the provided
document context.

Keep answers clear and easy to understand.
"""


# ============================================================
# 4. CREATE CHUNKS
# ============================================================

def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(
                chunk.strip()
            )

        start += chunk_size - overlap

    return chunks


# ============================================================
# 5. EXTRACT TEXT FROM FILE
# ============================================================

def extract_text(uploaded_file):

    file_name = uploaded_file.name.lower()


    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    if file_name.endswith(".txt"):

        file_bytes = uploaded_file.read()

        text = file_bytes.decode(
            "utf-8",
            errors="ignore"
        )

        return text


    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    elif file_name.endswith(".pdf"):

        pdf_reader = PdfReader(
            uploaded_file
        )

        text = ""

        for page in pdf_reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text

                text += "\n\n"

        return text


    return ""


# ============================================================
# 6. LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# 7. LOAD CHROMADB
# ============================================================

@st.cache_resource
def load_collection():

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection


collection = load_collection()


# ============================================================
# 8. QUERY TYPO CORRECTION
# ============================================================

def correct_query(query, pdf_words):

    words = query.split()

    corrected_words = []

    for word in words:

        clean_word = word.strip(
            ".,!?;:()[]{}\"'"
        )


        # Very short words are not corrected
        if len(clean_word) < 4:

            corrected_words.append(word)

            continue


        # Find closest word in document
        result = process.extractOne(
            clean_word,
            pdf_words,
            scorer=fuzz.ratio
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


# ============================================================
# 9. SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]


if "pdf_words" not in st.session_state:

    st.session_state.pdf_words = []


if "uploaded_files" not in st.session_state:

    existing_data = collection.get(
        include=["metadatas"]
    )

    existing_sources = {
        m["source"]
        for m in existing_data.get("metadatas", [])
        if m and "source" in m
    }

    st.session_state.uploaded_files = list(
        existing_sources
    )


# ============================================================
# 10. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 Knowledge Base")

    st.write(
        f"Model: `{MODEL_NAME}`"
    )

    st.write(
        "Vector DB: `ChromaDB`"
    )


    st.divider()


    # ========================================================
    # FILE UPLOAD
    # ========================================================

    uploaded_file = st.file_uploader(
        "Upload TXT or PDF",
        type=["txt", "pdf"],
        accept_multiple_files=False
    )


    if uploaded_file is not None:

        st.write(
            "Selected:",
            uploaded_file.name
        )


        # ----------------------------------------------------
        # PROCESS DOCUMENT
        # ----------------------------------------------------

        if st.button(
            "📥 Process Document"
        ):

            with st.spinner(
                "Reading document..."
            ):

                text = extract_text(
                    uploaded_file
                )


            # ------------------------------------------------
            # CHECK TEXT
            # ------------------------------------------------

            if not text.strip():

                st.error(
                    "Could not extract text from this file."
                )

            else:

                # ------------------------------------------------
                # SAVE WORDS FOR TYPO CORRECTION
                # ------------------------------------------------

                pdf_words = list(
                    set(
                        text.split()
                    )
                )

                st.session_state.pdf_words = pdf_words


                # ------------------------------------------------
                # CREATE CHUNKS
                # ------------------------------------------------

                chunks = create_chunks(
                    text
                )


                st.write(
                    "Number of chunks:",
                    len(chunks)
                )


                # ------------------------------------------------
                # CREATE EMBEDDINGS
                # ------------------------------------------------

                with st.spinner(
                    "Creating embeddings..."
                ):

                    embeddings = embedding_model.encode(
                        chunks
                    )


                # ------------------------------------------------
                # CREATE UNIQUE IDS
                # ------------------------------------------------

                ids = [

                    f"{uploaded_file.name}_{i}"

                    for i in range(
                        len(chunks)
                    )

                ]


                # ------------------------------------------------
                # CREATE METADATA
                # ------------------------------------------------

                metadatas = [

                    {
                        "source": uploaded_file.name,
                        "chunk": i
                    }

                    for i in range(
                        len(chunks)
                    )

                ]


                # ------------------------------------------------
                # STORE IN CHROMADB
                # ------------------------------------------------

                collection.upsert(

                    ids=ids,

                    documents=chunks,

                    embeddings=embeddings.tolist(),

                    metadatas=metadatas

                )


                # ------------------------------------------------
                # SAVE FILE NAME
                # ------------------------------------------------

                if uploaded_file.name not in st.session_state.uploaded_files:

                    st.session_state.uploaded_files.append(
                        uploaded_file.name
                    )


                st.success(
                    "Document processed successfully!"
                )


                # ------------------------------------------------
                # SHOW DOCUMENT
                # ------------------------------------------------

                with st.expander(
                    "View extracted text"
                ):

                    st.text_area(
                        "Document content",
                        text,
                        height=300
                    )


    # ========================================================
    # SHOW UPLOADED FILES
    # ========================================================

    st.divider()

    st.subheader(
        "📄 Uploaded Files"
    )


    if st.session_state.uploaded_files:

        for file_name in st.session_state.uploaded_files:

            st.write(
                "•",
                file_name
            )

        if st.button(
            "🧹 Clear Database"
        ):

            existing_ids = collection.get()["ids"]

            if existing_ids:

                collection.delete(
                    ids=existing_ids
                )

            st.session_state.uploaded_files = []

            st.session_state.pdf_words = []

            st.rerun()

    else:

        st.write(
            "No documents processed yet."
        )


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    st.divider()

    if st.button(
        "🗑️ Clear Chat"
    ):

        st.session_state.messages = [

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }

        ]

        st.rerun()


# ============================================================
# 11. MAIN TITLE
# ============================================================

st.title(
    "🤖 RAG AI Chatbot"
)

st.caption(
    "Ask questions about your documents or anything else."
)


# ============================================================
# 12. DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "system":

        continue


    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# 13. CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask something..."
)


# ============================================================
# 14. PROCESS USER QUESTION
# ============================================================

if user_input:


    # ========================================================
    # DISPLAY USER MESSAGE
    # ========================================================

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_input
        )


    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    st.session_state.messages.append(

        {
            "role": "user",
            "content": user_input
        }

    )


    # ========================================================
    # QUERY CORRECTION
    # ========================================================

    corrected_question = correct_query(

        user_input,

        st.session_state.pdf_words

    )


    # ========================================================
    # DEBUG INFORMATION
    # ========================================================

    if corrected_question.lower() != user_input.lower():

        with st.expander(
            "🔎 Search query"
        ):

            st.write(
                "Original:",
                user_input
            )

            st.write(
                "Corrected:",
                corrected_question
            )


    # ========================================================
    # CREATE QUESTION EMBEDDING
    # ========================================================

    question_embedding = embedding_model.encode(

        [corrected_question]

    )


    # ========================================================
    # SEARCH CHROMADB
    # ========================================================

    try:

        results = collection.query(

            query_embeddings=
            question_embedding.tolist(),

            n_results=TOP_K,

            include=[
                "documents",
                "distances",
                "metadatas"
            ]

        )

    except Exception:

        results = {
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]]
        }


    # ========================================================
    # GET RETRIEVED DOCUMENTS
    # ========================================================

    retrieved_documents = results.get(
        "documents",
        [[]]
    )[0]


    distances = results.get(
        "distances",
        [[]]
    )[0]


    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]


    # ========================================================
    # CREATE CONTEXT
    # ========================================================

    context_parts = []

    sources = []


    for i, document in enumerate(
        retrieved_documents
    ):

        if i < len(distances) and distances[i] > DISTANCE_THRESHOLD:

            continue


        context_parts.append(
            document
        )


        if i < len(metadatas):

            metadata = metadatas[i]

            if metadata:

                source = metadata.get(
                    "source"
                )

                if source and source not in sources:

                    sources.append(
                        source
                    )


    if context_parts:

        context = "\n\n".join(
            context_parts
        )

    else:

        context = (
            "No document context was retrieved."
        )


    # ========================================================
    # CREATE RAG PROMPT
    # ========================================================

    prompt = f"""
You are a helpful AI assistant.

The user may ask questions about uploaded
documents or ask general questions.

Here is the information retrieved from the
uploaded documents:

================ DOCUMENT CONTEXT ================

{context}

====================================================

User question:

{user_input}

Instructions:

1. If the document context contains information
   relevant to the user's question, use that
   information to answer.

2. If the user is asking about the uploaded
   document, prefer the document context.

3. If the document context does not contain
   useful information for the question, answer
   normally using your general knowledge.

4. Do not invent information from the document.

5. Do not say "the document says" unless the
   provided context actually supports the answer.

6. If the question is a normal/general question,
   answer it normally.

7. Give a clear and useful answer.

Answer the user now.
"""


    # ========================================================
    # ASK OLLAMA
    # ========================================================

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Thinking..."
        ):

            response = ollama.chat(

                model=MODEL_NAME,

                messages=[

                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },

                    {
                        "role": "user",
                        "content": prompt
                    }

                ],

                stream=True

            )


        # ====================================================
        # STREAM RESPONSE
        # ====================================================

        bot_response = st.write_stream(

            chunk["message"]["content"]

            for chunk in response

        )


    # ========================================================
    # SHOW SOURCES
    # ========================================================

    if sources:

        st.caption(
            "📄 Retrieved from: "
            + ", ".join(sources)
        )


    # ========================================================
    # SAVE ASSISTANT RESPONSE
    # ========================================================

    st.session_state.messages.append(

        {
            "role": "assistant",
            "content": bot_response
        }

    )