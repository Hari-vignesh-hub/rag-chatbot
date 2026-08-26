import json
import re
import requests
import streamlit as st
import ollama
import chromadb

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from rapidfuzz import process, fuzz


# ============================================================
# SYSTEM PROMPT (Bot Identity & Rules)
# ============================================================

SYSTEM_PROMPT = """You are a helpful, accurate, and intelligent AI assistant. 
You were created and developed by [Harivignesh.S]. If anyone asks who created you, who built you, or who your owner is, state clearly that you were created by [Your Name].

Answer questions using the provided context (real-time API context or document context).
If the context contains real-time weather information, present it naturally to the user.
If no context is relevant, rely on your general knowledge base."""
# ============================================================
# 1. QUERY CORRECTION FUNCTION
# ============================================================

def correct_query(query: str, pdf_words: list) -> str:
    if not pdf_words:
        return query

    vocab_map = {w.lower(): w for w in pdf_words}
    vocab_keys = list(vocab_map.keys())

    words = query.split()
    corrected_words = []
    i = 0

    while i < len(words):
        if i + 1 < len(words):
            word1 = words[i].strip(".,!?;:()[]{}\"'")
            word2 = words[i + 1].strip(".,!?;:()[]{}\"'")
            combined = (word1 + word2).lower()

            result = process.extractOne(combined, vocab_keys, scorer=fuzz.ratio)
            if result:
                match_key, score, _ = result
                if len(combined) >= 5 and score >= 70:
                    corrected_words.append(vocab_map[match_key])
                    i += 2
                    continue

        word = words[i]
        clean_word = word.strip(".,!?;:()[]{}\"'").lower()

        if len(clean_word) < 4:
            corrected_words.append(word)
            i += 1
            continue

        result = process.extractOne(clean_word, vocab_keys, scorer=fuzz.ratio)
        if result:
            match_key, score, _ = result
            if score >= 75:
                corrected_words.append(vocab_map[match_key])
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)

        i += 1

    return " ".join(corrected_words)


# ============================================================
# 2. REAL-TIME WEATHER API FUNCTION
# ============================================================

def fetch_weather_api(city: str) -> str:
    """Fetches live temperature and wind speed for a target city using Open-Meteo."""
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_resp = requests.get(geo_url, timeout=5).json()

        if not geo_resp.get("results"):
            return f"Could not find geographic coordinates for '{city}'."

        location = geo_resp["results"][0]
        lat, lon = location["latitude"], location["longitude"]
        city_name = location.get("name", city)
        country = location.get("country", "")

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        w_resp = requests.get(weather_url, timeout=5).json()

        if "current_weather" in w_resp:
            cw = w_resp["current_weather"]
            return f"Current live weather in {city_name}, {country}: {cw['temperature']}°C with wind speed of {cw['windspeed']} km/h."
        return f"Failed to retrieve current weather data for {city_name}."
    except Exception as e:
        return f"API Error: {str(e)}"


# ============================================================
# 3. PAGE CONFIGURATION & SYSTEM PROMPT
# ============================================================

st.set_page_config(page_title="RAG AI Chatbot", page_icon="🤖", layout="wide")

SYSTEM_PROMPT = """You are a helpful AI assistant.

Capabilities & Guidelines:
1. Live Weather: Provide accurate advice based on provided API context.
2. Documents: Answer questions using document context when available.
3. Memory: Remember user details like name and home city across turns.
4. Ambiguity: When user mentions Indian cities (e.g., Salem, Chennai), refer to Tamil Nadu/India, not US locations."""

st.title("🤖 RAG AI Chatbot")
st.caption("Ask questions about your documents or have a normal conversation.")


# ============================================================
# 4. LOAD EMBEDDING MODEL & VECTOR STORE
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name="documents_v2")

collection = load_collection()


# ============================================================
# 5. SESSION STATE INITIALIZATION
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_words" not in st.session_state:
    st.session_state.pdf_words = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False
if "current_document" not in st.session_state:
    st.session_state.current_document = None
if "user_home_city" not in st.session_state:
    st.session_state.user_home_city = None


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:
    st.header("📚 Knowledge Base")
    st.write("Model: `llama3.2` | Vector DB: `ChromaDB`")
    if st.session_state.user_home_city:
        st.info(f"📍 Home City: **{st.session_state.user_home_city}**")
    st.divider()

    uploaded_file = st.file_uploader("📚 Upload document", type=["txt", "pdf"])

    if uploaded_file is not None:
        text = ""
        if uploaded_file.name.lower().endswith(".txt"):
            file_bytes = uploaded_file.read()
            text = file_bytes.decode("utf-8" if "utf-8" else "latin-1")
        elif uploaded_file.name.lower().endswith(".pdf"):
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"

        if text.strip():
            chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
            if st.button("📥 Process Document", use_container_width=True):
                words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text)
                st.session_state.pdf_words = list(set(words))
                embeddings = embedding_model.encode(chunks)
                ids = [f"{uploaded_file.name}_{i}" for i in range(len(chunks))]
                metadatas = [{"source": uploaded_file.name, "chunk": i} for i in range(len(chunks))]

                collection.upsert(ids=ids, documents=chunks, embeddings=embeddings.tolist(), metadatas=metadatas)
                st.session_state.document_loaded = True
                st.session_state.current_document = uploaded_file.name
                if uploaded_file.name not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files.append(uploaded_file.name)
                st.success("Document processed!")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.user_home_city = None
        st.rerun()


# ============================================================
# 7. DISPLAY CONVERSATION
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# 8. CHAT INPUT & EXECUTION
# ============================================================

user_input = st.chat_input("Ask about your document or anything else...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    # Track home city when user explicitly mentions where they live
    city_match = re.search(r"(?:living in|i am in|i live in|from)\s+([A-Za-z]+)", user_input, re.IGNORECASE)
    if city_match:
        st.session_state.user_home_city = city_match.group(1).title()

    normalized_question = correct_query(user_input, st.session_state.pdf_words)
    if normalized_question != user_input:
        with st.expander("🔎 Search query"):
            st.write("Original:", user_input)
            st.write("Corrected:", normalized_question)

    relevant_documents = []
    sources = []
    realtime_context = ""
    lower_query = user_input.lower()
    weather_keywords = ["weather", "temperature", "climate", "forecast", "rain", "degree", "outside"]

    # --------------------------------------------------------
    # ROUTE A: REAL-TIME WEATHER API WITH HOME CITY MEMORY
    # --------------------------------------------------------
    if any(keyword in lower_query for keyword in weather_keywords):
        with st.status("🌐 Executing Real-Time Weather Tool...", expanded=True):
            ignore_words = {
                "what", "whats", "is", "the", "weather", "in", "like", "today",
                "temperature", "at", "for", "of", "condition", "conditions",
                "forecast", "outside", "going", "good", "not", "based", "on",
                "my", "home", "city", "town", "place", "here", "there", "am", "i"
            }

            clean_text = re.sub(r"[^\w\s]", "", user_input)
            words = clean_text.split()
            
            # Check if asking about "my city" or "here"
            if any(term in lower_query for term in ["my city", "here", "my area", "outside"]):
                target_city = st.session_state.user_home_city or "Chennai"
            else:
                city_candidates = [w for w in words if w.lower() not in ignore_words]
                lower_words = [w.lower() for w in words]
                
                if "in" in lower_words:
                    idx = lower_words.index("in")
                    target_city = words[idx + 1] if idx + 1 < len(words) else "Chennai"
                elif "at" in lower_words:
                    idx = lower_words.index("at")
                    target_city = words[idx + 1] if idx + 1 < len(words) else "Chennai"
                elif city_candidates:
                    target_city = city_candidates[-1]
                else:
                    target_city = st.session_state.user_home_city or "Chennai"

            st.write(f"Detected Target City: **{target_city}**")
            realtime_context = fetch_weather_api(target_city)
            st.write(f"API Response: `{realtime_context}`")

    # --------------------------------------------------------
    # ROUTE B: CHROMADB VECTOR SEARCH
    # --------------------------------------------------------
    elif st.session_state.document_loaded:
        question_embedding = embedding_model.encode([normalized_question])
        try:
            results = collection.query(
                query_embeddings=question_embedding.tolist(),
                n_results=5,
                where={"source": st.session_state.current_document},
                include=["documents", "distances", "metadatas"]
            )
            retrieved_documents = results["documents"][0] if results.get("documents") else []
            distances = results["distances"][0] if results.get("distances") else []
            metadatas = results["metadatas"][0] if results.get("metadatas") else []

            for doc, dist, meta in zip(retrieved_documents, distances, metadatas):
                if dist < 1.2:
                    relevant_documents.append(doc)
                    if meta and meta.get("source") and meta["source"] not in sources:
                        sources.append(meta["source"])
        except Exception as e:
            st.error(f"Document search error: {e}")

    # --------------------------------------------------------
    # OLLAMA PROMPT INVOCATION
    # --------------------------------------------------------
    doc_context = "\n\n".join(relevant_documents) if relevant_documents else "None"
    api_context = realtime_context if realtime_context else "None"

    dynamic_system_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"User's Saved Home City: {st.session_state.user_home_city or 'Unknown'}\n"
        f"Real-Time API Context: {api_context}\n"
        f"Relevant Document Context:\n{doc_context}"
    )

    conversation_messages = [{"role": "system", "content": dynamic_system_prompt}]
    for msg in st.session_state.messages:
        conversation_messages.append(msg)

    with st.chat_message("assistant"):
        try:
            response = ollama.chat(
                model="llama3.2",
                messages=conversation_messages,
                stream=True
            )
            bot_response = st.write_stream(chunk["message"]["content"] for chunk in response)
        except Exception as e:
            bot_response = f"Error: {e}"
            st.error(bot_response)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    if relevant_documents and sources:
        st.caption("📄 Source: " + ", ".join(sources))