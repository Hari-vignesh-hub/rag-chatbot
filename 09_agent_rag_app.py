import streamlit as st
import chromadb
import pypdf
import requests
import json
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama

st.set_page_config(page_title="RAG AI Chatbot", layout="wide")

# ---------------------------------------------------------
# 1. SETUP EMBEDDINGS & CHROMADB COLLECTION
# ---------------------------------------------------------
@st.cache_resource
def load_vector_store():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="documents_v2")
    return embedding_model, collection

embedding_model, collection = load_vector_store()
llm = Ollama(model="llama3.2")

# ---------------------------------------------------------
# 2. REAL-TIME WEATHER API TOOL
# ---------------------------------------------------------
def get_weather_data(city: str) -> str:
    """Fetches real-time weather information using Open-Meteo API."""
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_resp = requests.get(geo_url).json()

        if not geo_resp.get("results"):
            return f"Could not find coordinates for city: {city}"

        location = geo_resp["results"][0]
        lat, lon = location["latitude"], location["longitude"]
        city_name = location.get("name", city)
        country = location.get("country", "")

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        w_resp = requests.get(weather_url).json()

        if "current_weather" in w_resp:
            cw = w_resp["current_weather"]
            return f"Weather in {city_name}, {country}: {cw['temperature']}°C with wind speed of {cw['windspeed']} km/h."
        return f"Failed to fetch weather data for {city_name}."
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

# ---------------------------------------------------------
# 3. SIDEBAR (YOUR PREVIOUS UI)
# ---------------------------------------------------------
with st.sidebar:
    st.caption("Model: **llama3.2**")
    st.caption("Vector DB: **ChromaDB**")
    st.divider()

    st.header("📂 Upload your document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.info(f"Selected: **{uploaded_file.name}**")
        
        if st.button("📁 Process Document"):
            with st.spinner("Extracting & Indexing PDF..."):
                pdf_reader = pypdf.PdfReader(uploaded_file)
                extracted_text = ""
                for page in pdf_reader.pages:
                    extracted_text += page.extract_text() or ""
                
                # Chunk text into blocks
                chunks = [extracted_text[i:i+500] for i in range(0, len(extracted_text), 500)]
                embeddings = embedding_model.encode(chunks).tolist()
                ids = [f"{uploaded_file.name}_chunk_{i}" for i in range(len(chunks))]

                collection.add(
                    documents=chunks,
                    embeddings=embeddings,
                    ids=ids
                )
                
                st.session_state["active_doc"] = uploaded_file.name
                st.session_state["extracted_text"] = extracted_text
                st.session_state["chunks_count"] = len(chunks)

    if "active_doc" in st.session_state:
        st.success("Document text extracted successfully.")
        with st.expander("View extracted text"):
            st.text_area("Extracted Content", st.session_state.get("extracted_text", ""), height=150)
        
        st.caption(f"Chunks: **{st.session_state.get('chunks_count', 0)}**")
        st.divider()
        st.markdown("### 📄 Active Document")
        st.write(f"**{st.session_state['active_doc']}**")
        st.success("Document is ready for RAG.")
        st.divider()
        st.markdown("### 📁 Processed Files")
        st.write(f"• {st.session_state['active_doc']}")

# ---------------------------------------------------------
# 4. CHAT INTERFACE & ROUTING
# ---------------------------------------------------------
st.title("🤖 RAG AI Chatbot")
st.caption("Ask questions about your documents or have a normal conversation.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask about your document or anything else...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        # 1. Check for real-time weather query intent
        lower_input = user_input.lower()
        if "weather" in lower_input or "temperature" in lower_input:
            with st.status("🌐 Fetching real-time API data...", expanded=True):
                st.write("Detected weather intent. Calling Open-Meteo API...")
                # Simple city extraction rule or fallback default
                words = user_input.split()
                city = words[-1].replace("?", "").replace(".", "") if len(words) > 1 else "Tokyo"
                api_result = get_weather_data(city)
                st.write(f"API Result: {api_result}")

            prompt = f"Using this live weather data: '{api_result}', answer the user query: '{user_input}'"
            response = llm.invoke(prompt)

        # 2. Otherwise execute standard RAG workflow from previous chatbot
        else:
            with st.status("🔍 Searching query...", expanded=True):
                st.write(f"**Original:** {user_input}")
                st.write(f"**Corrected:** {user_input}")

            # Retrieve chunks from ChromaDB
            q_embed = embedding_model.encode([user_input]).tolist()
            results = collection.query(
                query_embeddings=q_embed,
                n_results=3,
                include=["documents", "distances"]
            )

            docs = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            relevant_chunks = [d for d, dist in zip(docs, distances) if dist < 1.2]

            if relevant_chunks:
                context = "\n\n".join(relevant_chunks)
                prompt = f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer based on the context above:"
            else:
                prompt = f"Answer this query naturally: {user_input}"

            response = llm.invoke(prompt)

        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})