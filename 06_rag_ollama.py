import chromadb
import ollama
from sentence_transformers import SentenceTransformer


# ---------------------------------------
# 1. Load embedding model
# ---------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------
# 2. Connect to ChromaDB
# ---------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)


# ---------------------------------------
# 3. Get our collection
# ---------------------------------------

collection = client.get_or_create_collection(
    name="documents"
)


# ---------------------------------------
# 4. Ask the user a question
# ---------------------------------------

question = input("You: ")


# ---------------------------------------
# 5. Convert question into embedding
# ---------------------------------------

question_embedding = model.encode([question])


# ---------------------------------------
# 6. Search ChromaDB
# ---------------------------------------

results = collection.query(
    query_embeddings=question_embedding.tolist(),
    n_results=2
)


# ---------------------------------------
# 7. Get relevant documents
# ---------------------------------------

retrieved_documents = results["documents"][0]


# ---------------------------------------
# 8. Combine retrieved documents
# ---------------------------------------

context = "\n\n".join(retrieved_documents)


print("\nRetrieved Context:")
print(context)


# ---------------------------------------
# 9. Create prompt for Llama
# ---------------------------------------

prompt = f"""
You are a helpful AI assistant.

Answer the user's question using the provided context.

If the answer is not present in the context,
say that the information is not available
in the provided documents.

Context:
{context}

User Question:
{question}
"""


# ---------------------------------------
# 10. Send context + question to Ollama
# ---------------------------------------

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# ---------------------------------------
# 11. Get AI answer
# ---------------------------------------

answer = response["message"]["content"]


# ---------------------------------------
# 12. Display answer
# ---------------------------------------

print("\nBot:")
print(answer)