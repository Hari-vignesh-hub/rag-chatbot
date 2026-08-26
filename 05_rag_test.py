with open("documents/python.txt", "r", encoding="utf-8") as file:
    text = file.read()

chunks = text.split("\n\n")

for chunk in chunks:
    print("-----")
    print(chunk)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
print(len(embeddings))
print(embeddings[0])

import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="documents"
)

ids = [str(i) for i in range(len(chunks))]
collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings.tolist()
)

question = "What is Python used for?"
question_embedding = model.encode([question])
results = collection.query(
    query_embeddings=question_embedding.tolist(),
    n_results=2
)
print(results["documents"])