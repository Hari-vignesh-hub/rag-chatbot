import chromadb


def test_chromadb_stores_and_retrieves_document():
    client = chromadb.Client()
    collection = client.create_collection(name="test_documents")

    collection.add(
        ids=["doc1"],
        documents=["The AI chatbot uses retrieval augmented generation."],
        embeddings=[[0.1, 0.2, 0.3]]
    )

    result = collection.query(
        query_embeddings=[[0.1, 0.2, 0.3]],
        n_results=1
    )

    assert result["ids"][0] == ["doc1"]
    assert "retrieval augmented generation" in result["documents"][0][0]
