from typing import List

from app.services.embeddings import embed_texts
from app.services.vector_store import recreate_collection, upsert_texts, search_similar

COLLECTION_NAME = "demo_docs"


# Tiny demo corpus – later this can be replaced with real data
_DEMO_DOCUMENTS = [
    {
        "text": "Our system architecture has a Streamlit frontend, a FastAPI backend, "
                "a Qdrant vector database, and uses OpenAI for LLM calls.",
        "metadata": {"source": "architecture_overview"},
    },
    {
        "text": "The RAG pipeline splits documents into chunks, embeds them with OpenAI, "
                "stores vectors in Qdrant, and retrieves top-k chunks for each query.",
        "metadata": {"source": "rag_overview"},
    },
    {
        "text": "We optimize latency and cost by using async FastAPI endpoints, response streaming, "
                "and caching repeated queries.",
        "metadata": {"source": "performance_notes"},
    },
]


def init_rag() -> None:
    """
    Called at app startup:
      - recreate the Qdrant collection
      - embed demo docs
      - upsert them
    """
    texts = [d["text"] for d in _DEMO_DOCUMENTS]
    vectors = embed_texts(texts)

    if not vectors:
        print("Warning: no embeddings generated for demo docs")
        return

    vector_size = len(vectors[0])

    # Recreate collection (dev/demo only!)
    recreate_collection(COLLECTION_NAME, vector_size)

    metadata_list = [d.get("metadata", {}) for d in _DEMO_DOCUMENTS]
    upsert_texts(COLLECTION_NAME, vectors, texts, metadata_list)

    print(f"RAG initialized: {len(texts)} demo docs upserted into '{COLLECTION_NAME}'")


def retrieve_context(query: str, top_k: int = 3) -> List[str]:
    """
    Given a user query, returns a list of top-k relevant text chunks.
    """
    query_vecs = embed_texts([query])
    if not query_vecs:
        return []

    query_vec = query_vecs[0]
    hits = search_similar(COLLECTION_NAME, query_vec, top_k=top_k)

    chunks = [h["text"] for h in hits]
    return chunks


def build_rag_prompt(user_message: str, context_chunks: List[str]) -> str:
    """
    Build the final prompt given retrieved context.
    Very simple for now.
    """
    context_str = "\n\n".join(f"- {c}" for c in context_chunks) if context_chunks else "No context."

    prompt = (
        "You are an AI assistant that uses the provided context to answer questions.\n\n"
        f"Context:\n{context_str}\n\n"
        "When answering, be concise but clear. "
        "If the context is not sufficient, say so explicitly.\n\n"
        f"User question: {user_message}"
    )

    return prompt
