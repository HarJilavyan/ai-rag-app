from typing import List

from openai import OpenAI

from app.core.config import get_settings

settings = get_settings()

# Sync client is fine for embeddings
_client = OpenAI(api_key=settings.openai_api_key)

EMBEDDING_MODEL = "text-embedding-3-small"


def embed_texts(texts: List[str]) -> List[list[float]]:
    """
    Returns a list of embedding vectors for the given texts.
    """
    if not texts:
        return []

    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    # Each item has .embedding
    return [item.embedding for item in response.data]
