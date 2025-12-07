from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# _qdrant = QdrantClient(host="localhost", port=6333)
_qdrant = QdrantClient(host="qdrant", port=6333)


def recreate_collection(collection_name: str, vector_size: int) -> None:
    _qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )


def upsert_texts(
    collection_name: str,
    vectors: List[list[float]],
    texts: List[str],
    metadata_list: List[Dict],
) -> None:
    points = []
    for idx, (vec, text, meta) in enumerate(zip(vectors, texts, metadata_list)):
        payload = {"text": text}
        payload.update(meta or {})
        points.append(
            PointStruct(
                id=idx,
                vector=vec,
                payload=payload,
            )
        )

    _qdrant.upsert(
        collection_name=collection_name,
        points=points,
    )


def search_similar(
    collection_name: str,
    query_vector: list[float],
    top_k: int = 3,
) -> List[Dict]:
    """
    Uses query_points (newer qdrant-client API) instead of search().
    """
    response = _qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,       # in older versions this might be 'vector=query_vector'
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    results = []
    for p in response.points:
        results.append(
            {
                "score": p.score,
                "text": p.payload.get("text", ""),
                "payload": p.payload,
            }
        )

    return results
