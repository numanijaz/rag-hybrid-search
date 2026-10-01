"""Qdrant: one collection, a dense and a sparse vector per chunk.

Qdrant runs all three searches itself. Hybrid search is a single query:
two `prefetch` sub-queries (BM25 and dense) and an RRF fusion over their
results, all on the server.
"""

import uuid

from fastembed import SparseEmbedding
from qdrant_client import QdrantClient, models

from hybrid_search.corpus import Chunk
from hybrid_search.stores import Hit

DENSE = "dense"
BM25 = "bm25"


def _sparse(embedding: SparseEmbedding) -> models.SparseVector:
    return models.SparseVector(
        indices=embedding.indices.tolist(), values=embedding.values.tolist()
    )


def _point_id(chunk_id: str) -> str:
    # Qdrant point IDs are integers or UUIDs; derive a stable UUID.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str = "chunks") -> None:
        self.client = client
        self.collection = collection

    def recreate(self, dense_dim: int) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
            },
            # The IDF modifier makes Qdrant compute inverse document frequency
            # over the collection at query time; with the term weights from
            # the Qdrant/bm25 model, the dot product is the BM25 score.
            sparse_vectors_config={BM25: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )

    def upsert(
        self, chunks: list[Chunk], dense: list[list[float]], sparse: list[SparseEmbedding]
    ) -> None:
        self.client.upsert(
            self.collection,
            points=[
                models.PointStruct(
                    id=_point_id(chunk.id),
                    vector={DENSE: d, BM25: _sparse(s)},
                    payload=chunk.__dict__,
                )
                for chunk, d, s in zip(chunks, dense, sparse, strict=True)
            ],
        )

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection).count

    def _query(self, **kwargs) -> list[Hit]:
        points = self.client.query_points(self.collection, with_payload=True, **kwargs).points
        return [Hit(chunk=Chunk(**p.payload), score=p.score) for p in points]

    def lexical_search(self, query: SparseEmbedding, limit: int) -> list[Hit]:
        return self._query(query=_sparse(query), using=BM25, limit=limit)

    def dense_search(self, query: list[float], limit: int) -> list[Hit]:
        return self._query(query=query, using=DENSE, limit=limit)

    def hybrid_search(
        self,
        dense: list[float],
        sparse: SparseEmbedding,
        limit: int,
        candidates: int,
    ) -> list[Hit]:
        return self._query(
            prefetch=[
                models.Prefetch(query=_sparse(sparse), using=BM25, limit=candidates),
                models.Prefetch(query=dense, using=DENSE, limit=candidates),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )
