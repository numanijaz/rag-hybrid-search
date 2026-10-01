"""Vector stores. Each one indexes chunks and runs the searches itself.

A store implements the `Store` protocol below; the pipeline only talks to that
interface, so another store (pgvector, Elasticsearch, ...) can be added next to
qdrant.py without touching the rest.
"""

from dataclasses import dataclass
from typing import Protocol

from fastembed import SparseEmbedding

from hybrid_search.corpus import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Store(Protocol):
    def recreate(self, dense_dim: int) -> None:
        """Drop and create an empty index."""

    def upsert(
        self, chunks: list[Chunk], dense: list[list[float]], sparse: list[SparseEmbedding]
    ) -> None:
        """Store each chunk with its dense and sparse vector."""

    def count(self) -> int:
        """Number of indexed chunks, 0 if there is no index."""

    def lexical_search(self, query: SparseEmbedding, limit: int) -> list[Hit]:
        """BM25 alone."""

    def dense_search(self, query: list[float], limit: int) -> list[Hit]:
        """Nearest neighbours of the query vector alone."""

    def hybrid_search(
        self,
        dense: list[float],
        sparse: SparseEmbedding,
        limit: int,
        candidates: int,
    ) -> list[Hit]:
        """Top `candidates` from each retriever, fused with reciprocal rank fusion."""
