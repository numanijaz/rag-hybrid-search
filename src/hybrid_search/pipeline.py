"""Indexing and retrieval, independent of the store behind them."""

from dataclasses import dataclass

from hybrid_search.corpus import Chunk
from hybrid_search.models import Models
from hybrid_search.stores import Hit, Store

# Candidates each retriever hands to fusion.
CANDIDATES = 20
# Fused chunks the cross-encoder rescores. Small on purpose: every chunk costs
# one forward pass of the cross-encoder.
RERANK_INPUT = 10


@dataclass(frozen=True)
class Results:
    lexical: list[Hit]
    dense: list[Hit]
    fused: list[Hit]
    reranked: list[Hit]


def index(store: Store, models: Models, chunks: list[Chunk]) -> None:
    dense, sparse = models.embed_chunks([c.text for c in chunks])
    store.recreate(models.dense_dim)
    store.upsert(chunks, dense, sparse)


def rerank(models: Models, query: str, hits: list[Hit]) -> list[Hit]:
    scores = models.rerank(query, [h.chunk.text for h in hits])
    rescored = [Hit(chunk=h.chunk, score=s) for h, s in zip(hits, scores, strict=True)]
    return sorted(rescored, key=lambda h: h.score, reverse=True)


def search(store: Store, models: Models, query: str, k: int, final: int) -> Results:
    """Run lexical, dense and hybrid search, then rerank the fused list.

    Lexical and dense search alone are only there for comparison; a real
    system runs just the hybrid query and the rerank.
    """
    dense, sparse = models.embed_query(query)
    fused = store.hybrid_search(dense, sparse, limit=max(k, RERANK_INPUT), candidates=CANDIDATES)
    return Results(
        lexical=store.lexical_search(sparse, limit=k),
        dense=store.dense_search(dense, limit=k),
        fused=fused[:k],
        reranked=rerank(models, query, fused[:RERANK_INPUT])[:final],
    )
