"""End to end on the committed corpus, with Qdrant's in-memory mode.

Downloads the three models on the first run (about 150 MB).
"""

import pytest
from qdrant_client import QdrantClient

from hybrid_search import corpus, pipeline
from hybrid_search.cli import DEFAULT_QUERY
from hybrid_search.models import Models
from hybrid_search.stores.qdrant import QdrantStore

LYSO_GB3_CHUNK = "fda-elfabrio#1"  # "Plasma globotriaosylsphingosine (lyso-Gb3, ...) ..."


@pytest.fixture(scope="module")
def results():
    store = QdrantStore(QdrantClient(":memory:"))
    models = Models()
    chunks = corpus.chunk_documents(corpus.load_documents())
    pipeline.index(store, models, chunks)
    assert store.count() == len(chunks)
    return pipeline.search(store, models, DEFAULT_QUERY, k=5, final=3)


def test_every_stage_returns_ranked_chunks(results):
    for hits in (results.lexical, results.dense, results.fused):
        assert len(hits) == 5
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert len(results.reranked) == 3


def test_bm25_finds_the_exact_biomarker_term(results):
    assert LYSO_GB3_CHUNK in [h.chunk.id for h in results.lexical]


def test_rerank_puts_the_biomarker_chunk_first(results):
    assert results.reranked[0].chunk.id == LYSO_GB3_CHUNK
