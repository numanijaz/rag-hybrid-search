"""The three models in the pipeline, all local ONNX models via fastembed.

No GPU and no API key: each model is downloaded once (to fastembed's cache)
and runs on the CPU.
"""

from dataclasses import dataclass
from functools import cached_property

from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

# Embedding model for dense search: 384-dimensional vectors.
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
# BM25 term weights as a sparse vector. The IDF part is computed by Qdrant
# (see stores/qdrant.py), so this model only supplies term frequencies.
SPARSE_MODEL = "Qdrant/bm25"
# Cross-encoder: scores each (query, chunk) pair jointly.
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


@dataclass
class Models:
    dense_model: str = DENSE_MODEL
    sparse_model: str = SPARSE_MODEL
    rerank_model: str = RERANK_MODEL

    @cached_property
    def dense(self) -> TextEmbedding:
        return TextEmbedding(self.dense_model)

    @cached_property
    def sparse(self) -> SparseTextEmbedding:
        return SparseTextEmbedding(self.sparse_model)

    @cached_property
    def reranker(self) -> TextCrossEncoder:
        return TextCrossEncoder(self.rerank_model)

    @property
    def dense_dim(self) -> int:
        return self.dense.embedding_size

    def embed_chunks(self, texts: list[str]) -> tuple[list[list[float]], list[SparseEmbedding]]:
        dense = [v.tolist() for v in self.dense.embed(texts)]
        sparse = list(self.sparse.embed(texts))
        return dense, sparse

    def embed_query(self, query: str) -> tuple[list[float], SparseEmbedding]:
        # query_embed, not embed: some models treat queries differently from
        # chunks (BM25, for one, ignores query term frequency).
        dense = next(iter(self.dense.query_embed(query))).tolist()
        sparse = next(iter(self.sparse.query_embed(query)))
        return dense, sparse

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        return list(self.reranker.rerank(query, texts))
