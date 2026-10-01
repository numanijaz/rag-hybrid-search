"""Command line: fetch the corpus, index it, search it."""

import argparse
import sys
import textwrap

from qdrant_client import QdrantClient

from hybrid_search import corpus, pipeline
from hybrid_search.models import Models
from hybrid_search.stores import Hit
from hybrid_search.stores.qdrant import QdrantStore

# The article's running example: an everyday symptom description (a job for
# dense search) and an exact biomarker name (a job for BM25).
DEFAULT_QUERY = "burning pain in hands and feet, raised lyso-Gb3"


def _store(args: argparse.Namespace) -> QdrantStore:
    return QdrantStore(QdrantClient(url=args.qdrant_url))


def cmd_fetch(args: argparse.Namespace) -> None:
    documents = corpus.fetch_all()
    for doc in documents:
        print(f"  {corpus.write_document(doc).relative_to(corpus.DATA_DIR.parent)}")
    corpus.write_sources_index(documents)
    print(f"Fetched {len(documents)} documents; see data/SOURCES.md.")


def cmd_index(args: argparse.Namespace) -> None:
    chunks = corpus.chunk_documents(corpus.load_documents())
    store = _store(args)
    pipeline.index(store, Models(), chunks)
    print(f"Indexed {store.count()} chunks into Qdrant at {args.qdrant_url}.")


def _print_hits(heading: str, hits: list[Hit]) -> None:
    print(f"\n{heading}")
    width = max(len(h.chunk.id) for h in hits)
    for rank, hit in enumerate(hits, 1):
        text = textwrap.shorten(hit.chunk.text, 60, placeholder="...")
        print(f"  {rank:>2}  {hit.score:7.3f}  {hit.chunk.id:<{width}}  {text}")


def cmd_search(args: argparse.Namespace) -> None:
    store = _store(args)
    if store.count() == 0:
        sys.exit("No index yet. Run `uv run hybrid-search index` first.")
    results = pipeline.search(store, Models(), args.query, k=args.k, final=args.final)
    print(f"Query: {args.query}")
    _print_hits(f"BM25 (lexical search), top {args.k}", results.lexical)
    _print_hits(f"Dense search, top {args.k}", results.dense)
    _print_hits(f"Hybrid: RRF over both, top {args.k}", results.fused)
    _print_hits(
        f"Hybrid + cross-encoder rerank of the fused top {pipeline.RERANK_INPUT}, "
        f"top {args.final}",
        results.reranked,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="hybrid-search", description=__doc__)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    commands = parser.add_subparsers(required=True)

    fetch = commands.add_parser("fetch", help="download the corpus into data/documents/")
    fetch.set_defaults(run=cmd_fetch)

    index = commands.add_parser("index", help="chunk, embed and index data/documents/")
    index.set_defaults(run=cmd_index)

    search = commands.add_parser("search", help="compare BM25, dense, hybrid and rerank")
    search.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    search.add_argument("-k", type=int, default=5, help="chunks shown per retriever")
    search.add_argument("--final", type=int, default=3, help="chunks kept after rerank")
    search.set_defaults(run=cmd_search)

    args = parser.parse_args()
    args.run(args)
