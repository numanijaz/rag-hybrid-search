# Hybrid search and reranking, in practice

1. **BM25 (lexical search)** and **dense search** over the same chunks, in
   parallel.
2. **Reciprocal rank fusion (RRF)** of the two ranked lists.
3. **Cross-encoder rerank** of the fused top 10, keeping the top 3.

It uses [Qdrant](https://qdrant.tech/) as the vector index and
[fastembed](https://github.com/qdrant/fastembed) for all three models, so it
runs on a laptop CPU with no API keys.

## Run it

You need [uv](https://docs.astral.sh/uv/) and Docker.

```sh
docker compose up -d              # Qdrant on localhost:6333
uv run hybrid-search index        # chunk, embed and index data/documents/
uv run hybrid-search search       # the article's query
uv run hybrid-search search "enzyme replacement therapy for Fabry disease"
```

The first `index` downloads the models, about 150 MB in total, into
fastembed's cache. `uv run pytest` runs the tests. They use Qdrant's
in-memory mode, so they don't need Docker.

## The corpus

Thirteen short, plain-English documents, committed in
[data/documents/](data/documents/). Each file records its source URL, licence
and retrieval date. [data/SOURCES.md](data/SOURCES.md) lists them all.

- **MedlinePlus Genetics** summaries from the U.S. National Library of
  Medicine: Fabry disease, the GLA gene, and nine conditions with similar
  symptoms (erythromelalgia, small fiber neuropathy, and others). They are
  public domain
  ([terms](https://medlineplus.gov/about/using/usingcontent/)).
- **FDA drug labels** via [openFDA](https://open.fda.gov/): two Fabry
  treatments (ELFABRIO, Galafold) and the hand-foot syndrome section of a
  chemotherapy drug (XELODA). They are public domain, CC0 1.0
  ([terms](https://open.fda.gov/terms/)).

[data/sources.toml](data/sources.toml) lists what is fetched.
`uv run hybrid-search fetch` downloads everything again and rewrites the
files, so a diff shows what changed upstream.

## The query and what to look for

The query is the one used throughout the article:

> burning pain in hands and feet, raised lyso-Gb3

It has two halves on purpose. The first is an everyday description of a
symptom that several conditions share. The second is an exact biomarker name,
and in this corpus it appears in only one place: the ELFABRIO label, which
says that *"plasma globotriaosylsphingosine (lyso-Gb3, a metabolite of Gb3)
concentrations are elevated in patients with Fabry disease"* (chunk
`fda-elfabrio#1`).

Output at the time of writing (abridged):

```text
BM25 (lexical search), top 5
   1   13.917  medlineplus-small-fiber-neuropathy#0    Small fiber neuropathy is a condition...
   2    9.833  fda-elfabrio#3                          ERT-experienced for at least 104 weeks...
   3    9.651  fda-elfabrio#1                          12.2 Pharmacodynamics Plasma globotriaosylsphingosine...
   ...
Dense search, top 5
   1    0.723  medlineplus-small-fiber-neuropathy#0    Small fiber neuropathy is a condition...
   2    0.713  medlineplus-hereditary-sensory-...#2    The signs and symptoms of HSAN2...
   3    0.699  fda-elfabrio#1                          12.2 Pharmacodynamics Plasma globotriaosylsphingosine...
   ...
Hybrid: RRF over both, top 5
   1    1.000  medlineplus-small-fiber-neuropathy#0    Small fiber neuropathy is a condition...
   2    0.533  medlineplus-hereditary-sensory-...#2    The signs and symptoms of HSAN2...
   3    0.500  fda-elfabrio#1                          12.2 Pharmacodynamics Plasma globotriaosylsphingosine...
   ...
Hybrid + cross-encoder rerank of the fused top 10, top 3
   1   -1.798  fda-elfabrio#1                          12.2 Pharmacodynamics Plasma globotriaosylsphingosine...
   2   -2.966  medlineplus-small-fiber-neuropathy#0    Small fiber neuropathy is a condition...
   3   -3.241  fda-elfabrio#3                          ERT-experienced for at least 104 weeks...
```

What it shows:

- **The scores are not comparable across retrievers.** BM25 returns
  unbounded scores (13.9, 9.8, ...), while dense search returns cosine
  similarities around 0.7. RRF ignores both and uses only ranks.
- **Fusion helps, but it does not decide.** The symptom half of the query
  pulls in neuropathies that really do cause burning pain in the hands and
  feet. Both retrievers rank them highly, so RRF keeps them on top.
- **The reranker reads query and chunk together.** It moves the only chunk
  that matches *both* halves of the query, the lyso-Gb3 one, from rank 3 to
  rank 1.

The output is real, not staged, and it differs from the article's
illustrative example in one place. The MedlinePlus sentence about Fabry pain
("episodes of pain, particularly in the hands and feet (acroparesthesias)")
lists seven other symptoms in the same sentence, and it never reaches the top
5 of any list. Better chunking or a domain-tuned embedding model are the usual
next steps. Neither is needed to show the mechanism.

## How the code is laid out

| File | What it does |
|---|---|
| [src/hybrid_search/corpus.py](src/hybrid_search/corpus.py) | Fetches documents, reads and writes `data/documents/*.md`, chunks by paragraph, and splits paragraphs longer than 80 words at sentence boundaries |
| [src/hybrid_search/models.py](src/hybrid_search/models.py) | Embedding model (`BAAI/bge-small-en-v1.5`), BM25 (`Qdrant/bm25`), cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`) |
| [src/hybrid_search/stores/qdrant.py](src/hybrid_search/stores/qdrant.py) | One collection with a dense vector and a BM25 sparse vector per chunk; hybrid search as a single query with two `prefetch`es and server-side RRF |
| [src/hybrid_search/pipeline.py](src/hybrid_search/pipeline.py) | Indexing, the searches and the rerank, written against the `Store` protocol |
| [src/hybrid_search/cli.py](src/hybrid_search/cli.py) | `fetch`, `index`, `search` |

The vector index runs every search itself. The query vector goes in, and
chunk IDs with scores come out. For BM25, the `Qdrant/bm25` model supplies
the term weights and Qdrant applies IDF across the collection
(`Modifier.IDF`). That makes the sparse dot product a BM25 score, and new
chunks don't need re-weighting.

**Adding another vector store:** implement the `Store` protocol in
[src/hybrid_search/stores/\_\_init\_\_.py](src/hybrid_search/stores/__init__.py)
next to `qdrant.py`. Nothing else changes.

## Not included

Generation by an LLM, evaluation (precision, recall) and production concerns
such as batching, filtering or persistence settings. This repo only shows the
retrieval step.
