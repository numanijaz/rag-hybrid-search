from hybrid_search.corpus import (
    Document,
    chunk_documents,
    extract_subsection,
    load_documents,
    read_document,
    split_paragraph,
    write_document,
)


def test_short_paragraph_stays_whole():
    assert split_paragraph("One. Two.", max_words=10) == ["One. Two."]


def test_long_paragraph_is_packed_by_sentence():
    paragraph = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
    assert split_paragraph(paragraph, max_words=6) == [
        "Alpha beta gamma. Delta epsilon zeta.",
        "Eta theta iota.",
    ]


def test_extract_subsection_skips_cross_references():
    section = (
        "Highlights ( 2.5 , 5.8 ) More text. 5.8 Palmar-Plantar Syndrome "
        "PPES can occur [see (2.5) ] . 5.9 Myelosuppression Monitor counts."
    )
    assert extract_subsection(section, "5.8") == (
        "5.8 Palmar-Plantar Syndrome PPES can occur [see (2.5) ] ."
    )


def test_document_round_trip(tmp_path):
    doc = Document(
        id="medlineplus-x",
        title="X",
        source="Somewhere",
        url="https://example.org/x",
        license="Public domain",
        retrieved="2026-10-01",
        paragraphs=["First paragraph.", "Second paragraph."],
    )
    assert read_document(write_document(doc, tmp_path)) == doc


def test_committed_corpus_has_sources_and_chunks():
    documents = load_documents()
    assert documents, "run `uv run hybrid-search fetch`"
    assert all(d.url.startswith("https://") and d.license for d in documents)
    chunks = chunk_documents(documents)
    assert len({c.id for c in chunks}) == len(chunks)
    assert not any(c.text.startswith("#") for c in chunks)
