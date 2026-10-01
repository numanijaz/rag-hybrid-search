"""Fetch the public-domain corpus, store it as Markdown, and cut it into chunks."""

import datetime
import json
import re
import tomllib
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"

MEDLINEPLUS_LICENSE = "Public domain (https://medlineplus.gov/about/using/usingcontent/)"
OPENFDA_LICENSE = "Public domain, CC0 1.0 (https://open.fda.gov/terms/)"

# Chunks longer than this are split at sentence boundaries.
MAX_CHUNK_WORDS = 80


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    source: str
    url: str
    license: str
    retrieved: str
    paragraphs: list[str]


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    url: str
    text: str


# --- fetching ---------------------------------------------------------------


class _Paragraphs(HTMLParser):
    """Collect the text of <p> and <li> elements as separate paragraphs."""

    def __init__(self) -> None:
        super().__init__()
        self.paragraphs: list[str] = []
        self._buf: list[str] = []

    def handle_data(self, data: str) -> None:
        self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("p", "li"):
            text = " ".join("".join(self._buf).split())
            if text:
                self.paragraphs.append(text)
            self._buf = []


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def fetch_medlineplus(kind: str, slug: str, today: str) -> Document:
    """A condition or gene summary from MedlinePlus Genetics (data files API)."""
    data = _get_json(f"https://medlineplus.gov/download/genetics/{kind}/{slug}.json")
    parser = _Paragraphs()
    for item in data["text-list"]:
        parser.feed(item["text"]["html"])
    title = data["name"] if kind == "condition" else f"{data['gene-symbol']} gene"
    return Document(
        id=f"medlineplus-{slug}",
        title=title,
        source="MedlinePlus Genetics, U.S. National Library of Medicine",
        # Condition files use "ghr_page", gene files "ghr-page".
        url=data.get("ghr_page") or data["ghr-page"],
        license=MEDLINEPLUS_LICENSE,
        retrieved=today,
        paragraphs=parser.paragraphs,
    )


def extract_subsection(text: str, number: str) -> str:
    """Cut label subsection `number` (e.g. "5.8") out of a flattened section.

    openFDA returns each label section as one string; subsections start with
    their number followed by a capitalised heading ("5.8 Palmar-Plantar ...").
    """
    match = re.search(
        rf"(?<![\d.]){re.escape(number)} [A-Z].*?(?= \d+\.\d+ [A-Z]|$)", text, re.S
    )
    if not match:
        raise ValueError(f"subsection {number} not found")
    return match.group(0).strip()


def fetch_fda_label(
    brand: str, sections: list[str], today: str, subsection: str | None = None
) -> Document:
    """Selected sections of a drug label from openFDA."""
    query = urllib.parse.quote(f'openfda.brand_name:"{brand}"')
    data = _get_json(f"https://api.fda.gov/drug/label.json?search={query}&limit=1")
    label = data["results"][0]
    paragraphs = []
    for section in sections:
        text = " ".join(label[section][0].split())
        paragraphs.append(extract_subsection(text, subsection) if subsection else text)
    return Document(
        id=f"fda-{brand.lower()}",
        title=f"{label['openfda']['brand_name'][0]} drug label",
        source=f"FDA drug label via openFDA (set ID {label['set_id']}, "
        f"version {label['effective_time']})",
        url=f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={label['set_id']}",
        license=OPENFDA_LICENSE,
        retrieved=today,
        paragraphs=paragraphs,
    )


def fetch_all(sources_file: Path = DATA_DIR / "sources.toml") -> list[Document]:
    sources = tomllib.loads(sources_file.read_text())
    today = datetime.date.today().isoformat()
    documents = [
        fetch_medlineplus(s["kind"], s["slug"], today) for s in sources["medlineplus"]
    ]
    documents += [
        fetch_fda_label(s["brand"], s["sections"], today, s.get("subsection"))
        for s in sources["fda_label"]
    ]
    return documents


# --- storing and loading ----------------------------------------------------

_FIELDS = ("title", "source", "url", "license", "retrieved")


def write_document(doc: Document, directory: Path = DOCUMENTS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    header = "\n".join(f"{field}: {getattr(doc, field)}" for field in _FIELDS)
    body = "\n\n".join(doc.paragraphs)
    path = directory / f"{doc.id}.md"
    path.write_text(f"---\n{header}\n---\n\n# {doc.title}\n\n{body}\n")
    return path


def read_document(path: Path) -> Document:
    _, header, body = path.read_text().split("---\n", 2)
    meta = dict(line.split(": ", 1) for line in header.strip().splitlines())
    paragraphs = [
        p.strip() for p in body.split("\n\n") if p.strip() and not p.strip().startswith("#")
    ]
    return Document(id=path.stem, paragraphs=paragraphs, **{f: meta[f] for f in _FIELDS})


def load_documents(directory: Path = DOCUMENTS_DIR) -> list[Document]:
    return [read_document(p) for p in sorted(directory.glob("*.md"))]


def write_sources_index(documents: list[Document], path: Path = DATA_DIR / "SOURCES.md") -> None:
    rows = "\n".join(
        f"| [{d.title}](documents/{d.id}.md) | [{d.source}]({d.url}) | {d.retrieved} |"
        for d in documents
    )
    path.write_text(
        "# Sources\n\n"
        "Generated by `uv run hybrid-search fetch`. Licences: MedlinePlus Genetics "
        "summaries are public domain\n(<https://medlineplus.gov/about/using/usingcontent/>); "
        "openFDA data is public domain, CC0 1.0\n(<https://open.fda.gov/terms/>).\n\n"
        "| Document | Source | Retrieved |\n|---|---|---|\n"
        f"{rows}\n"
    )


# --- chunking ---------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def split_paragraph(paragraph: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """Keep short paragraphs whole; pack long ones sentence by sentence."""
    if len(paragraph.split()) <= max_words:
        return [paragraph]
    pieces: list[str] = []
    current: list[str] = []
    for sentence in _SENTENCE_END.split(paragraph):
        if current and len(" ".join(current + [sentence]).split()) > max_words:
            pieces.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    chunks = []
    for doc in documents:
        texts = [piece for p in doc.paragraphs for piece in split_paragraph(p)]
        chunks += [
            Chunk(id=f"{doc.id}#{n}", doc_id=doc.id, title=doc.title, url=doc.url, text=t)
            for n, t in enumerate(texts)
        ]
    return chunks
