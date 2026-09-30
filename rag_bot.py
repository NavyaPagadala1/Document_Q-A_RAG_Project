"""A small, source-grounded document Q&A pipeline."""

from __future__ import annotations
import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COLLECTION_NAME = "rag_documents_gemini"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 180
EMBED_BATCH_SIZE = 64


@dataclass(frozen=True)
class SourceText:
    source: str
    text: str
    page: int | None = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


def clean_text(text: str) -> str:
    """Normalize extracted text and remove standalone page-number lines."""
    lines = [line.strip() for line in text.replace("\x00", " ").splitlines()]
    lines = [line for line in lines if not re.fullmatch(r"(?:page\s*)?\d+", line, re.IGNORECASE)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def load_documents(data_dir: Path) -> list[SourceText]:
    """Read supported files from a folder; each PDF page keeps its page number."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Document folder not found: {data_dir}")

    documents: list[SourceText] = []
    paths = sorted(path for path in data_dir.rglob("*") if path.suffix.lower() in {".pdf", ".txt"})
    for path in paths:
        relative_name = path.relative_to(data_dir).as_posix()
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            for page_number, page in enumerate(PdfReader(path).pages, start=1):
                text = clean_text(page.extract_text() or "")
                if text:
                    documents.append(SourceText(relative_name, text, page_number))
        else:
            text = clean_text(path.read_text(encoding="utf-8"))
            if text:
                documents.append(SourceText(relative_name, text))

    if not documents:
        raise ValueError(f"No readable PDF or TXT documents found in {data_dir}")
    return documents


def split_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character windows, preferring paragraph boundaries."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be between zero and chunk_size")

    text = clean_text(text)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind("\n\n", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        if end <= start:
            end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def make_chunks(
    documents: list[SourceText],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        for index, text in enumerate(split_text(document.text, chunk_size, overlap), start=1):
            location = f"page:{document.page}" if document.page is not None else f"section:{index}"
            digest = hashlib.sha256(f"{document.source}|{location}|{text}".encode("utf-8")).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=digest,
                    text=text,
                    metadata={
                        "source": document.source,
                        "page": document.page if document.page is not None else -1,
                        "section": "" if document.page is not None else f"Section {index}",
                        "chunk": index,
                    },
                )
            )
    return chunks


def _gemini_client():
    import os

    from dotenv import load_dotenv
    from google import genai

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Add it to .env or the environment.")
    return genai.Client(api_key=api_key)


def _embedding_model() -> str:
    import os

    return os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")


def embed_texts(
    texts: list[str],
    client: Any | None = None,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """Embed multiple texts per request, splitting only at a bounded batch size."""
    if not texts:
        return []
    client = client or _gemini_client()
    vectors: list[list[float]] = []
    for offset in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[offset : offset + EMBED_BATCH_SIZE]
        response = client.models.embed_content(
            model=_embedding_model(),
            contents=batch,
            config={"task_type": task_type},
        )
        vectors.extend(item.values for item in response.embeddings)
    return vectors


def _vector_store(persist_dir: Path):
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    return client, client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def ingest(data_dir: Path, persist_dir: Path) -> int:
    documents = load_documents(data_dir)
    chunks = make_chunks(documents)
    if not chunks:
        raise ValueError("No text could be chunked from the documents")

    client = _gemini_client()
    vectors = embed_texts([chunk.text for chunk in chunks], client)
    persist_dir.mkdir(parents=True, exist_ok=True)
    _, collection = _vector_store(persist_dir)
    old_ids = collection.get(include=[])["ids"]
    if old_ids:
        collection.delete(ids=old_ids)
    for offset in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch_chunks = chunks[offset : offset + EMBED_BATCH_SIZE]
        collection.add(
            ids=[chunk.chunk_id for chunk in batch_chunks],
            documents=[chunk.text for chunk in batch_chunks],
            metadatas=[chunk.metadata for chunk in batch_chunks],
            embeddings=vectors[offset : offset + EMBED_BATCH_SIZE],
        )
    print(f"Indexed {len(chunks)} chunks from {len({doc.source for doc in documents})} documents.")
    return len(chunks)


def _format_location(metadata: dict[str, Any]) -> str:
    page = metadata.get("page", -1)
    if page not in (-1, None, ""):
        return f"page {page}"
    section = metadata.get("section")
    return str(section) if section else f"chunk {metadata.get('chunk', '?')}"


def build_context(documents: list[str], metadatas: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
    """Label retrieved passages consistently so the model can cite each one."""
    excerpts: list[str] = []
    sources: list[dict[str, str]] = []
    for index, (text, metadata) in enumerate(zip(documents, metadatas), start=1):
        reference = f"S{index}"
        source = str(metadata.get("source", "unknown source"))
        location = _format_location(metadata)
        excerpts.append(f"[{reference}] Source: {source}, {location}\n{text}")
        sources.append({"reference": reference, "source": source, "location": location})
    return "\n\n".join(excerpts), sources


def generate_grounded_answer(question: str, context: str, client: Any) -> str:
    import os

    chat = client.chats.create(
        model=os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash"),
        config={
            "temperature": 0,
            "system_instruction": (
                "Answer the question using only the supplied source excerpts. "
                "Cite every factual claim with its bracketed source label, such as [S1]. "
                "If the excerpts do not contain enough information, say so clearly; do not guess."
            ),
        },
    )
    response = chat.send_message(f"Question: {question}\n\nSource excerpts:\n{context}")
    return response.text or ""


def answer_question(question: str, persist_dir: Path, top_k: int = 4) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("Question cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    client, collection = _vector_store(persist_dir)
    if collection.count() == 0:
        raise ValueError("The vector database is empty. Run the ingest command first.")

    gemini_client = _gemini_client()
    query_vector = embed_texts([question], gemini_client, task_type="RETRIEVAL_QUERY")[0]
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    context, sources = build_context(documents, metadatas)

    answer = generate_grounded_answer(question, context, gemini_client)
    return {"answer": answer, "sources": sources}


def _default_persist_dir() -> Path:
    import os

    return Path(os.getenv("CHROMA_PERSIST_DIR", ".chroma"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask grounded questions about your local PDF and TXT documents.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subparsers.add_parser("ingest", help="Load, chunk, embed, and index documents")
    ingest_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    ingest_parser.add_argument("--db-dir", type=Path, default=_default_persist_dir())

    ask_parser = subparsers.add_parser("ask", help="Ask one question")
    ask_parser.add_argument("question", nargs="+")
    ask_parser.add_argument("--top-k", type=int, default=4)
    ask_parser.add_argument("--db-dir", type=Path, default=_default_persist_dir())

    chat_parser = subparsers.add_parser("chat", help="Ask multiple questions in a session")
    chat_parser.add_argument("--top-k", type=int, default=4)
    chat_parser.add_argument("--db-dir", type=Path, default=_default_persist_dir())

    args = parser.parse_args()
    try:
        if args.command == "ingest":
            ingest(args.data_dir, args.db_dir)
            return 0
        if args.command == "ask":
            result = answer_question(" ".join(args.question), args.db_dir, args.top_k)
            print(result["answer"])
            print("\nSources:")
            for source in result["sources"]:
                print(f"[{source['reference']}] {source['source']} ({source['location']})")
            return 0
        print("Document Q&A is ready. Type 'exit' to quit.")
        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() in {"exit", "quit"}:
                return 0
            if not question:
                continue
            result = answer_question(question, args.db_dir, args.top_k)
            print(f"\n{result['answer']}\n")
            for source in result["sources"]:
                print(f"[{source['reference']}] {source['source']} ({source['location']})")
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
