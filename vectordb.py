from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

try:
    from langchain_community.vectorstores import FAISS
except ModuleNotFoundError:  # pragma: no cover
    FAISS = None  # type: ignore[assignment]

try:
    from langchain_openai import OpenAIEmbeddings
except ModuleNotFoundError:  # pragma: no cover
    OpenAIEmbeddings = None  # type: ignore[assignment]


@dataclass
class RetrievedChunk:
    text: str
    source: str


class VectorDB:
    def __init__(self) -> None:
        self._enabled = False
        self._vs: Any | None = None

        if FAISS is None or OpenAIEmbeddings is None:
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return

        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled and self._vs is not None

    def build_from_texts(self, items: list[tuple[str, str]]) -> None:
        if not self._enabled:
            return

        if FAISS is None or OpenAIEmbeddings is None:
            return

        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for source, text in items:
            for chunk in _chunk_text(text, chunk_size=900, overlap=120):
                texts.append(chunk)
                metadatas.append({"source": source})

        embeddings = OpenAIEmbeddings()
        self._vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        if not self.enabled:
            return []

        docs = self._vs.similarity_search(query, k=k)
        out: list[RetrievedChunk] = []
        for d in docs:
            source = str(d.metadata.get("source", "unknown"))
            out.append(RetrievedChunk(text=str(d.page_content), source=source))
        return out


def _chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    cleaned = text.replace("\r\n", "\n")
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    n = len(cleaned)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(cleaned[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks
