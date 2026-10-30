import ast
import hashlib
import io
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, Protocol

import chromadb
import requests
from pypdf import PdfReader


OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
RESULT_COUNT = 4
DISTANCE_THRESHOLD = 0.9
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "to", "what", "when",
    "where", "which", "who", "why", "with",
}
DATA_DIR = Path(__file__).parent / "data"


class LLMClient(Protocol):
    """The small interface shared by local and hosted language models."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    def generate(self, prompt: str) -> str:
        ...


class OllamaClient:
    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for start in range(0, len(texts), 32):
            batch = texts[start : start + 32]
            try:
                response = requests.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={"model": OLLAMA_EMBED_MODEL, "input": batch},
                    timeout=120,
                )
                response.raise_for_status()
            except requests.RequestException as error:
                raise RuntimeError(
                    "Could not reach Ollama for embeddings. Start Ollama and "
                    f"install the '{OLLAMA_EMBED_MODEL}' model."
                ) from error
            result = response.json().get("embeddings")
            if not isinstance(result, list) or len(result) != len(batch):
                raise RuntimeError("Ollama returned an unexpected embedding response.")
            embeddings.extend(result)
        return embeddings

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_CHAT_MODEL,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=180,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(
                "Could not reach Ollama to generate an answer. Start Ollama and "
                f"install the '{OLLAMA_CHAT_MODEL}' model."
            ) from error
        answer = response.json().get("response")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Ollama returned an empty or unexpected answer.")
        return answer.strip()


class GeminiClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Set GEMINI_API_KEY to use Gemini, or set LLM_PROVIDER=ollama."
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    f"{GEMINI_API_URL}/models/{GEMINI_EMBED_MODEL}:embedContent",
                    headers={"x-goog-api-key": self.api_key},
                    json={
                        "model": f"models/{GEMINI_EMBED_MODEL}",
                        "content": {"parts": [{"text": text}]},
                    },
                    timeout=60,
                )
                response.raise_for_status()
            except requests.RequestException as error:
                raise RuntimeError(
                    "Gemini could not create an embedding. Check your API key "
                    "and Gemini API access."
                ) from error
            values = response.json().get("embedding", {}).get("values")
            if not isinstance(values, list):
                raise RuntimeError("Gemini returned an unexpected embedding response.")
            embeddings.append(values)
        return embeddings

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{GEMINI_API_URL}/models/{GEMINI_CHAT_MODEL}:generateContent",
                headers={"x-goog-api-key": self.api_key},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=180,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(
                "Gemini could not generate an answer. Check your API key and "
                "Gemini API access."
            ) from error
        candidates = response.json().get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no answer.")
        parts = candidates[0].get("content", {}).get("parts", [])
        answer = "\n".join(part.get("text", "") for part in parts).strip()
        if not answer:
            raise RuntimeError("Gemini returned an empty or unexpected answer.")
        return answer


def get_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider not in {"ollama", "gemini"}:
        raise ValueError("LLM_PROVIDER must be either 'ollama' or 'gemini'.")
    return provider


@lru_cache(maxsize=2)
def get_llm_client(provider: str) -> LLMClient:
    if provider == "ollama":
        return OllamaClient()
    if provider == "gemini":
        return GeminiClient()
    raise ValueError(f"Unknown LLM provider: {provider}")


@lru_cache(maxsize=2)
def get_chroma_client():
    chroma_host = os.getenv("CHROMA_HOST")
    if chroma_host:
        return chromadb.HttpClient(
            host=chroma_host,
            port=int(os.getenv("CHROMA_PORT", "8000")),
        )
    path = os.getenv("CHROMA_DATA_DIR", str(DATA_DIR / "chroma"))
    return chromadb.PersistentClient(path=path)


def get_collection(provider: Optional[str] = None):
    provider = provider or get_provider()
    return get_chroma_client().get_or_create_collection(
        name=f"devdocs_{provider}",
        metadata={"hnsw:space": "cosine"},
    )


def read_document(filename: str, file_bytes: bytes) -> str:
    """Turn a supported uploaded file into plain text."""
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif extension in {".txt", ".md", ".py"}:
        text = file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError("Please upload a PDF, Markdown, text, or Python file.")
    if not text.strip():
        raise ValueError(f"No readable text was found in {filename}.")
    return text


def split_into_chunks(text: str) -> list[str]:
    """Split prose into overlapping pieces."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def make_document_chunks(filename: str, text: str) -> list[dict]:
    """Create chunks with line references; Python functions stay together."""
    if Path(filename).suffix.lower() == ".py":
        try:
            tree = ast.parse(text)
            functions = [
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.end_lineno is not None
            ]
        except SyntaxError:
            functions = []

        if functions:
            lines = text.splitlines()
            return [
                {
                    "text": "\n".join(lines[node.lineno - 1 : node.end_lineno]),
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "kind": "function",
                }
                for node in sorted(functions, key=lambda item: item.lineno)
            ]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(
                {
                    "text": chunk,
                    "start_line": text.count("\n", 0, start) + 1,
                    "end_line": text.count("\n", 0, end) + 1,
                    "kind": "text",
                }
            )
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def index_document(filename: str, file_bytes: bytes) -> int:
    """Read, split, embed, and save one document."""
    safe_filename = Path(filename).name
    text = read_document(safe_filename, file_bytes)
    chunks = make_document_chunks(safe_filename, text)
    if not chunks:
        raise ValueError(f"No readable text was found in {safe_filename}.")

    provider = get_provider()
    embeddings = get_llm_client(provider).embed([chunk["text"] for chunk in chunks])
    collection = get_collection(provider)
    existing = collection.get(where={"source": safe_filename})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    ids = [
        hashlib.sha256(
            f"{provider}:{safe_filename}:{number}:{chunk['text']}".encode("utf-8")
        ).hexdigest()
        for number, chunk in enumerate(chunks, start=1)
    ]
    collection.add(
        ids=ids,
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "source": safe_filename,
                "section": number,
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "kind": chunk["kind"],
            }
            for number, chunk in enumerate(chunks, start=1)
        ],
    )
    return len(chunks)


def _lexical_score(question: str, text: str) -> float:
    query_terms = {
        term.lower()
        for term in re.findall(r"\w+", question)
        if len(term) > 1 and term.lower() not in STOP_WORDS
    }
    if not query_terms:
        return 0.0
    document_terms = {term.lower() for term in re.findall(r"\w+", text)}
    return len(query_terms & document_terms) / len(query_terms)


def search_documents(question: str) -> list[dict]:
    """Combine vector similarity with lexical keyword search using rank fusion."""
    provider = get_provider()
    collection = get_collection(provider)
    total = collection.count()
    if total == 0:
        raise ValueError("Add at least one document before asking a question.")

    records = collection.get(include=["documents", "metadatas"])
    query_embedding = get_llm_client(provider).embed([question])[0]
    vector_matches = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(total, RESULT_COUNT * 4),
        include=["documents", "metadatas", "distances"],
    )

    ranked = {}
    for rank, (record_id, document, metadata, distance) in enumerate(
        zip(
            vector_matches["ids"][0],
            vector_matches["documents"][0],
            vector_matches["metadatas"][0],
            vector_matches["distances"][0],
        ),
        start=1,
    ):
        ranked[record_id] = {
            "text": document,
            **metadata,
            "distance": distance,
            "score": 1 / (60 + rank),
        }

    lexical_matches = []
    for record_id, document, metadata in zip(
        records["ids"], records["documents"], records["metadatas"]
    ):
        score = _lexical_score(question, document)
        if score:
            lexical_matches.append((record_id, document, metadata, score))
    lexical_matches.sort(key=lambda result: result[3], reverse=True)

    for rank, (record_id, document, metadata, score) in enumerate(
        lexical_matches[: RESULT_COUNT * 4], start=1
    ):
        if record_id not in ranked:
            ranked[record_id] = {
                "text": document,
                **metadata,
                "distance": 1.0,
                "score": 0.0,
            }
        ranked[record_id]["score"] += score / (60 + rank)
        ranked[record_id]["lexical_score"] = score

    results = sorted(ranked.values(), key=lambda result: result["score"], reverse=True)
    relevant = [
        result
        for result in results
        if result.get("distance", 1.0) <= DISTANCE_THRESHOLD
        or result.get("lexical_score", 0.0) >= 0.25
    ]
    return relevant[:RESULT_COUNT]


def answer_question(question: str) -> tuple[str, list[dict]]:
    """Retrieve supporting text and answer only when a relevant match exists."""
    if not question.strip():
        raise ValueError("Type a question first.")
    matches = search_documents(question)
    if not matches:
        return "I couldn't find that information in the indexed documents.", []

    context = "\n\n".join(
        f"[{item['source']}:{item['start_line']}-{item['end_line']}]\n{item['text']}"
        for item in matches
    )
    prompt = f"""Answer the question using only the supplied context.
If the context does not contain the answer, say you could not find it in the indexed documents.
Do not make up facts. Keep the answer clear and concise. Cite supporting files and line numbers.

Context:
{context}

Question: {question}
Answer:"""
    answer = get_llm_client(get_provider()).generate(prompt)
    citations = [
        {
            "source": item["source"],
            "start_line": item["start_line"],
            "end_line": item["end_line"],
            "kind": item["kind"],
        }
        for item in matches
    ]
    return answer, citations


def get_indexed_sources() -> list[str]:
    collection = get_collection()
    records = collection.get(include=["metadatas"])
    return sorted({metadata["source"] for metadata in records["metadatas"]})


def remove_document(filename: str) -> None:
    collection = get_collection()
    existing = collection.get(where={"source": Path(filename).name})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
