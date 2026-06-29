"""
config.py
---------
Central configuration for the Multimodal RAG pipeline.
All tuneable parameters live here so that other modules stay free of magic values.
"""

from dataclasses import dataclass, field


@dataclass
class RAGConfig:
    # ── Ollama model names ──────────────────────────────────────────────────
    chat_model: str = "qwen3.5:4b"          # VLM used for both chat and image summarisation
    embedding_model: str = "nomic-embed-text"

    # ── Ollama base URL ─────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"

    # ── Text splitting ──────────────────────────────────────────────────────
    chunk_size: int = 1_000
    chunk_overlap: int = 200

    # ── Retrieval ───────────────────────────────────────────────────────────
    retrieval_k: int = 6          # number of documents to retrieve per query

    # ── Vector store backend: "faiss" or "chroma" ──────────────────────────
    vector_store_backend: str = "faiss"

    # ── Image summarisation prompt ──────────────────────────────────────────
    image_summary_prompt: str = (
        "You are an expert document analyst. "
        "Describe the image below in detail. "
        "Include all visible text, labels, numbers, trends, and key insights. "
        "Your description will be used for semantic search, so be thorough and precise. "
        "Do not add information that is not visible in the image."
    )


# Module-level singleton so callers can do `from config import cfg`
cfg = RAGConfig()
