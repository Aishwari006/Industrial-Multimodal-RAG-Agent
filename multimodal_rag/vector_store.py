"""
vector_store.py
---------------
Thin factory + wrapper around FAISS / Chroma vector stores.

A *single* vector store holds both text chunks and image summaries,
distinguished only by their ``content_type`` metadata field.
No raw image data is stored here.
"""

import logging
from typing import List, Literal

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_ollama import OllamaEmbeddings

from .config import RAGConfig

logger = logging.getLogger(__name__)

Backend = Literal["faiss", "chroma"]


class VectorStoreManager:
    """
    Creates and manages an in-memory vector store for a single PDF session.

    Parameters
    ----------
    config : RAGConfig
        Pipeline configuration (backend choice, embedding model, retrieval k …).
    """

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._embeddings = OllamaEmbeddings(
            model=config.embedding_model,
            base_url=config.ollama_base_url,
        )
        self._store: VectorStore | None = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def build(self, documents: List[Document]) -> None:
        """
        Embed *documents* and populate the vector store.

        Parameters
        ----------
        documents : List[Document]
            Pre-processed Documents from the ingestion layer.

        Raises
        ------
        ValueError
            If *documents* is empty.
        ImportError
            If the requested backend package is not installed.
        """
        if not documents:
            raise ValueError("Cannot build a vector store from an empty document list.")

        backend: Backend = self._config.vector_store_backend.lower()  # type: ignore[assignment]
        logger.info(
            "Building %s vector store with %d documents …", backend, len(documents)
        )

        if backend == "faiss":
            self._store = self._build_faiss(documents)
        elif backend == "chroma":
            self._store = self._build_chroma(documents)
        else:
            raise ValueError(
                f"Unknown vector_store_backend '{backend}'. Choose 'faiss' or 'chroma'."
            )

        logger.info("Vector store ready.")

    def retrieve(self, query: str) -> List[Document]:
        """
        Run a similarity search against the vector store.

        Parameters
        ----------
        query : str
            The user's question (or a reformulated version of it).

        Returns
        -------
        List[Document]
            Top-k most relevant Documents.

        Raises
        ------
        RuntimeError
            If :meth:`build` has not been called yet.
        """
        if self._store is None:
            raise RuntimeError(
                "Vector store has not been built. Call build() first."
            )
        results = self._store.similarity_search(query, k=self._config.retrieval_k)
        logger.debug("Retrieved %d documents for query.", len(results))
        return results

    @property
    def is_ready(self) -> bool:
        """Return ``True`` if the vector store has been built."""
        return self._store is not None

    # ------------------------------------------------------------------ #
    # Backend builders                                                     #
    # ------------------------------------------------------------------ #

    def _build_faiss(self, documents: List[Document]) -> VectorStore:
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError as exc:
            raise ImportError(
                "langchain-community is required for the FAISS backend. "
                "Install it with: pip install langchain-community faiss-cpu"
            ) from exc
        return FAISS.from_documents(documents, self._embeddings)

    def _build_chroma(self, documents: List[Document]) -> VectorStore:
        try:
            from langchain_chroma import Chroma
        except ImportError as exc:
            raise ImportError(
                "langchain-chroma is required for the Chroma backend. "
                "Install it with: pip install langchain-chroma"
            ) from exc
        return Chroma.from_documents(documents, self._embeddings)
