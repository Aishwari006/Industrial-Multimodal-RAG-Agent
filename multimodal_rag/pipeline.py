"""
pipeline.py
-----------
Core context retrieval pipeline for a single session.

Flow
~~~~
User question
    │
    ▼
Retriever  →  Relevant Documents
    │
    ▼
Format Context
    │
    ▼
Return context string
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import RAGConfig
from .ingestion import PDFIngester
from .retrieval import Retriever
from .session_manager import Session

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Handles ingestion (once) and context retrieval for one session.

    Parameters
    ----------
    config  : RAGConfig
    session : Session   – the active session carrying the vector store.
    """

    def __init__(self, config: RAGConfig, session: Session) -> None:
        self._config = config
        self._session = session
        self._ingester = PDFIngester(config)
        # LLM completely removed - LangGraph orchestrates the LLM now

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def ensure_indexed(self, pdf_path: str) -> None:
        """
        Ingest *pdf_path* into the session's vector store if not done yet.

        If the session is already indexed (same or different PDF), this is a
        no-op — the existing index is reused.  To force re-indexing with a
        new PDF, delete the session via ``SessionManager.delete()`` first.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF to ingest.
        """
        if self._session.indexed:
            logger.info(
                "Session '%s' already indexed ('%s'). Skipping ingestion.",
                self._session.thread_id,
                self._session.filename,
            )
            return

        filename = Path(pdf_path).name
        logger.info(
            "Indexing '%s' for session '%s' …",
            filename,
            self._session.thread_id,
        )

        documents = self._ingester.ingest(pdf_path)
        self._session.vector_store.build(documents)
        self._session.indexed = True
        self._session.filename = filename
        logger.info("Indexing complete: %d documents stored.", len(documents))

    def retrieve_context(self, question: str) -> str:
        """
        Retrieve relevant context for *question* using the session's vector store.

        Parameters
        ----------
        question : str
            The user's question or search query.

        Returns
        -------
        str
            The formatted raw context from the vector store.

        Raises
        ------
        RuntimeError
            If the session has not been indexed yet.
        """
        if not self._session.indexed:
            raise RuntimeError(
                "PDF has not been ingested for this session. "
                "Call ensure_indexed() first."
            )

        # 1. Retrieve relevant documents
        retriever = Retriever(self._session.vector_store)
        docs = retriever.retrieve(question)
        context_block = retriever.format_context(docs)

        logger.debug(
            "Context block (%d chars) retrieved for session '%s'.",
            len(context_block),
            self._session.thread_id,
        )
        
        # 2. Return the pure context string directly to the ToolNode
        return context_block