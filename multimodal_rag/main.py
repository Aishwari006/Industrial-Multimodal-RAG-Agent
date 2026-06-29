"""
main.py
-------
Public API for the Multimodal Retrieval pipeline.

This is the **only** file an external caller (e.g., a LangGraph tool) needs
to import. Everything else is an implementation detail.

Usage
-----
    from multimodal_rag.main import retrieve_context

    context = retrieve_context(
        pdf_path="report.pdf",
        query="Explain the revenue trend.",
        thread_id="session-001",
    )
    print(context)
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import RAGConfig, cfg
from .pipeline import RAGPipeline
from .session_manager import SessionManager

# ── Module-level singletons ──────────────────────────────────────────────────
# A single SessionManager and shared config are reused across calls.
# Callers that need custom configuration should use the lower-level API directly.
_session_manager: Optional[SessionManager] = None
_active_config: Optional[RAGConfig] = None


def _get_session_manager(config: RAGConfig) -> SessionManager:
    """Return (or initialise) the module-level SessionManager."""
    global _session_manager, _active_config
    if _session_manager is None or config is not _active_config:
        _session_manager = SessionManager(config)
        _active_config = config
    return _session_manager


# ── Public function ──────────────────────────────────────────────────────────


def retrieve_context(
    pdf_path: str,
    query: str,
    thread_id: str,
    *,
    config: Optional[RAGConfig] = None,
) -> str:
    """
    Retrieve relevant document context for a given query.

    The PDF is ingested **once** per ``thread_id``. Subsequent calls with
    the same ``thread_id`` reuse the existing vector store, enabling fast
    multi-turn context retrieval without re-processing the file.

    Parameters
    ----------
    pdf_path  : str
        Absolute or relative path to the PDF file.
    query     : str
        The user's question or search query for the retriever.
    thread_id : str
        Unique identifier for the session/conversation (e.g., from LangGraph).
    config    : RAGConfig, optional
        Override the default configuration.

    Returns
    -------
    str
        The raw, formatted context string retrieved from the document.

    Raises
    ------
    FileNotFoundError
        If *pdf_path* does not exist.

    Examples
    --------
    >>> context = retrieve_context(
    ...     pdf_path="annual_report.pdf",
    ...     query="What was the net revenue in Q3?",
    ...     thread_id="user-42-session-1",
    ... )
    >>> print(context)
    """
    effective_config = config or cfg
    manager = _get_session_manager(effective_config)
    session = manager.get_or_create(thread_id)
    
    pipeline = RAGPipeline(config=effective_config, session=session)
    pipeline.ensure_indexed(pdf_path)
    
    # Delegate to the newly refactored pipeline method
    return pipeline.retrieve_context(query)


def reset_session(thread_id: str, *, config: Optional[RAGConfig] = None) -> None:
    """
    Delete a session and free its vector store resources.

    The next call to :func:`retrieve_context` with the same ``thread_id`` will
    start a fresh session (re-ingesting the PDF).

    Parameters
    ----------
    thread_id : str
    config    : RAGConfig, optional
    """
    effective_config = config or cfg
    manager = _get_session_manager(effective_config)
    manager.delete(thread_id)


# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)