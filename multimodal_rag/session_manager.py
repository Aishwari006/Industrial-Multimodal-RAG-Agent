"""
session_manager.py
------------------
Manages independent RAG sessions keyed by ``thread_id``.

Each session owns:
- a ``VectorStoreManager``   (PDF index)
- an ``indexed`` flag        (avoid re-ingesting the same PDF)
- the ``filename`` of the current PDF

Sessions are stored in-process (a plain dict). This is appropriate for a
single-node deployment; swap the backing store for Redis / DB in production.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from .config import RAGConfig
from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """
    All mutable state associated with one conversation thread.

    Attributes
    ----------
    thread_id    : str                – unique conversation identifier.
    vector_store : VectorStoreManager – index over the current PDF.
    indexed      : bool               – True once the PDF has been ingested.
    filename     : Optional[str]      – basename of the current PDF.
    """

    thread_id: str
    vector_store: VectorStoreManager
    indexed: bool = False
    filename: Optional[str] = None


class SessionManager:
    """
    Registry of active sessions.

    Parameters
    ----------
    config : RAGConfig
        Shared pipeline configuration propagated to every new session.
    """

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._sessions: Dict[str, Session] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def get_or_create(self, thread_id: str) -> Session:
        """
        Return an existing session for *thread_id*, or create a fresh one.

        Parameters
        ----------
        thread_id : str
            Unique identifier for the conversation.

        Returns
        -------
        Session
        """
        if thread_id not in self._sessions:
            logger.info("Creating new session for thread_id='%s'.", thread_id)
            self._sessions[thread_id] = Session(
                thread_id=thread_id,
                vector_store=VectorStoreManager(self._config),
            )
        return self._sessions[thread_id]

    def get(self, thread_id: str) -> Optional[Session]:
        """
        Return the session for *thread_id*, or ``None`` if it does not exist.
        """
        return self._sessions.get(thread_id)

    def delete(self, thread_id: str) -> bool:
        """
        Remove a session and release its resources.

        Parameters
        ----------
        thread_id : str

        Returns
        -------
        bool
            ``True`` if the session existed and was removed, ``False`` otherwise.
        """
        if thread_id in self._sessions:
            del self._sessions[thread_id]
            logger.info("Session '%s' deleted.", thread_id)
            return True
        return False

    def list_sessions(self) -> list[str]:
        """Return all active thread IDs."""
        return list(self._sessions.keys())