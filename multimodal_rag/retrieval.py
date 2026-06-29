"""
retrieval.py
------------
Orchestrates the per-query retrieval step.

For every user question the retriever:
1. Searches the unified vector store (text chunks + image summaries).
2. Formats the retrieved Documents into a context string.

The caller (pipeline.py) returns this formatted string directly 
to the external agent (e.g., LangGraph) as a tool observation.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document

from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class Retriever:
    """
    Wraps a ``VectorStoreManager`` and adds context-formatting utilities.

    Parameters
    ----------
    vector_store : VectorStoreManager
        The session's vector store (must already be built).
    """

    def __init__(self, vector_store: VectorStoreManager) -> None:
        self._vs = vector_store

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve the top-k most relevant Documents for *query*.

        Parameters
        ----------
        query : str
            The user's search query for the vector database.

        Returns
        -------
        List[Document]
        """
        docs = self._vs.retrieve(query)
        logger.debug("Retrieved %d documents.", len(docs))
        return docs

    @staticmethod
    def format_context(docs: List[Document]) -> str:
        """
        Convert a list of Documents into a single formatted string.

        Text chunks and image summaries are clearly labelled so the 
        orchestrating agent can distinguish their origin.

        Parameters
        ----------
        docs : List[Document]

        Returns
        -------
        str
            Formatted context block ready to be returned as a Tool observation.
        """
        if not docs:
            return "No relevant context found in the document."

        sections: List[str] = []
        for i, doc in enumerate(docs, start=1):
            content_type: str = doc.metadata.get("content_type", "unknown")
            page: int | str = doc.metadata.get("page_number", "?")
            source: str = doc.metadata.get("source", "unknown")

            if content_type == "image_summary":
                header = f"[Image Description — {source}, page {page}]"
            else:
                header = f"[Text — {source}, page {page}]"

            sections.append(f"--- Context {i} {header} ---\n{doc.page_content}")

        return "\n\n".join(sections)