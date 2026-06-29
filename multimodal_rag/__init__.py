"""
multimodal_rag
==============
A modular, LangChain-based Multimodal Context Retrieval pipeline for PDFs. 
Designed to be called as a tool from an external agent (e.g., LangGraph) 
without any tight coupling to the orchestrator itself.

Quick start
-----------
    from multimodal_rag import retrieve_context

    context = retrieve_context(
        pdf_path="report.pdf",
        query="What is the revenue trend?",
        thread_id="session-001",
    )
"""

from .main import retrieve_context, reset_session
from .config import RAGConfig, cfg

__all__ = [
    "retrieve_context",
    "reset_session",
    "RAGConfig",
    "cfg",
]