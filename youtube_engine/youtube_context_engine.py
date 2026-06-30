"""
youtube_context_engine.py
-------------------------
Handles YouTube transcript fetching, chunking, and FAISS indexing.
"""
import os
import re
import logging
from typing import Dict, List
from dataclasses import dataclass

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Use the Ollama embeddings package
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
logger = logging.getLogger(__name__)
load_dotenv()
# ─── Utility Function ─────────────────────────────────────────────────────────

def extract_video_id(url_or_id: str) -> str:
    """Extracts the 11-character YouTube video ID from a URL."""
    if len(url_or_id) == 11:
        return url_or_id
    
    pattern = r'(?:v=|\/v\/|youtu\.be\/|\/embed\/|\/shorts\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url_or_id)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract a valid YouTube video ID from: {url_or_id}")

# ─── Core Engine ──────────────────────────────────────────────────────────────

@dataclass
class YouTubeSession:
    """Stores the active state and vector index for a single thread session."""
    thread_id: str
    video_id: str
    vector_store: FAISS

class YouTubeContextEngine:
    """Manages transcript download, vector indexing, and raw context retrieval."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        # Pointing to your local Ollama instance for embeddings
        self._embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        self._sessions: Dict[str, YouTubeSession] = {}

    def _fetch_transcript(self, video_id: str) -> str:
        """Fetches raw transcript string using the required proxy configurations."""
        ytt_api = YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=os.getenv("YOUTUBE_PROXY_USERNAME"),
                proxy_password=os.getenv("YOUTUBE_PROXY_PASSWORD"),
            )
        )
        try:
            transcript_list = ytt_api.fetch(video_id, languages=["en"])
            return " ".join(chunk.text for chunk in transcript_list)
        except Exception as e:
            logger.error(f"Failed to fetch transcript for video {video_id}: {e}")
            raise e

    def _get_or_create_session(self, thread_id: str, video_id: str) -> YouTubeSession:
        """Returns cached vector store session or builds a new one if video or thread changes."""
        if thread_id in self._sessions and self._sessions[thread_id].video_id == video_id:
            logger.info(f"Reusing existing vector index for thread {thread_id} and video {video_id}")
            return self._sessions[thread_id]

        logger.info(f"Indexing new transcript for video {video_id} on thread {thread_id}")
        raw_text = self._fetch_transcript(video_id)
        
        metadata = {"source": f"YouTube Video ({video_id})", "video_id": video_id}
        documents = self._splitter.create_documents([raw_text], metadatas=[metadata])
        
        # Build the local FAISS database using Nomic embeddings
        vector_store = FAISS.from_documents(documents, self._embeddings)
        
        session = YouTubeSession(
            thread_id=thread_id,
            video_id=video_id,
            vector_store=vector_store
        )
        self._sessions[thread_id] = session
        return session

    def retrieve_context(self, video_url_or_id: str, query: str, thread_id: str, k: int = 4) -> str:
        """Performs semantic similarity matching and returns an unsummarized raw text context block."""
        video_id = extract_video_id(video_url_or_id)
        session = self._get_or_create_session(thread_id, video_id)
        
        retriever = session.vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": k}
        )
        retrieved_docs: List[Document] = retriever.invoke(query)
        
        if not retrieved_docs:
            return "No relevant context found in the provided YouTube transcript."

        formatted_sections = []
        for i, doc in enumerate(retrieved_docs, start=1):
            source_info = doc.metadata.get("source", f"YouTube Video ({video_id})")
            formatted_sections.append(
                f"--- Context {i} [{source_info}] ---\n{doc.page_content}"
            )
            
        return "\n\n".join(formatted_sections)