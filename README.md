# Industrial-Multimodal-RAG-Agent

AI-powered Industrial Knowledge Intelligence Platform built with LangGraph, LangChain, Multimodal RAG, and Ollama. Supports PDF ingestion, image understanding, semantic retrieval, conversational memory, live web search, automated web crawling, and tool-calling to answer engineering and industrial knowledge queries using local LLMs.

## Features

* Multimodal PDF understanding
* Image-aware retrieval using a Vision Language Model
* Hybrid semantic search
* LangGraph agent with tool calling
* Live web search and automated web crawling
* Calculator and stock tools
* Persistent conversation memory
* Multi-thread conversations
* Fully local inference using Ollama

## Architecture Flow

```text
User
        │
        ▼
Streamlit Frontend
        │
        ▼
LangGraph Agent
        │
 ┌──────┼──────────────┬──────────────┐
 │      │              │              │
 ▼      ▼              ▼              ▼
Calc  Web Search  Web Crawler  Multimodal RAG
                                      │
                                      ▼
                               PDF Ingestion
                                      │
                               Text + Images
                                      │
                             Vision Summarization
                                      │
                                 FAISS Index
                                      │
                               Retrieved Context
                                      │
                                      ▼
                                Final Response
```
## Tech Stack

**AI & Orchestration**
* LangGraph
* LangChain
* Ollama (Qwen 3.5, Nomic Embed Text)

**Retrieval & Storage**
* FAISS
* RecursiveCharacterTextSplitter
* SQLite (Conversational Memory)

**External Data Access**
* Web Search API
* Web Crawling Engine (BeautifulSoup / Playwright / Custom Scraper)

**Vision & Document Processing**
* PyMuPDF
* Pillow

**Frontend & Backend Core**
* Python
* Streamlit

---

## Project Structure

```text
Industrial-Knowledge-Intelligence-Platform/
│
├── streamlit_frontend.py
├── langgraph_database_backend.py
├── langgraph_tools_used.py
├── chatbot.db
│
├── multimodal_rag/
│   ├── config.py
│   ├── ingestion.py
│   ├── image_summarizer.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── session_manager.py
│   ├── pipeline.py
│   └── main.py
│
├── uploaded_pdfs/
│
├── speech_client.py
├── whisper_server.py
│
├── requirements.txt
└── README.md
```
## Pipeline Execution

### 1. PDF Processing Phase
1. Upload PDF document
2. Extract raw text
3. Extract embedded images
4. Generate image summaries using Vision Models
5. Create vector embeddings for text and summaries
6. Store embeddings in a local FAISS index

### 2. Query Processing Phase
1. User submits a question
2. LangGraph evaluates intent and dynamically routes to the appropriate tool (PDF Retriever, Web Search, Web Crawler, or Calculator)
3. If crawling is required, the agent extracts target URLs, scrapes page content, and filters for relevant data
4. Retrieved context from all sources is returned to the agent
5. The agent synthesizes the final response using the unified context

---

## Installation & Setup

```bash
# Clone the repository
git clone [https://github.com/yourusername/Industrial-Knowledge-Intelligence-Platform.git](https://github.com/yourusername/Industrial-Knowledge-Intelligence-Platform.git)
cd Industrial-Knowledge-Intelligence-Platform

# Install dependencies
pip install -r requirements.txt

# Pull required local models via Ollama
ollama pull qwen3.5:4b
ollama pull nomic-embed-text

# Run the application
streamlit run streamlit_frontend.py
```
## Future Improvements

* Optical Character Recognition (OCR) support for scanned documents
* Advanced table extraction and parsing
* Knowledge Graph integration for complex entity relationships
* Multi-document retrieval capabilities
* Inline citation support for generated answers
* Cross-document reasoning
