from langchain_core.tools import tool
from dotenv import load_dotenv
from exa_py import Exa
import requests
import os
from multimodal_rag.main import retrieve_context
from langchain_core.runnables import RunnableConfig
from pathlib import Path
from pydantic import BaseModel, Field
from youtube_engine.youtube_context_engine import YouTubeContextEngine
load_dotenv()

# ****************************************
# Exa Client
# ****************************************

exa = Exa(api_key=os.getenv("EXA_API_KEY"))

# Force the tools file to resolve paths absolutely from the exact execution root folder
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploaded_pdfs"

def find_existing_pdf(thread_id: str) -> str | None:
    """Helper to look for an existing file associated with this thread ID."""
    for file in UPLOAD_DIR.glob(f"{thread_id}_*.pdf"):
        return str(file.resolve())
    return None
# ****************************************
# RAG Tool
# ****************************************
@tool
def pdf_qa_tool(query: str, config: RunnableConfig) -> str:
    """
    Answer questions about an uploaded PDF.

    Use this tool whenever the user's question refers to the uploaded PDF,
    its text, tables, figures, charts, or images.

    Args:
        query: User's question about the PDF content.
    """
    # 1. Extract the thread_id securely from the backend context
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return "System Error: Could not determine the current conversation thread."
    
    # 2. Derive the pdf_path using the thread_id
    pdf_path = find_existing_pdf(thread_id)
    if not pdf_path:
        return "Error: No PDF document has been uploaded for this conversation yet. Please ask the user to upload one."

    # 3. Execute your core logic
    return retrieve_context(
        pdf_path=pdf_path,
        query=query,
        thread_id=thread_id,
    )

# ****************************************
# Youtube engine
# ****************************************


# Instantiate the engine globally or inside your routing context
youtube_engine = YouTubeContextEngine()

class YouTubeContextInput(BaseModel):
    url: str = Field(description="The complete YouTube URL or video ID to fetch context from.")
    query: str = Field(description="The specific question or semantic query regarding the video content.")
    thread_id: str = Field(description="The persistent conversation session or thread ID.")

@tool("youtube_transcript_context_retriever", args_schema=YouTubeContextInput)
def youtube_transcript_context_retriever(url: str, query: str, thread_id: str) -> str:
    """
    Extracts relevant context chunks from a YouTube video transcript via semantic similarity search. 
    Use this tool whenever a user provides a YouTube video link and asks questions regarding its text content.
    """
    try:
        context_block = youtube_engine.retrieve_context(
            video_url_or_id=url,
            query=query,
            thread_id=thread_id
        )
        return context_block
    except Exception as e:
        return f"Error retrieving context from YouTube transcript: {str(e)}"

# ****************************************
# Stock Price
# ****************************************

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price.

    Use this tool whenever the user asks for
    the latest price of a publicly traded company.
    """

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey={os.getenv('STOCK_API_KEY')}"
    )

    try:
        response = requests.get(url, timeout=15)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# ****************************************
# Web Search
# ****************************************

@tool
def web_search(query: str) -> dict:
    """
    Executes a live search query to discover current, external, or highly specific web information.

    CRITICAL USAGE GUIDELINES:
    1. Use when the user asks about real-world facts, current events, recent updates, 
       software documentation, APIs, GitHub repositories, company data, or research papers.
    2. Mandatory if your internal knowledge base is insufficient, outdated, or lacks specific evidence.
    3. Do NOT use this tool if the user provided a direct URL to analyze (use 'crawl_webpage' instead).
    4. Do NOT use for basic logic, mathematical computations, or general conversation.

    Args:
        query (str): A precise, keyword-optimized search query. Avoid conversational phrases.

    Returns:
        dict: A dictionary containing the original query alongside a list of the top 5 
              most relevant web results, URLs, and contextual text highlights.
    """

    try:

        response = exa.search(
            query=query,
            type="auto",
            num_results=2,
            contents={
                "highlights": True
            }
        )

        output = []
        combined_context = ""

        for result in response.results:

            title = getattr(result, "title", "")
            url = getattr(result, "url", "")
            highlights = getattr(result, "highlights", [])

            output.append(
                {
                    "title": title,
                    "url": url,
                    "highlights": highlights
                }
            )

            combined_context += (
                f"\nTitle: {title}\n"
                f"URL: {url}\n"
                f"{' '.join(highlights)}\n"
            )

        return {
            "query": query,
            "results": output,
            "context": combined_context.strip()
        }

    except Exception as e:
        return {
            "error": str(e)
        }

# ****************************************
# Crawl Webpage
# ****************************************

@tool
def crawl_webpage(url: str) -> dict:
    """
    Directly extracts and summarizes content from a specific, user-provided URL.

    CRITICAL USAGE GUIDELINES:
    1. Use ONLY when the user explicitly provides a fully qualified URL (e.g., 'https://...') 
       and requests analysis, summaries, or answers based strictly on that page's content.
    2. Do NOT use this tool if you need to discover information or find links; use 'web_search' instead.
    3. Essential for deep-diving into specific documentation pages, articles, or reference links provided in the prompt.

    Args:
        url (str): The exact, absolute URL of the webpage to crawl.

    Returns:
        dict: A dictionary containing the target URL, page titles, structured summaries, 
              and dense text chunks required to accurately synthesize an answer.
    """

    try:

        response = exa.get_contents(
            [url],
            summary=True,
            subpages=1,
            text={
                "max_characters": 1000
            }
        )

        if not response.results:
            return {
                "error": "No content could be extracted."
            }

        summaries = []
        combined_context = ""

        for page in response.results:

            title = getattr(page, "title", "")
            page_url = getattr(page, "url", "")
            summary = getattr(page, "summary", "")
            text = getattr(page, "text", "")

            summaries.append(
                {
                    "title": title,
                    "url": page_url,
                    "summary": summary
                }
            )

            combined_context += (
                f"\n========== PAGE ==========\n"
                f"Title: {title}\n"
                f"URL: {page_url}\n"
                f"Summary: {summary}\n\n"
                f"{text}\n"
            )

        return {
            "url": url,
            "summaries": summaries,
            "context": combined_context.strip()
        }

    except Exception as e:
        return {
            "error": str(e)
        }

# ****************************************
# Tool List
# ****************************************

tools = [
    youtube_transcript_context_retriever,
    get_stock_price,
    web_search,
    crawl_webpage,
    pdf_qa_tool,
]