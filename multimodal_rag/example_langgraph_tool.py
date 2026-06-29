"""
example_langgraph_tool.py
-------------------------
Shows how a LangGraph chatbot would import and invoke the Multimodal RAG
pipeline as a tool.  The RAG pipeline has zero knowledge of LangGraph.

This file is NOT part of the multimodal_rag package — it lives in the
LangGraph chatbot project.
"""

from langchain_core.tools import tool
from multimodal_rag import run_query   # ← the only import needed


# ── 1. Wrap run_query as a LangGraph-compatible tool ────────────────────────

@tool
def pdf_qa_tool(pdf_path: str, query: str, thread_id: str) -> str:
    """
    Answer a question about the contents of a PDF file.

    Use this tool whenever the user asks a question that can be answered
    from a PDF document.

    Args:
        pdf_path:  Absolute path to the PDF file.
        query:     The user's question.
        thread_id: Unique conversation ID (use the LangGraph thread id).
    """
    return run_query(pdf_path=pdf_path, query=query, thread_id=thread_id)


# ── 2. Minimal LangGraph agent skeleton (illustrative) ──────────────────────

def build_agent():
    """
    Skeleton showing how to wire pdf_qa_tool into a LangGraph ReAct agent.
    Requires: langgraph, langchain-ollama
    """
    from langgraph.prebuilt import create_react_agent
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model="qwen2.5:3b", temperature=0)
    agent = create_react_agent(llm, tools=[pdf_qa_tool])
    return agent


# ── 3. Demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = build_agent()
    config = {"configurable": {"thread_id": "demo-session-001"}}

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Using the file /tmp/report.pdf, "
                        "what are the top clients and their purchases?"
                    ),
                }
            ]
        },
        config=config,
    )
    print(result["messages"][-1].content)
