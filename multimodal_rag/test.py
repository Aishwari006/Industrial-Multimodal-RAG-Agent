from multimodal_rag.main import run_query

answer = run_query(
    pdf_path="multimodal_rag/multimodal_sample.pdf",
    query="Summarize this document.",
    thread_id="thread-1",
)

print("\nANSWER:\n")
print(answer)