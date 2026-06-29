from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph_tools_used import tools

load_dotenv()

SYSTEM_PROMPT="""You are a helpful AI assistant"""
llm = ChatOllama(model="qwen3.5:4b")
llm_with_tools=llm.bind_tools(tools=tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm_with_tools.invoke(
        [
            SystemMessage(SystemMessage(content=SYSTEM_PROMPT)),
            *messages,
        ]
    )
    return {"messages": [response]}

tool_node=ToolNode(tools=tools)

#connection
conn=sqlite3.connect(database="chatbot.db",check_same_thread=False)

# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools",tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

# retrieving the thread_id's and titles and sending it in form of a dict to the frontend
def retrieve_all_threads():
    threads_with_titles = {}
    
    # Iterate through all saved checkpoints
    for checkpoint_tuple in checkpointer.list(None):
        thread_id = checkpoint_tuple.config["configurable"]["thread_id"]
        
        # Skip if we already processed this thread
        if thread_id in threads_with_titles:
            continue
            
        # Extract messages from the checkpoint
        messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])
        
        # Find the very first HumanMessage to use as the title
        title = "Untitled Chat"
        for msg in messages:
            if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'user'):
                content = msg.content
                title = content[:40] + "..." if len(content) > 40 else content
                break # Grab the first message and stop
                
        threads_with_titles[thread_id] = title
        
    return threads_with_titles
