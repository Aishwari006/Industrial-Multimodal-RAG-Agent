import streamlit as st
from langgraph_database_backend import chatbot,checkpointer, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
from streamlit_mic_recorder import mic_recorder
from speech_client import transcribe_audio
import uuid
import tempfile
import os
import asyncio

# **************************************** utility functions *************************

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat(): # new chat
    st.session_state['thread_id'] = generate_thread_id()
    st.session_state['message_history'] = []
    st.session_state['text_input'] = ""  # Reset input field content on new chat
    if hasattr(st, "cache_data"):
        st.cache_data.clear()
    
async def load_conversation(thread_id):
    state = await chatbot.aget_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])


# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""


# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()
    st.rerun()

st.sidebar.header('My Conversations')

# Fetch the absolute latest threads from the database dynamically
all_chat_threads = asyncio.run(retrieve_all_threads(checkpointer=checkpointer))

# Render the sidebar using the live database threads
for thread_id, title in reversed(list(all_chat_threads.items())):
    button_label = f"💬 {title}" if thread_id == st.session_state['thread_id'] else title
    
    if st.sidebar.button(button_label, key=f"btn_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        st.session_state['text_input'] = "" # Clear text area on switching threads
        messages = asyncio.run(load_conversation(thread_id))

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'user'):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages
        st.rerun()


# **************************************** Main UI ************************************

# Loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


# **************************************** Voice + Text Input ****************************

st.markdown("---")

audio = mic_recorder(
    start_prompt="🎤 Record",
    stop_prompt="⏹ Stop",
    just_once=True,
    use_container_width=True,
    key="voice_recorder",
)

# If audio is recorded
if audio:
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp_file:
        tmp_file.write(audio["bytes"])
        temp_audio_path = tmp_file.name

    with st.spinner("Transcribing..."):
        try:
            transcript = transcribe_audio(temp_audio_path)
            st.session_state["text_input"] = transcript
        except Exception as e:
            st.error(str(e))
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    
    # Rerun so the text area picks up the newly updated st.session_state["text_input"]
    st.rerun()

# Editable textbox
user_input = st.text_area(
    "Message",
    value=st.session_state["text_input"],
    height=120,
)

if st.button("Send"):
    if user_input.strip() == "":
        st.stop()

    st.session_state["text_input"] = ""

    # First add the message to message_history
    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }

    # Stream the assistant's reply
    with st.chat_message("assistant"):
        
        async def ai_only_stream():
            async for message_chunk, metadata in chatbot.astream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, AIMessage): #to handle tool message output format
                    yield message_chunk.content
        #Since st.write_stream() expects a synchronous generator
        def sync_stream():
            async def collect():
                chunks = []

                async for chunk in ai_only_stream():
                    chunks.append(chunk)

                return chunks

            for chunk in asyncio.run(collect()):
                yield chunk

        ai_message = st.write_stream(sync_stream())

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message,
        }
    )

    st.rerun()