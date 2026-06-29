import streamlit as st
from langgraph_database_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
from streamlit_mic_recorder import mic_recorder
from speech_client import transcribe_audio
import uuid
import tempfile
import os
from pathlib import Path

# **************************************** Constants & Config *************************
MAX_FILE_SIZE_MB = 10  # Enforce small file size limit

# FORCE AN ABSOLUTE PATH based on the root of this project file to eliminate relative path bugs
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploaded_pdfs"
UPLOAD_DIR.mkdir(exist_ok=True)

# **************************************** utility functions *************************

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat(): # new chat
    st.session_state['thread_id'] = generate_thread_id()
    st.session_state['message_history'] = []
    st.session_state['text_input'] = ""  # Reset input field content on new chat
    if "pdf_path" in st.session_state:
        del st.session_state["pdf_path"]
    if hasattr(st, "cache_data"):
        st.cache_data.clear()
    
def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

# Helper to look for an existing file associated with this thread ID using absolute matching
def find_existing_pdf(thread_id: str) -> str | None:
    for file in UPLOAD_DIR.glob(f"{thread_id}_*.pdf"):
        return str(file.resolve())
    return None

# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""

# Keep track of active PDF path across reruns for this thread
if "pdf_path" not in st.session_state:
    st.session_state["pdf_path"] = find_existing_pdf(st.session_state['thread_id'])


# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()
    st.rerun()

st.sidebar.header('My Conversations')

# Fetch the absolute latest threads from the database dynamically
all_chat_threads = retrieve_all_threads()

# Render the sidebar using the live database threads
for thread_id, title in reversed(list(all_chat_threads.items())):
    button_label = f"💬 {title}" if thread_id == st.session_state['thread_id'] else title
    
    if st.sidebar.button(button_label, key=f"btn_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        st.session_state['text_input'] = "" # Clear text area on switching threads
        st.session_state["pdf_path"] = find_existing_pdf(thread_id) # Check for thread-linked PDF
        messages = load_conversation(thread_id)

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
        st.markdown(message['content'])


# **************************************** PDF Upload Implementation ******************
st.markdown("---")
st.subheader("Document Context")

uploaded_file = st.file_uploader(
    "Upload a PDF for this conversation", 
    type=["pdf"], 
    key=f"pdf_uploader_{st.session_state['thread_id']}" # Tied to thread id to reset on new chat
)

if uploaded_file is not None:
    # 1. Compulsorily check size limit
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large! Max allowed size is {MAX_FILE_SIZE_MB}MB. (Your file: {file_size_mb:.2f}MB)")
    else:
        # 2. Sanitize and save cleanly using absolute path pattern
        safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in "._-").strip()
        save_path = UPLOAD_DIR / f"{st.session_state['thread_id']}_{safe_filename}"
        
        # Avoid rewriting the file if it already exists
        if not save_path.exists():
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Successfully processed: {uploaded_file.name}")
        
        # Cache absolute path in state
        st.session_state["pdf_path"] = str(save_path.resolve())

# Information readout for the user
if st.session_state["pdf_path"]:
    filename_clean = Path(st.session_state["pdf_path"]).name.split("_", 1)[-1]
    st.info(f"📁 **Active PDF:** {filename_clean}")


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

    # First add the message to message_history as typed by the user
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
        def ai_only_stream():
            # Construct the payload message text
            final_content = user_input
            
            # CRITICAL FIX: If a PDF is active, seamlessly append a system injection instruction 
            # to let the LLM know the document is active and it is authorized to call its RAG tool.
            if st.session_state["pdf_path"]:
                filename_clean = Path(st.session_state["pdf_path"]).name.split("_", 1)[-1]
                final_content += f"\n\n[System Context: An active PDF document named '{filename_clean}' is uploaded for this conversation thread. If the user's message requires data or context from the file, use your pdf retrieval tool to query it.]"
            
            user_payload = {"messages": [HumanMessage(content=final_content)]}
            
            for message_chunk, metadata in chatbot.stream(
                user_payload,
                config=CONFIG,  # CONFIG already holds the thread_id
                stream_mode="messages",
            ):
                if isinstance(message_chunk, AIMessage): 
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message,
        }
    )

    st.rerun()