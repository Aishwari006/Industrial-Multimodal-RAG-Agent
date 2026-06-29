import streamlit as st
from langgraph_database_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# **************************************** utility functions *************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():  # new chat
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []  # Clear old chat from UI
    # Clear the Streamlit cache so it forces a fresh read from the DB
    st.cache_data.clear()

# def add_thread(thread_id, title):
#     st.session_state["chat_threads"][thread_id] = title

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])


# **************************************** Session Setup ******************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

# We don't store chat_threads in session state anymore.
# They are always fetched from the database.
# if 'chat_threads' not in st.session_state:
#     st.session_state['chat_threads'] = retrieve_all_threads()


# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()
    st.rerun()

st.sidebar.header('My Conversations')

# Fetch the absolute latest threads from the database dynamically
# We use st.cache_data so it doesn't hurt performance on every keystroke
all_chat_threads = retrieve_all_threads()

# Render the sidebar using the live database threads
for thread_id, title in reversed(list(all_chat_threads.items())):
    if st.sidebar.button(title, key=str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        # converting to correct format since messages and the message format streamlit expects is different
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            elif isinstance(msg, AIMessage):
                role = 'assistant'
            else:
                continue

            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages
        st.rerun()


# **************************************** Main UI ************************************

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # # Create the conversation only when the first message is sent
    # if st.session_state['thread_id'] not in st.session_state['chat_threads']:
    #     title = user_input[:40]  # first 40 characters
    #     if len(user_input) > 40:
    #         title += "..."
    # # add_thread(st.session_state["thread_id"], title)

    # first add the message to message_history
    st.session_state['message_history'].append(
        {'role': 'user', 'content': user_input}
    )

    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # first add the message to message_history
    with st.chat_message("assistant"):

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage): #to handle tool message output format
                    # yield only assistant tokens
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state['message_history'].append(
        {'role': 'assistant', 'content': ai_message}
    )

    # Force a rerun so the sidebar instantly pulls the newly created thread from the DB!
    st.rerun()
    