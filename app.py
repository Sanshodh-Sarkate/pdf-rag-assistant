import streamlit as st
import uuid

from langchain_core.messages import HumanMessage

from rag_chatbot.chains.conversation.conversation_chain import (
    create_conversation_chain
)
from rag_chatbot.utils.storage import (
    load_conversations,
    save_conversations,
    sync_langgraph_state
)


# Configure the application
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)


# Initialize the conversation chain
@st.cache_resource 
def get_conversation_chain():
    try:
        return create_conversation_chain()
    except Exception as e:
        st.error(f"Failed to initialize conversation chain: {e}")
        return None


conversation_chain = get_conversation_chain()

if conversation_chain is None:
    st.error("The chatbot backend could not be initialized. Please check your credentials and configuration.")
    st.stop()


# Application title
st.title("📚 PDF RAG Chatbot")

# Application description
st.write("Ask questions about your PDF documents.")
  

# 1. Initialize conversations from persistent disk storage
if "conversations" not in st.session_state:
    st.session_state.conversations = load_conversations()

# 2. Initialize current active thread_id
if "thread_id" not in st.session_state:
    query_thread_id = st.query_params.get("thread_id")
    
    if query_thread_id and query_thread_id in st.session_state.conversations:
        st.session_state.thread_id  =  query_thread_id
    elif st.session_state.conversations:
        # Restore the latest conversation
        st.session_state.thread_id = list(st.session_state.conversations.keys())[-1]
        st.query_params["thread_id"] = st.session_state.thread_id    
    else:
        new_thread_id = str(uuid.uuid4())
        st.session_state.thread_id = new_thread_id
        st.query_params["thread_id"] = new_thread_id
        st.session_state.conversations[new_thread_id] = {
            "title": "New Chat",
            "messages": []
        }
        save_conversations(st.session_state.conversations)

# 3. Initialize chat messages for active conversation
if "messages" not in st.session_state:
    st.session_state.messages = st.session_state.conversations.get(
        st.session_state.thread_id, {}
    ).get("messages", [])
    # Synchronize LangGraph checkpointer memory for the active thread
    sync_langgraph_state(
        conversation_chain,
        st.session_state.thread_id,
        st.session_state.messages
    )


# Sidebar
st.sidebar.title("📚 PDF RAG Chatbot")

# New Chat button
if st.sidebar.button("➕ New Chat", use_container_width=True):
    new_thread_id = str(uuid.uuid4())
    st.session_state.thread_id = new_thread_id
    st.query_params["thread_id"] = new_thread_id
    st.session_state.messages = []
    st.session_state.conversations[new_thread_id] = {
        "title": "New Chat",
        "messages": []
    }
    save_conversations(st.session_state.conversations)
    st.rerun()

# Display conversation history
st.sidebar.subheader("Chat History")

# this for display the chat history and when click on that then open the conversation 
for thread_id, conv_data in list(st.session_state.conversations.items()):
    title = conv_data.get("title", "New Chat")
    is_active = (thread_id == st.session_state.thread_id)
    btn_label = f"👉 {title}" if is_active else f"💬 {title}"

    if st.sidebar.button(
        btn_label,
        key=thread_id,
        use_container_width=True
    ):
        st.session_state.thread_id = thread_id
        st.query_params["thread_id"] =  thread_id
        st.session_state.messages = conv_data.get("messages", [])
        # Synchronize LangGraph checkpointer for the chosen thread
        sync_langgraph_state(
            conversation_chain,
            thread_id,
            st.session_state.messages
        )
        st.rerun()

# Clear all chats button
if st.session_state.conversations:
    st.sidebar.write("")
    if st.sidebar.button("🗑️ Clear All Chats", use_container_width=True):
        st.session_state.conversations = {}
        st.session_state.messages = []
        new_thread_id = str(uuid.uuid4())
        st.session_state.thread_id = new_thread_id
        st.query_params["thread_id"] = new_thread_id
        st.session_state.conversations[new_thread_id] = {
            "title": "New Chat",
            "messages": []
        }
        save_conversations(st.session_state.conversations)
        st.rerun()


# Display previous messages of current conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Get user input
user_input = st.chat_input("Ask something about your PDFs...")


if user_input:
    current_conv = st.session_state.conversations.setdefault(
        st.session_state.thread_id,
        {"title": "New Chat", "messages": []}
    )

    # Generate title from the first question
    if current_conv.get("title") == "New Chat":
        title = user_input.strip()
        if len(title) > 35:
            title = title[:35] + "..."
        current_conv["title"] = title

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = conversation_chain.invoke(
                    {
                        "messages": [
                            HumanMessage(content=user_input)
                        ]
                    },
                    config={
                        "configurable": {
                            "thread_id": st.session_state.thread_id
                        }
                    }
                )

                answer = response["messages"][-1].content
            except Exception as e:
                answer = f"Sorry, an error occurred while processing your request: {e}"
                st.error(answer)

        # Display AI response
        st.markdown(answer)

    # Store AI response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # Persist updated messages to disk storage
    current_conv["messages"] = list(st.session_state.messages)
    save_conversations(st.session_state.conversations)
