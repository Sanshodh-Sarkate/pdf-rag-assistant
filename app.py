
import streamlit as st
import uuid

from langchain_core.messages import HumanMessage

from rag_chatbot.chains.conversation.conversation_chain import (
    create_conversation_chain
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
    return create_conversation_chain()


conversation_chain = get_conversation_chain()


# Application title
st.title("📚 PDF RAG Chatbot")

# Application description
st.write("Ask questions about your PDF documents.")


# Initialize chat history
# Initialize conversations
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

# Sidebar
st.sidebar.title("📚 PDF RAG Chatbot")

# New Chat button
# if st.sidebar.button("➕ New Chat", use_container_width=True):

#     thread_id = str(uuid.uuid4())

#     st.session_state.thread_id = thread_id
#     st.session_state.messages = []

#     st.session_state.conversations[thread_id] = {
#         "title": "New Chat"
#     }

#     st.rerun()


# Display conversation history
st.sidebar.subheader("Chat History")

# This for creating new conversations and reopening previous conversations from the Streamlit sidebar,
for thread_id, conversation in st.session_state.conversations.items():
  ## create the  btn for the  every chat and assign id  
    if st.sidebar.button(
        conversation["title"],
        key=thread_id,
        use_container_width=True
    ):
       ## when click  on this  btn then  
        st.session_state.thread_id = thread_id

        # Load selected conversation messages
        state = conversation_chain.get_state(
            {
                "configurable": {
                    "thread_id": thread_id
                }
            }
        )
        
        ## first make it empty 
        st.session_state.messages = []
        # here we  update teh  role and content
        for message in state.values.get("messages", []):
            ## I'll do this  beacuse the langgraph and  streamlit  type defination is diffrent
            if message.type == "human":
                role = "user"
            elif message.type == "ai":
                role = "assistant"
            else:
                continue

            st.session_state.messages.append({
                "role": role,
                "content": message.content
            })

        st.rerun()    

# Initialize current conversation
if "thread_id" not in st.session_state:
    thread_id = str(uuid.uuid4())

    ## store thread_Id in session
    st.session_state.thread_id = thread_id

    ##
    st.session_state.conversations[thread_id] = {
        "title": "New Chat"
    }

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# New Chat button
if st.sidebar.button("➕ New Chat"):

    st.session_state.messages = []

    st.session_state.thread_id = str(uuid.uuid4())

    st.rerun()
    
    
# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Get user input
user_input = st.chat_input("Ask something about your PDFs...")


if user_input:
        # Generate title from the first question
    if st.session_state.conversations[
        st.session_state.thread_id
    ]["title"] == "New Chat":

        title = user_input.strip()

        if len(title) > 35:
            title = title[:35] + "..."

        st.session_state.conversations[
            st.session_state.thread_id
        ]["title"] = title

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

        # Display AI response
        st.markdown(answer)

    # Store AI response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    }) 