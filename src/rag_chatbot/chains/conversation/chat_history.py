from langchain_core.chat_history import InMemoryChatMessageHistory
from langgraph.checkpoint.memory import InMemorySaver

# InMemorySaver stores conversation state using a session/
chat_memory = InMemorySaver()
chat_history = InMemoryChatMessageHistory()

def add_user_message(message: str):
    try:
        chat_history.add_user_message(message)
    except Exception as e:
        print(f"Error adding user message to history: {e}")


def add_ai_message(message: str):
    try:
        chat_history.add_ai_message(message)
    except Exception as e:
        print(f"Error adding AI message to history: {e}")


def get_chat_history():
    try:
        return chat_history.messages
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return []    



## FOR TESTINGS  
# if __name__ == "__main__":

#     add_user_message("What are employee responsibilities?")
#     add_ai_message("Employees must follow company policies and complete required training.")

#     add_user_message("What about managers?")
#     add_ai_message("Managers must explain expectations and monitor assigned controls.")

#     for message in get_chat_history():
#         print(f"{message.type}: {message.content}")
