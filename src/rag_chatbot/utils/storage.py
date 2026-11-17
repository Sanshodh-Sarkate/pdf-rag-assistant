import json
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage

# Path to the persistent conversations file
BASE_DIR = Path(__file__).resolve().parents[3]
STORAGE_FILE = BASE_DIR / "data" / "chat_history.json"

# CHECK FILE CHAT HISTORY EXIST OR NOT
def load_conversations() -> dict:
    """
    Load saved conversations from local disk.

    Returns:
        dict: Mapping of thread_id -> {"title": str, "messages": list}
    """
    if STORAGE_FILE.exists():
        try:
            with open(STORAGE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as e:
            print(f"Error loading chat history from {STORAGE_FILE}: {e}")
            return {}
    return {}

## SVAE THE  HISTORY IN THE FILE AND IF  NOT EXIST THEN CREATE THIS 
def save_conversations(conversations: dict) -> None:
    """
    Save conversations to local disk.
    """
    try:
        STORAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STORAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving chat history to {STORAGE_FILE}: {e}")



def sync_langgraph_state(conversation_chain, thread_id: str, messages: list) -> None:
    """
    Ensure the LangGraph conversation_chain checkpointer has the messages
    for thread_id so multi-turn memory works after page refreshes or restarts.
    """
    if not conversation_chain or not thread_id or not messages:
        return

    config = {"configurable": {"thread_id": thread_id}}
    try:
        ## get the  conversation from the langGraph  for this given threadId 
        current_state = conversation_chain.get_state(config)
        existing_msgs = current_state.values.get("messages", [])
        if not existing_msgs:
            langchain_messages = []
            for msg in messages:
                role = msg.get("role") ## user
                content = msg.get("content", "")  ## rag
                if role == "user":
                    langchain_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=content))
            ## update the langgraph state 
            if langchain_messages:
                conversation_chain.update_state(config, {"messages": langchain_messages})
    except Exception as e:
        print(f"Warning: could not sync LangGraph state for {thread_id}: {e}")
