import pytest
import uuid
from langchain_core.messages import HumanMessage
from rag_chatbot.chains.conversation.conversation_chain import create_conversation_chain


# --- 1. Fixture: Initialize the chatbot chain once ---
@pytest.fixture(scope="module")
def bot():
    """Initializes the conversation chain once for all tests."""
    chain = create_conversation_chain()
    return chain


# --- 2. Helper function to invoke the chatbot ---
def ask_bot(bot, question: str, thread_id: str = None) -> str:
    """Helper to query the bot and get the assistant answer string."""
    if not thread_id:
        thread_id = str(uuid.uuid4())

    response = bot.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}}
    )
    answer = response["messages"][-1].content

    # --- Print to see the output in the terminal ---
    print(f"\n\n❓ QUESTION: {question}")
    print(f"ANSWER:\n{answer}")
    print("-" * 60)

    return answer

# --- 3. Test Direct / In-Scope Questions ---
@pytest.mark.parametrize("question, expected_keywords", [
    ("What are employee responsibilities?", ["responsibilities", "employee", "duty"]),
    ("What is the policy review procedure?", ["policy", "review"]),
])
def test_in_scope_questions(bot, question, expected_keywords):
    answer = ask_bot(bot, question)
    
    # 1. Answer should not be empty
    assert answer is not None
    assert len(answer.strip()) > 20
    
    # 2. Answer should not contain backend crash messages
    assert "An error occurred" not in answer
    
    # 3. Check if at least one expected keyword is discussed
    found_keyword = any(k.lower() in answer.lower() for k in expected_keywords)
    assert found_keyword, f"Expected keywords {expected_keywords} not found in: {answer}"


# --- 4. Test Out-of-Scope Questions (Anti-Hallucination) ---
@pytest.mark.parametrize("question", [
    "How to bake a chocolate cake?",
    "Who won the FIFA World Cup in 2022?",
    "Write a Python script for binary search."
])
def test_out_of_scope_questions(bot, question):
    answer = ask_bot(bot, question)
    
    # Answer should indicate the information is not in the documents
    # (adjust based on your system prompt instructions)
    assert answer is not None
    assert "An error occurred" not in answer


# --- 5. Test Multi-Turn / Conversation Memory ---
def test_conversation_followup(bot):
    session_id = f"test_thread_{uuid.uuid4()}"

    # First Turn
    ans1 = ask_bot(bot, "What are the rules for annual leave?", thread_id=session_id)
    assert len(ans1) > 0
    assert "An error occurred" not in ans1

    # Follow-up Turn (Relies on context of the first question)
    ans2 = ask_bot(bot, "How many days are allowed?", thread_id=session_id)
    assert len(ans2) > 0
    assert "An error occurred" not in ans2


# --- 6. Test Edge Cases & Robustness ---
@pytest.mark.parametrize("edge_input", [
    "???",
    "hi",
    "a" * 300,  # very long query
])
def test_edge_cases(bot, edge_input):
    answer = ask_bot(bot, edge_input)
    assert answer is not None
    assert "An error occurred" not in answer