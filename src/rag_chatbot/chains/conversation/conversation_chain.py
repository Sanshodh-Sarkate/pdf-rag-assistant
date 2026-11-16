from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AIMessage, HumanMessage
# from rag_chatbot.chains import rag_chain
from rag_chatbot.chains.rag_chain import create_rag_chain
from rag_chatbot.chains.conversation.chat_history import chat_memory

def create_conversation_chain():
    rag_chain = create_rag_chain()

   # process on question and genarate anser
    def generate_answer(state: MessagesState):
        #get latest msg or question
        question = state["messages"][-1].content
    
        ##previous conversation msg 
        chat_history = state["messages"][:-1]
    
        # Execute the existing RAG chain
        response = rag_chain.invoke({
           "chat_history": chat_history,
           "question": question
        })
    
         # Return the AI response
        return {
           "messages": [AIMessage(content=response)]
        }
    
    # create the  graphState
    graph = StateGraph(MessagesState) ## contain the  humman and Ai msg 
    
    graph.add_node("generate_answer", generate_answer)
    
    
    # Define the workflow
    graph.add_edge(START, "generate_answer")
    graph.add_edge("generate_answer", END)

    # Compile graph with memory checkpointer
    return graph.compile(checkpointer=chat_memory)
    



if __name__ == "__main__":

    conversation_chain = create_conversation_chain()

    config = {
        "configurable": {
            "thread_id": "session_001"
        }
    }

    # First question
    response1 = conversation_chain.invoke(
        {
            "messages": [
                HumanMessage(content="What are employee responsibilities?")
            ]
        },
        config=config
    )

    print("\nFirst Answer:")
    print(response1["messages"][-1].content)

    # Second question
    response2 = conversation_chain.invoke(
        {
            "messages": [
                HumanMessage(content="What about managers?")
            ]
        },
        config=config
    )

    print("\nSecond Answer:")
    print(response2["messages"][-1].content)