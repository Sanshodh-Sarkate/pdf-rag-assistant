from langchain_core.output_parsers import StrOutputParser

from rag_chatbot.chains.llm import get_llm
from rag_chatbot.chains.conversation.contextualize_prompt import (get_contextualize_prompt)

def create_contextualize_chain():
    try:
        prompt = get_contextualize_prompt()
        llm = get_llm()

        contextualize_chain = (
            prompt 
            | llm
            | StrOutputParser()  
        )

        return contextualize_chain
    except Exception as e:
        raise RuntimeError(f"Failed to create contextualize chain: {e}") from e




# ## Testing  
# if __name__ == "__main__":

#     contextualize_chain = create_contextualize_chain()

#     response = contextualize_chain.invoke(
#         {
#             "chat_history": [
#                 ("human", "What are employee responsibilities?"),
#                 ("ai", "Employees must follow company policies and complete required training.")
#             ],
#             "question": "What about managers?"
#         }
#     )

#     print("\nRewritten Question:")
#     print(response)


# uv run python -m rag_chatbot.chains.conversation.contextualize_chain  