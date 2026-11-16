from langchain_core.prompts import  MessagesPlaceholder , ChatPromptTemplate

## This rewrite the  our question based on the  privious  input question
def get_contextualize_prompt():
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Given the conversation history and the latest user question,
                rewrite the latest question as a standalone question.

                Do not answer the question.
                If the question is already standalone, return it unchanged.
                """
            ),
            
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ]
    )
    
    return prompt