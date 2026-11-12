#It helps us create a reusable prompt structure instead of manually writing the entire prompt every time.
from langchain_core.prompts import ChatPromptTemplate


## system prompts
SYSTEM_PROMPT = """
You are a helpful AI assistant that answers questions
based on the provided PDF document context.

Follow these rules:

1. Answer questions using only the provided context.
2. Do not make up information or use external knowledge.
3. If the answer is not available in the context,
   clearly say that the information was not found.
4. Provide clear and concise answers.
5. Include the source PDF filename and page number
   for the information used in your answer.

Context:
{context}  ## The relevant chunks retrieved from Astra DB.
"""

## prompts  
def get_prompt():
   prompt=ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}")
        ]
    )
   
   return prompt




# if __name__ == "__main__":

#     prompt = get_prompt()

#     formatted_prompt = prompt.invoke({
#         "context": """
#         Employees must read applicable policies,
#         complete required training, and protect
#         company information.

#         Source: Company_Policies_Sample_50_Pages.pdf
#         Page: 4
#         """,
#         "question": "What are the responsibilities of employees?"
#     })

#     print(formatted_prompt)
