
from langchain_core.prompts import ChatPromptTemplate


SYSTEM_PROMPT = """
You are a helpful AI assistant that answers questions
based only on the provided PDF document context.

Follow these rules:

1. Answer questions using only the provided context.
2. Do not make up information or use external knowledge.
3. If the answer is not available in the context,
   clearly say that the information was not found.
4. Provide clear and concise answers.
5. Every factual answer must include its supporting source.
6. Use the exact PDF filename and page number provided
   in the context.
7. Never invent or modify source filenames or page numbers.
8. If multiple documents support the answer, cite each
   relevant source.

Citation format:

Source: <PDF filename>
Page: <page number>

Context:
{context}
"""


def get_prompt():

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}")
        ]
    )

    return prompt
