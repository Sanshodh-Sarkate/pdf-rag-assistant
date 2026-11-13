## it convert the llm output to the plain python string 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda


from rag_chatbot.chains.llm import get_llm
from  rag_chatbot.chains.prompt import get_prompt
from rag_chatbot.retrieval.retriever import retrieve_documents

## This  function are helep to proper formated the the retrieve documents

def format_documents(documents):
    formatted_documents = []

    for doc in documents:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")

        content = f"""
Source: {source}
Page: {int(page) if isinstance(page, (int, float)) else page}

Content:
{doc.page_content}
"""
        formatted_documents.append(content)

    return "\n\n".join(formatted_documents)


def create_rag_chain():

    prompt = get_prompt()
    llm = get_llm()

    # TODO:  NEED TO SEE 
    #  Retrieve documents only once
    retrieval_chain = RunnableLambda(
        lambda inputs: {
            "question": inputs["question"],
            "documents": retrieve_documents(inputs["question"])
        }
    )

    # Step 2: Generate answer using retrieved context
    generation_chain = (
        {
            "context": lambda inputs: format_documents(
                inputs["documents"]
            ),
            "question": lambda inputs: inputs["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Step3: Check whether relevant documents exist
    rag_chain = retrieval_chain | RunnableBranch(
        (
            lambda inputs: len(inputs["documents"]) == 0,
            RunnableLambda(
                lambda inputs:
                "I could not find relevant information in the provided PDF documents."
            )
        ),
        generation_chain
    )

    return rag_chain


if __name__ == "__main__":

    rag_chain = create_rag_chain()

    # question = "What is the LLM?"
    question = "What are the responsibilities of employees and managers?"

    response = rag_chain.invoke({
        "question": question
    })

    print("\nUser Question:")
    print(question)

    print("\nGenerated Answer:")
    print(response)
