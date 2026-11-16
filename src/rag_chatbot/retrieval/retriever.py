from langchain_core.documents import Document 
from  rag_chatbot.database.vector_store import get_vector_store


##here we  faced the one  problem like  when i pass the  question  out of the  PDF then stil our Astra
## Astra Db return the similarity score so for this we set the  threashHold  ammount  
SIMILARITY_THRESHOLD = 0.65


## This function retrieves relevant document search and returns results above threshold
def retrieve_documents(query: str, k: int = 4) -> list[Document]:
    try:
        ## Get the existing vector data from astra db 
        vector_store = get_vector_store()
        
        # Search for the most relevant document with score 
        results = vector_store.similarity_search_with_score(query=query, k=k)
        
        relevant_documents = []
        for doc, score in results:
            if score >= SIMILARITY_THRESHOLD:
                relevant_documents.append(doc)

        print(f"Relevant Documents: {len(relevant_documents)}")         
        return relevant_documents
    except Exception as e:
        print(f"Error retrieving documents for query '{query}': {e}")
        return []


 

if __name__ == "__main__":

    question = "What are the responsibilities of employees and managers?"
    # question = "What is the LLM?"

    documents = retrieve_documents(question)

    print(f"\nUser Question: {question}")
    print(f"Retrieved Documents: {len(documents)}")

    for i, document in enumerate(documents, start=1):

        print("\n---------------------------")
        print(f"Result: {i}")

        print("Source:", document.metadata.get("source"))
        print("Page:", document.metadata.get("page"))

        print("Content:", document.page_content)
