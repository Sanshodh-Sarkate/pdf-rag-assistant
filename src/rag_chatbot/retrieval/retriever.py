from langchain_core.documents import Document 
from  rag_chatbot.database.vector_store import get_vector_store


## This function is retrive the  releavent document search and return some  socore  here  k means return the maximum 4 result  
def  retrieve_documents(query:str , k: int = 4)->list[Document]:
    ##Get the existing vector data from astra db 
    vector_store = get_vector_store()
    # print(vector_store)
    
    # Search for the most relevant document chunks
    results  = vector_store.similarity_search(query=query , k=k)
    
    return results





if __name__ == "__main__":
    
    question = "What are the responsibilities of employees and managers?"

    documents = retrieve_documents(question)

    print(f"\nUser Question: {question}")
    print(f"Retrieved Documents: {len(documents)}")

    for i, document in enumerate(documents, start=1):

        print("\n---------------------------")
        print(f"Result: {i}")
        print("Source:", document.metadata.get("source"))
        print("Page:", document.metadata.get("page"))
        print("Content:", document.page_content)    