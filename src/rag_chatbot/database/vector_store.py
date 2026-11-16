from langchain_community.vectorstores import Cassandra

from rag_chatbot.database.cassandra_client import connect_to_cassandra
from rag_chatbot.data_ingestion.embeddings import get_embedding_model

# this for getting the vector store
def get_vector_store():
    try:
        # Connect to Astra DB
        connect_to_cassandra()

        # Load embedding model
        embedding_model = get_embedding_model()

        # Create Cassandra vector store
        vector_store = Cassandra(
            embedding=embedding_model,
            table_name="pdf_rag_documents"
        )

        return vector_store
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Cassandra vector store: {e}") from e


## Storing the embeddings in the Db  
def store_documents(chunks):
    if not chunks:
        print("No document chunks to store.")
        return None

    try:
        vector_store = get_vector_store()

        # Store text chunks and their embeddings
        inserted_ids = vector_store.add_documents(chunks)

        print(f"Successfully stored {len(inserted_ids)} chunks in Astra DB.")

        return vector_store
    except Exception as e:
        raise RuntimeError(f"Failed to store documents in Astra DB: {e}") from e


if __name__ == "__main__":

    from rag_chatbot.data_ingestion.pdf_loader import load_pdf_documents
    from rag_chatbot.data_ingestion.text_splitter import split_documents

    documents = load_pdf_documents()
    chunks = split_documents(documents)

    vector_store = store_documents(chunks)