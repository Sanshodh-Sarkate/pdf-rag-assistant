from  langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents(documents:list[Document]) -> list[Document]:
    
    text_splitter = RecursiveCharacterTextSplitter( chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""])
    
    chunks  =  text_splitter.split_documents(documents)
    print(f"Total pages: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    return chunks


if __name__ == "__main__":

    from rag_chatbot.data_ingestion.pdf_loader import load_pdf_documents

    documents = load_pdf_documents()

    chunks = split_documents(documents)

    for chunk in chunks[:3]:
        print("\n---------------------------")
        print("Source:", chunk.metadata["source"])
        print("Page:", chunk.metadata["page"])
        print("Content:", chunk.page_content[:300])
