from pathlib import Path

## pyPdf use to  load the PDF and extract the text from the pdf 
from langchain_community.document_loaders import  PyPDFLoader
from langchain_core.documents import Document


## project root directory  
BASE_DIR = Path(__file__).resolve().parents[3]

# PDF documents directory
DOCUMENTS_DIR = BASE_DIR / "documents"

# this function for the loadPdf  and return  list  
def load_pdf_documents()->list[Document]:
    """
    Load all PDF files from the documents directory.

    Returns:
        list[Document]: Extracted PDF pages with metadata.
    """
    
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIR}"
        )
        
        
    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
        
    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files found in the documents directory."
        )
        
    all_documents = []

    for pdf_file in pdf_files:
        try:
            print(f"Loading PDF: {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()

            for page in pages:
                page.metadata["source"] = pdf_file.name

                # Convert PyPDFLoader's zero-based page
                # into a human-readable one-based page number.
                if "page" in page.metadata and isinstance(page.metadata["page"], (int, float)):
                    page.metadata["page"] = int(page.metadata["page"]) + 1

            all_documents.extend(pages)
        except Exception as e:
            print(f"Error loading PDF file '{pdf_file.name}': {e}")

    print(f"\nTotal PDF files processed: {len(pdf_files)}")
    print(f"Total pages extracted: {len(all_documents)}")

    return all_documents




if __name__ == "__main__":

    documents = load_pdf_documents()
    for document in documents[:3]:

        print("\n---------------------------")
        print("Source:", document.metadata["source"])
        print("Page:", document.metadata["page"])
        print("Content:", document.page_content[:300])


    
