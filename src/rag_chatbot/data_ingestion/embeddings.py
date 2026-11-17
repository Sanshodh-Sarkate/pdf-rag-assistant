from langchain_huggingface import HuggingFaceEmbeddings

## embedding model
def get_embedding_model():
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return embedding_model
    except Exception as e:
        raise RuntimeError(f"Failed to load HuggingFace embedding model: {e}") from e


if __name__ == "__main__":

    embedding_model = get_embedding_model()

    text = "Employees are entitled to annual paid leave."

    vector = embedding_model.embed_query(text)

    print("Embedding:", vector)
    print("Vector Dimensions:", len(vector))