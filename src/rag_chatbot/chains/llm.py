import os  
from dotenv  import  load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

def get_llm():
    groq_api_key  =  os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing from .env file")
    
    llm = ChatGroq(model="openai/gpt-oss-20b",  temperature=0, groq_api_key= groq_api_key)
    return llm 


if __name__ == "__main__":
    llm = get_llm()

    response = llm.invoke("What is Retrieval Augmented Generation?")

    print(response.content)
 