import os  
from dotenv  import  load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import  ChatOllama
load_dotenv()
# model  define  with groq  
def get_llm():
    # groq_api_key = os.getenv('GROQ_API_KEY')
    # if not groq_api_key:
        # raise ValueError("GROQ_API_KEY is missing from .env file")

    try:
        # llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, groq_api_key=groq_api_key)
        llm =  ChatOllama(model="qwen2.5:7b" ,  temperature= 0,  keep_alive=
        "30m",)
        # llm  =  ChatOllama(model="gemma:2b" , temperature = 0 ,)
        # llm =  ChatOllama(model="gemma3:4b" , temperature = 0 ,    num_ctx=2048,
        #     num_predict=150,
        #     keep_alive="30m")
        return llm
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ChatOllama LLM: {e}") from e 


if __name__ == "__main__":
    llm = get_llm()

    response = llm.invoke("What is Retrieval Augmented Generation?")

    print(response.content) 
