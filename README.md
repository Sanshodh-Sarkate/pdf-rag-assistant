# 📚 PDF Conversational RAG Chatbot

An intelligent, production-grade **Conversational Retrieval-Augmented Generation (RAG)** chatbot built with **LangChain**, **LangGraph**, **Streamlit**, **DataStax Astra DB (Cassandra)**, **HuggingFace Embeddings**, and **Groq Cloud**.

The system ingests internal PDF documents (e.g., Company Policies, Operations Manuals, SOPs), indexes them into vector storage, and provides strict, context-grounded conversational answers complete with exact source filenames and 1-based page citations.

---

## 🌟 Key Features

- **Strict Factual Grounding**: Answers queries exclusively using retrieved PDF context with exact source citations (`Source: <filename>`, `Page: <page>`). Out-of-scope questions trigger an explicit refusal rather than hallucinations.
- **Conversational Memory & Follow-up Rewriting**: Built on a **LangGraph StateGraph** workflow. Contextualizes multi-turn follow-up questions (e.g., *"What about managers?"*) into standalone search queries based on chat history.
- **Persistent Chat History**: Survives browser reloads (F5) and server restarts via disk-backed session storage (`data/chat_history.json`).
- **Multi-Chat Session Management**: Create new chats, navigate previous conversations via the Streamlit sidebar, and clear history on demand.
- **Production-Ready Error Handling**: Comprehensive `try-except` safeguards across the UI, LLM APIs, vector store connections, and PDF ingestion to prevent crashes.
- **High-Performance Vector Search**: Powered by DataStax Astra DB (CassIO) with cosine similarity filtering (`SIMILARITY_THRESHOLD = 0.65`) and HuggingFace MiniLM embeddings.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["📄 Data Ingestion Pipeline"]
        PDF["PDF Documents (documents/)"] -->|PyPDFLoader| Extracted["Extracted Pages + Metadata"]
        Extracted -->|RecursiveCharacterTextSplitter| Chunks["Text Chunks (1000 chars, 200 overlap)"]
        Chunks -->|HuggingFace Embeddings| Vectors["Vector Embeddings (384-dim)"]
        Vectors -->|CassIO / Cassandra| AstraDB[("DataStax Astra DB")]
    end

    subgraph Chatbot ["🤖 Conversational RAG Execution"]
        User["User Question (Streamlit UI)"] --> InputNode["generate_answer (MessagesState)"]
        InputNode --> Rewriter["Contextualize Chain (Groq LLM)"]
        Rewriter -->|Standalone Question| Search["Astra DB Vector Similarity Search"]
        Search -->|Top-k Chunks (score >= 0.65)| Filter{"Relevant Docs Found?"}
        Filter -->|No| Fallback["'I could not find relevant information...'"]
        Filter -->|Yes| Prompt["System Prompt + Document Context"]
        Prompt --> GenLLM["Groq LLM (openai/gpt-oss-20b)"]
        GenLLM --> Answer["Answer + Citations"]
        Fallback --> OutputNode["AIMessage"]
        Answer --> OutputNode
    end

    subgraph Storage ["💾 Persistence Layer"]
        OutputNode --> DiskStore["data/chat_history.json"]
        DiskStore <-->|Sync State| Checkpointer["LangGraph Memory Checkpointer"]
    end
```

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **User Interface** | [Streamlit](https://streamlit.io/) | Interactive chat UI with sidebar history management |
| **Orchestration** | [LangGraph](https://langchain-ai.github.io/langgraph/) & [LangChain](https://www.langchain.com/) | StateGraph agent workflow & memory checkpointer |
| **LLM Provider** | [Groq](https://groq.com/) | Ultra-fast inference with `openai/gpt-oss-20b` |
| **Embeddings** | [HuggingFace](https://huggingface.co/) (`sentence-transformers/all-MiniLM-L6-v2`) | Local, lightweight 384-dimensional text embeddings |
| **Vector Database** | [DataStax Astra DB](https://astra.datastax.com/) / [CassIO](https://cassio.org/) | Cloud Cassandra vector store (`pdf_rag_documents`) |
| **Document Loader** | `pypdf` / `langchain-community` | PDF loading and metadata tracking (1-based pages) |
| **Package Manager** | [uv](https://docs.astral.sh/uv/) | Fast Python package management |

---

## 📁 Project Directory Structure

```text
RAG_ChatBot/
├── .devcontainer/                  # Dev container configuration
├── data/                           # Runtime data storage
│   └── chat_history.json           # Persistent multi-thread chat history (git-ignored)
├── documents/                      # Target PDF documents for RAG indexing
│   ├── Company_Policies_Sample_50_Pages (1).pdf
│   ├── Operations_Manual_Sample_50_Pages.pdf
│   └── Standard_Operating_Procedures_Sample_50_Pages.pdf
├── logs/                           # Application and debugging logs
├── src/
│   └── rag_chatbot/
│       ├── chains/                 # LangChain & LangGraph chain definitions
│       │   ├── conversation/       # Conversational memory and graph nodes
│       │   │   ├── chat_history.py         # Memory state utilities
│       │   │   ├── contextualize_chain.py  # Standalone question rewriting chain
│       │   │   ├── contextualize_prompt.py # Question rewrite prompt template
│       │   │   └── conversation_chain.py   # Compiled LangGraph StateGraph
│       │   ├── llm.py              # Groq LLM factory with validation
│       │   ├── prompt.py           # Strict citation system prompt
│       │   └── rag_chain.py        # Core RAG retrieval & generation runnable
│       ├── database/               # Database integrations
│       │   ├── cassandra_client.py # Astra DB connection initializer
│       │   └── vector_store.py     # Cassandra vector store abstraction
│       ├── data_ingestion/         # Ingestion and indexing pipeline
│       │   ├── embeddings.py       # HuggingFace embedding model loader
│       │   ├── pdf_loader.py       # PDF file loader & 1-based page metadata
│       │   └── text_splitter.py    # Recursive chunking logic
│       ├── retrieval/              # Retrieval logic
│       │   └── retriever.py        # Similarity search with threshold filtering
│       └── utils/                  # Helper utilities
│           └── storage.py          # Disk persistence & LangGraph memory sync
├── .env.example                    # Template for required environment variables
├── .gitignore                      # Git ignore file
├── app.py                          # Streamlit web application entrypoint
├── DOCUMENTATION.md                # Comprehensive technical documentation
├── pyproject.toml                  # Project metadata and dependencies
├── requirements.txt                # Standard pip dependencies
└── README.md                       # Main project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.11 - 3.14)
- **uv** (recommended) or standard `pip`
- **Astra DB Account**: A vector-enabled database with Application Token and Database ID.
- **Groq API Key**: Obtainable from the [Groq Console](https://console.groq.com/).

---

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Sanshodh-Sarkate/RAG_ChatBot.git
cd RAG_ChatBot

# Using uv (recommended)
uv sync

# Or using pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

### 3. Environment Variables Configuration

Create a `.env` file in the project root with your credentials:

```ini
# Groq API Configuration
GROQ_API_KEY="gsk_your_groq_api_key_here"

# DataStax Astra DB Configuration
ASTRA_DB_APPLICATION_TOKEN="AstraCS:your_application_token_here"
ASTRA_DB_ID="your_database_id_uuid"
ASTRA_DB_KEYSPACE="default_keyspace"
```

---

### 4. Data Ingestion (One-time Setup)

Place your PDF documents inside the `documents/` folder, then run the ingestion pipeline to split and store document embeddings in Astra DB:

```bash
# Using uv
uv run python -m rag_chatbot.database.vector_store

# Or using standard python
python -m src.rag_chatbot.database.vector_store
```

---

### 5. Running the Application

Launch the Streamlit web interface:

```bash
uv run streamlit run app.py
```

The app will be accessible at `http://localhost:8501`.

---

## 🧪 Testing & Verification Suite

Use these verified example questions to test all capabilities of the chatbot:

### Test 1: Single-Turn Factual Question
* **Query:** `What details are required during request intake according to the operations manual?`
* **Expected Output:**
  - Requester
  - Summary
  - Business impact
  - Desired outcome
  - Supporting evidence
  - **Source:** `Operations_Manual_Sample_50_Pages.pdf` | **Page:** `12`

### Test 2: Conversational Multi-Turn Follow-up
* **Turn 1:** `What are the responsibilities of employees according to company policy?`
  - *(Answers employee duties from `Company_Policies_Sample_50_Pages (1).pdf`, Page 4)*
* **Turn 2 (Follow-up):** `What about managers?`
  - *(Correctly context-rewrites the question and answers manager responsibilities from Page 4)*

### Test 3: Out-of-Scope / Rejection Test
* **Query:** `What is quantum computing?`
* **Expected Output:**
  `I could not find relevant information in the provided PDF documents.`

---

## 🔒 Security & Best Practices

- **Zero Credentials in Git**: `.env` and `data/*.json` are explicitly ignored in `.gitignore`.
- **Fault-Tolerant Execution**: All network calls, API requests, and database queries are guarded with structured exception handling.
- **Document Integrity**: Source page numbers are adjusted from 0-based indexing to 1-based human-readable pages during ingestion and sanitized to integers (`Page: 12` instead of `Page: 12.0`).

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.
