# 📑 RAG Document Chatbot (RAG Application)

An offline-capable, interactive Retrieval-Augmented Generation (RAG) chatbot that allows you to upload any PDF document and ask questions about its content. This application uses **LangChain** for orchestration, **Groq (Llama 3.3 70B)** for ultra-fast inference, and local **Hugging Face BGE Embeddings** for precise document retrieval.

---

## 🚀 Features

* **PDF Document Parsing:** Upload any PDF to instantly split and index its pages.
* **In-Memory Vector Store:** Uses Chroma DB natively without bloat to isolate document data seamlessly per session.
* **Local Embeddings:** Generates vector embeddings locally using the highly efficient `BAAI/bge-small-en-v1.5` model.
* **Llama 3.3 Intelligence:** Leverages Groq Cloud API for near-instant, highly context-aware responses.
* **Guardrails:** Built-in instructions prevent the LLM from making up answers if the information isn't found in the document.
* **Clean Session Management:** Sidebar settings to wipe chat histories cleanly and switch documents effortlessly.

---

## 🛠️ Tech Stack

* **Frontend UI:** Streamlit
* **LLM Orchestration:** LangChain / LangChain Core
* **Inference Engine:** ChatGroq (Model: `llama-3.3-70b-versatile`)
* **Vector Database:** Chroma
* **Text Embedding Model:** Hugging Face Transformers (`bge-small-en-v1.5`)

---
