# Policy RAG — Company Policy Q&A System

A production-ready Retrieval-Augmented Generation (RAG) system that answers company policy questions using LLMs with accurate, grounded responses.

## Core Features

- Multi-format ingestion (PDF, DOCX, HTML, Markdown, TXT)
- Recursive chunking with overlap
- Local embeddings using Sentence Transformers
- Vector search using ChromaDB
- LLM answer generation (Groq/OpenAI/Anthropic)
- Flask API + web UI
- Topic guard for off-topic questions

## Architecture Flow

Load documents → Split into chunks → Generate embeddings → Store in vector DB → Query → Retrieve top K → Inject context → Generate answer

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask 3.0.0 |
| RAG Framework | LangChain 0.1.0 |
| Vector Database | ChromaDB 0.4.22 |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| LLM | Groq Llama 3.3 70B |

## Prerequisites

- Python 3.11 (NOT 3.12+)
- Groq API key (free at console.groq.com)

## Installation

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd Project_Final
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
pip install langchain-groq
```

### 5. Configure API Key

Create `.env` file:
```
GROQ_API_KEY=gsk_your-key-here
```

### 6. Run Application
```bash
python -m app.main
```

### 7. Open Browser
Go to: http://localhost:5000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| / | GET | Chat UI |
| /chat | POST | Send question |
| /health | GET | Health check |
| /reindex | POST | Re-index documents |

## Usage Examples

| Question | Answer |
|----------|--------|
| How many PTO days do new employees get? | 15 days |
| What is minimum internet speed for remote work? | 25 Mbps |
| How often must I change my password? | 90 days |

## Project Structure

```
Project_Final/
├── app/
│   ├── main.py
│   ├── rag_pipeline.py
│   ├── vector_store.py
│   ├── document_loader.py
│   └── data/policies/
├── requirements.txt
├── .env
└── README.md
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No module 'langchain_groq' | pip install langchain-groq |
| No LLM provider | Check .env file |
| Port 5000 in use | PORT=8080 python -m app.main |

## Stop Server

Press Ctrl + C