# ESG Chatbot API

FastAPI application untuk ESG RAG Chatbot - Intelligent ESG Data Assistant.

## Prerequisites

- Python 3.9+
- Qdrant server running (default: localhost:6333)
- Environment variables configured di `.env` (root project)

## Installation

### Opsi 1: Menggunakan venv project utama (Recommended)

```bash
# Dari root project
cd /Users/katadata/KTDT/py-esg-rag

# Activate existing venv
source venv/bin/activate

# Install FastAPI dependencies
pip install fastapi uvicorn[standard]
```

### Opsi 2: Install semua dependencies

```bash
# Dari folder api
cd /Users/katadata/KTDT/py-esg-rag/api

# Install dependencies
pip install -r requirements.txt
```

## Menjalankan API

### Development Mode (dengan auto-reload)

```bash
cd /Users/katadata/KTDT/py-esg-rag/api

# Cara 1: Langsung run main.py
python main.py

# Cara 2: Menggunakan uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

API akan berjalan di: `http://localhost:8000`

## API Documentation

Setelah API berjalan, akses dokumentasi di:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| POST | `/chat` | Main chat endpoint |
| GET | `/sessions` | List semua sessions |
| GET | `/sessions/{id}` | Detail session + history |
| GET | `/companies` | Daftar perusahaan per sektor |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI documentation |
| GET | `/redoc` | ReDoc documentation |

## Contoh Request

### POST /chat

**Request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Berapa jumlah emisi Bank Jago?",
    "session_id": null,
    "new_session": false
  }'
```

**Response:**
```json
{
  "success": true,
  "session_id": "abc12345",
  "response": "Berdasarkan laporan...",
  "token_usage": {
    "prompt_tokens": 1500,
    "completion_tokens": 350,
    "total_tokens": 1850
  },
  "session_token_usage": {
    "total_queries": 1,
    "total_prompt_tokens": 1500,
    "total_completion_tokens": 350,
    "total_tokens": 1850,
    "avg_tokens_per_query": 1850.0
  },
  "analysis": {
    "companies_detected": ["PT Bank Jago Tbk"],
    "topics_detected": ["emisi"],
    "sectors": ["Finance"],
    "is_comparison": false
  },
  "sources": {
    "pdf_count": 10,
    "data_count": 5
  },
  "processing_time_ms": 2340
}
```

### Melanjutkan Session

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Bagaimana dengan Bank Jatim?",
    "session_id": "abc12345",
    "new_session": false
  }'
```

### Membuat Session Baru

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Pertanyaan baru",
    "session_id": "my-custom-id",
    "new_session": true
  }'
```

### GET /companies

```bash
curl http://localhost:8000/companies
```

**Response:**
```json
{
  "success": true,
  "total_companies": 24,
  "total_sectors": 8,
  "companies_by_sector": {
    "Finance": ["PT Bank Amar Indonesia Tbk", "PT Bank Jago Tbk", "..."],
    "Mining": ["PT Adaro Andalan Indonesia Tbk", "..."],
    "...": "..."
  }
}
```

### GET /sessions

```bash
curl http://localhost:8000/sessions
```

### GET /sessions/{id}

```bash
curl http://localhost:8000/sessions/abc12345
```

### GET /health

```bash
curl http://localhost:8000/health
```

## Environment Variables

API menggunakan environment variables dari file `.env` di root project:

```env
# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=esg_reports
QDRANT_COLLECTION_DATA_NAME=esg_data_reports

# Embedding Configuration
EMBEDDING_MODEL=intfloat/multilingual-e5-base
EMBEDDING_DIMENSION=768

# Search Configuration
SCORE_THRESHOLD=0.3

# LLM Configuration
MODEL_ARK_API_KEY=your-api-key
MODEL_ARK_API_URL=https://ark.ap-southeast.bytepluses.com/api/v3
MODEL_ARK_LLM_NAME=gpt-oss-120b-250805
```

## Struktur File

```
api/
├── __init__.py           # Module init
├── main.py               # FastAPI application
├── models.py             # Pydantic models (request/response)
├── chatbot_service.py    # Chatbot service logic
├── requirements.txt      # Dependencies
├── README.md             # Dokumentasi ini
└── chat_history/         # Folder untuk menyimpan session history
```

## Fitur

- **Multi-source retrieval**: Mengambil data dari PDF dan Excel
- **Smart company detection**: Deteksi otomatis nama perusahaan dari query
- **Session management**: Mendukung percakapan berkelanjutan dengan history
- **Token tracking**: Tracking penggunaan token per query dan per session
- **Comparison support**: Mendukung perbandingan hingga 2 perusahaan
- **Query analysis**: Analisis topik ESG dan sektor secara otomatis
