import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "esg_chatbot")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.575))

MODEL_ARK_API_KEY = os.getenv("MODEL_ARK_API_KEY")
MODEL_ARK_API_URL = os.getenv("MODEL_ARK_API_URL")
MODEL_ARK_LLM_NAME = os.getenv("MODEL_ARK_LLM_NAME")

SYSTEM_PROMPT = """Anda adalah Asisten Advanced di bidang ESG (Environmental, Social, and Governance).

ATURAN PENTING:
1. Jawab HANYA berdasarkan konteks dokumen yang diberikan di bawah ini
2. JANGAN menjawab pertanyaan di luar konteks yang tersedia
3. Jika informasi tidak ditemukan dalam konteks, katakan: "Maaf, informasi tersebut tidak ditemukan dalam dokumen yang tersedia."
4. Selalu sebutkan sumber informasi (nama dokumen, perusahaan, tahun, halaman) saat menjawab
5. Jawab dalam Bahasa Indonesia dengan jelas dan terstruktur
6. Jika ada data angka/metrik, sebutkan dengan jelas beserta satuannya
7. Untuk data emisi GRK, pastikan menyebutkan scope 1, 2, 3 jika tersedia
8. Jika pertanyaan membandingkan beberapa perusahaan, berikan data dari SEMUA perusahaan yang disebutkan

Anda memiliki keahlian dalam:
- Analisis laporan keberlanjutan (sustainability report)
- Metrik ESG dan standar pelaporan (GRI, SASB, TCFD)
- Data emisi Gas Rumah Kaca (GRK) - Scope 1, 2, 3
- Konsumsi energi dan air
- Data ketenagakerjaan dan keragaman
- Tata kelola perusahaan dan etika bisnis"""


def search_context(client: QdrantClient, model: SentenceTransformer, query: str, top_k: int = 10) -> list:
    """Search Qdrant collection and return context documents."""
    query_vector = model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        score_threshold=SCORE_THRESHOLD
    ).points

    contexts = []
    for result in results:
        contexts.append({
            "content": result.payload.get("content", ""),
            "nama_perusahaan": result.payload.get("nama_perusahaan", ""),
            "sumber_file": result.payload.get("sumber_file", ""),
            "metadata": result.payload.get("metadata", {}),
            "score": result.score
        })

    return contexts


def format_context(contexts: list) -> str:
    """Format contexts into a string for LLM prompt."""
    if not contexts:
        return "Tidak ada dokumen yang ditemukan."

    formatted = []
    for i, ctx in enumerate(contexts, 1):
        metadata = ctx.get("metadata", {})
        page_range = metadata.get("page_range", "N/A")
        section = metadata.get("section", "N/A")

        formatted.append(f"""
--- Dokumen {i} ---
Perusahaan: {ctx['nama_perusahaan']}
Sumber: {ctx['sumber_file']}
Halaman: {page_range}
Section: {section}
Relevansi Score: {ctx['score']:.4f}

Konten:
{ctx['content']}
""")

    return "\n".join(formatted)


def chat_with_llm(llm_client: OpenAI, query: str, context: str) -> str:
    """Send query with context to LLM and get response."""
    user_message = f"""KONTEKS DOKUMEN:
{context}

PERTANYAAN PENGGUNA:
{query}

Berdasarkan konteks dokumen di atas, jawab pertanyaan pengguna dengan lengkap dan akurat."""

    response = llm_client.chat.completions.create(
        model=MODEL_ARK_LLM_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=2048
    )

    return response.choices[0].message.content


def main():
    print("=" * 50)
    print("ESG Chatbot")
    print("=" * 50)

    print(f"\nConnecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Connecting to LLM: {MODEL_ARK_LLM_NAME}")
    llm_client = OpenAI(
        api_key=MODEL_ARK_API_KEY,
        base_url=MODEL_ARK_API_URL
    )

    print("\nReady! Ketik 'exit' atau 'quit' untuk keluar\n")

    while True:
        query = input("Anda: ").strip()

        if query.lower() in ["exit", "quit"]:
            print("Bye!")
            break

        if not query:
            print("Pertanyaan tidak boleh kosong\n")
            continue

        print("\nMencari dokumen relevan...")
        contexts = search_context(qdrant_client, embedding_model, query, top_k=10)
        print(f"Ditemukan {len(contexts)} dokumen relevan")

        context_str = format_context(contexts)

        print("Generating response...\n")
        response = chat_with_llm(llm_client, query, context_str)

        print(f"Asisten: {response}\n")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
