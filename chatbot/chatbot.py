import os
import json
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "esg_chatbot")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.3))

MODEL_ARK_API_KEY = os.getenv("MODEL_ARK_API_KEY")
MODEL_ARK_API_URL = os.getenv("MODEL_ARK_API_URL")
MODEL_ARK_LLM_NAME = os.getenv("MODEL_ARK_LLM_NAME")

TOP_K = 20
CHATBOT_DIR = Path(__file__).parent
OUTPUT_FILE = CHATBOT_DIR / "chat_results.json"

COMPANY_ALIASES = {
    "bank jago": "PT Bank Jago Tbk",
    "jago": "PT Bank Jago Tbk",
    "bank jatim": "PT Bank Pembangunan Daerah Jawa Timur Tbk",
    "jatim": "PT Bank Pembangunan Daerah Jawa Timur Tbk",
}

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


class MergedResult:
    """Wrapper untuk hasil merge dengan score."""
    def __init__(self, id, score, payload):
        self.id = id
        self.score = score
        self.payload = payload


class ESGChatbot:
    """ESG Chatbot dengan RAG menggunakan Qdrant dan ModelArk LLM."""

    def __init__(self):
        print("=" * 60)
        print("ESG CHATBOT - Advanced ESG Assistant")
        print("=" * 60)
        print("Initializing...")

        print(f"  Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        print(f"  Connecting to Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        print(f"  Collection: {COLLECTION_NAME}")
        print(f"  LLM: {MODEL_ARK_LLM_NAME}")
        print("  Ready!\n")

    def _extract_companies(self, query: str) -> list:
        """Extract company names dari query."""
        query_lower = query.lower()
        companies = set()
        for alias, canonical in COMPANY_ALIASES.items():
            if alias in query_lower:
                companies.add(canonical)
        return list(companies)

    def search(self, query: str, company_filter: list = None, top_k: int = TOP_K) -> list:
        """Simple semantic search dengan optional company filter."""
        query_with_prefix = f"query: {query}"
        query_embedding = self.embedding_model.encode(query_with_prefix).tolist()

        query_filter = None
        if company_filter and len(company_filter) > 0:
            conditions = [
                FieldCondition(key="nama_perusahaan", match=MatchValue(value=company))
                for company in company_filter
            ]
            query_filter = Filter(should=conditions)

        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=SCORE_THRESHOLD
        )

        return results.points

    def hybrid_search(self, query: str, top_k: int = TOP_K) -> list:
        """Search dengan company filter jika ada multiple companies."""
        companies = self._extract_companies(query)

        print(f"[Search] Query: \"{query}\"")
        print(f"[Search] Companies: {companies if companies else 'None'}")

        merged_results = []

        if companies and len(companies) > 1:
            # Multi-company: search per company untuk hasil seimbang
            per_company_k = top_k
            print(f"[Search] Multi-company mode: {len(companies)} companies")

            for company in companies:
                results = self.search(query, company_filter=[company], top_k=per_company_k)
                print(f"[Search]   {company}: {len(results)} results")

                for result in results:
                    merged_results.append(MergedResult(
                        id=result.id,
                        score=result.score,
                        payload=result.payload
                    ))

            merged_results.sort(key=lambda x: x.score, reverse=True)

        else:
            # Single company atau tanpa filter
            results = self.search(query, company_filter=companies, top_k=top_k)
            print(f"[Search] Found: {len(results)} results")

            for result in results:
                merged_results.append(MergedResult(
                    id=result.id,
                    score=result.score,
                    payload=result.payload
                ))

        return merged_results[:top_k * 2] if len(companies) > 1 else merged_results[:top_k]

    def format_context(self, results: list) -> str:
        """Format hasil search menjadi context untuk LLM."""
        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            payload = result.payload
            content = payload.get("content", "")
            metadata = payload.get("metadata", {})

            context_parts.append(
                f"[Dokumen {i}]\n"
                f"Perusahaan: {payload.get('nama_perusahaan', 'N/A')}\n"
                f"Sumber: {payload.get('sumber_file', 'N/A')}\n"
                f"Halaman: {metadata.get('page_range', 'N/A')}\n"
                f"Section: {metadata.get('section', 'N/A')}\n"
                f"Relevansi: {result.score:.4f}\n"
                f"Konten:\n{content}\n"
            )

        return "\n---\n".join(context_parts)

    def call_llm(self, query: str, context: str) -> tuple:
        """Call ModelArk API untuk generate response."""
        user_message = f"""Konteks dari dokumen:
{context}

Pertanyaan: {query}"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MODEL_ARK_API_KEY}"
        }

        payload = {
            "model": MODEL_ARK_LLM_NAME,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000
        }

        try:
            response = requests.post(
                f"{MODEL_ARK_API_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            usage = result.get("usage", {})
            token_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }

            return content, token_usage

        except requests.exceptions.RequestException as e:
            return f"Error memanggil LLM: {str(e)}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def format_search_results(self, results: list) -> list:
        """Format hasil search untuk output JSON."""
        formatted = []
        for i, result in enumerate(results, 1):
            payload = result.payload
            metadata = payload.get("metadata", {})
            formatted.append({
                "rank": i,
                "score": round(result.score, 4),
                "id": str(result.id),
                "nama_perusahaan": payload.get("nama_perusahaan", "N/A"),
                "sumber_file": payload.get("sumber_file", "N/A"),
                "page_range": metadata.get("page_range", "N/A"),
                "section": metadata.get("section", "N/A"),
                "content": payload.get("content", "")[:500] + "..." if len(payload.get("content", "")) > 500 else payload.get("content", "")
            })
        return formatted

    def save_results(self, query: str, results: list, response: str, token_usage: dict):
        """Save results ke file JSON."""
        formatted_results = self.format_search_results(results)

        output_data = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": MODEL_ARK_LLM_NAME,
            "collection": COLLECTION_NAME,
            "total_results": len(formatted_results),
            "token_usage": token_usage,
            "response": response,
            "search_results": formatted_results
        }

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"[Output] Results saved to: {OUTPUT_FILE}")

    def chat(self, query: str) -> str:
        """Process query dan generate response."""
        results = self.hybrid_search(query, top_k=TOP_K)

        if not results:
            return "Maaf, tidak ditemukan dokumen yang relevan dengan pertanyaan Anda."

        context = self.format_context(results)
        print(f"[Context] Size: {len(context):,} characters")

        print("[LLM] Generating response...")
        response, token_usage = self.call_llm(query, context)

        print(f"[Token Usage] Input: {token_usage['prompt_tokens']:,} | Output: {token_usage['completion_tokens']:,} | Total: {token_usage['total_tokens']:,}")

        self.save_results(query, results, response, token_usage)

        return response


def main():
    chatbot = ESGChatbot()

    print("=" * 60)
    print("Selamat datang di ESG Chatbot!")
    print("Ketik 'quit' atau 'exit' untuk keluar")
    print("=" * 60)

    while True:
        try:
            query = input("\nAnda: ").strip()

            if not query:
                print("Pertanyaan tidak boleh kosong!")
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("\nTerima kasih! Sampai jumpa!")
                break

            response = chatbot.chat(query)

            print("\n" + "=" * 60)
            print("ESG Assistant:")
            print("=" * 60)
            print(response)
            print("=" * 60)

        except KeyboardInterrupt:
            print("\n\nTerima kasih! Sampai jumpa!")
            break


if __name__ == "__main__":
    main()
