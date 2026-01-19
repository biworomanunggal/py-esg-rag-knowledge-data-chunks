#!/usr/bin/env python3
"""
ESG Chatbot dengan RAG
=======================
Chatbot untuk menjawab pertanyaan ESG menggunakan:
- Qdrant untuk vector search
- E5 embedding model
- ModelArk LLM untuk generate response
"""

import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText, MatchValue
from sentence_transformers import SentenceTransformer

load_dotenv()

# Konfigurasi Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Collection untuk E5 embeddings
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "esg_dashboard_knowladge")

# E5: Multilingual model dengan dimensi 768
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

# Konfigurasi ModelArk Byteplus
MODEL_ARK_API_KEY = os.getenv("MODEL_ARK_API_KEY")
MODEL_ARK_API_URL = os.getenv("MODEL_ARK_API_URL")
MODEL_ARK_LLM_NAME = os.getenv("MODEL_ARK_LLM_NAME")

# Search configuration
TOP_K = 40
SCORE_THRESHOLD = 0.3

# Output file
OUTPUT_FILE = "llm/chat_results.json"

# Keyword synonyms untuk hybrid search
KEYWORD_SYNONYMS = {
    "emisi": ["emisi", "emission", "grk", "ghg", "karbon", "carbon", "co2", "scope 1", "scope 2", "scope 3", "cakupan 1", "cakupan 2", "cakupan 3", "tco2", "kgco2", "gas rumah kaca"],
    "karyawan": ["karyawan", "employee", "pegawai", "sdm", "human capital", "workforce", "laki-laki", "perempuan", "male", "female", "tenaga kerja"],
    "energi": ["energi", "energy", "listrik", "electricity", "kwh", "mwh", "konsumsi energi"],
    "air": ["air", "water", "pdam", "konsumsi air"],
    "limbah": ["limbah", "waste", "sampah", "b3"],
    "lingkungan": ["lingkungan", "environment", "environmental", "biaya lingkungan", "pengelolaan lingkungan"],
    "biaya": ["biaya", "cost", "anggaran", "budget", "alokasi", "allocation", "investasi", "investment", "pengeluaran", "expenditure"],
}

# Company aliases
COMPANY_ALIASES = {
    "bank jago": "Bank Jago",
    "jago": "Bank Jago",
    "bank jatim": "Bank Pembangunan Daerah Jawa Timur",
    "jatim": "Bank Pembangunan Daerah Jawa Timur",
}


class MergedResult:
    """Wrapper untuk hasil merge dengan score."""
    def __init__(self, id, score, payload):
        self.id = id
        self.score = score
        self.payload = payload

# System prompt untuk ESG Assistant
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


class ESGChatbot:
    """ESG Chatbot dengan RAG menggunakan Qdrant dan ModelArk LLM."""

    def __init__(self):
        print("=" * 60)
        print("ESG CHATBOT - Advanced ESG Assistant")
        print("=" * 60)
        print("Initializing...")

        # Load E5 embedding model
        print(f"  Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # Connect to Qdrant
        print(f"  Connecting to Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        print(f"  Collection: {COLLECTION_NAME}")
        print(f"  LLM: {MODEL_ARK_LLM_NAME}")
        print("  Ready!\n")

    def _extract_keywords(self, query: str) -> list:
        """Extract keywords dari query dan expand dengan synonyms."""
        query_lower = query.lower()
        keywords = set()

        for key, synonyms in KEYWORD_SYNONYMS.items():
            for syn in synonyms:
                if syn in query_lower:
                    keywords.update(synonyms)
                    break

        return list(keywords)

    def _extract_companies(self, query: str) -> list:
        """Extract multiple company names dari query (untuk perbandingan)."""
        query_lower = query.lower()
        companies = set()
        for alias, canonical in COMPANY_ALIASES.items():
            if alias in query_lower:
                companies.add(canonical)
        return list(companies)

    def semantic_search_with_keywords(self, query: str, company_filter: list = None,
                                       keywords: list = None, top_k: int = 50,
                                       score_threshold: float = SCORE_THRESHOLD) -> list:
        """
        Semantic search dengan keyword filter (seperti SQL AND).
        Keywords digunakan sebagai filter tambahan pada semantic search.
        Supports multiple companies (OR logic).
        """
        # E5: gunakan prefix "query:" untuk query
        query_with_prefix = f"query: {query}"
        query_embedding = self.embedding_model.encode(query_with_prefix).tolist()

        # Build filter conditions
        # Company filter - support multiple companies dengan OR logic
        company_conditions = []
        if company_filter and len(company_filter) > 0:
            for company in company_filter:
                company_conditions.append(
                    FieldCondition(key="company", match=MatchValue(value=company))
                )

        # Keyword filter - gunakan should untuk OR antar keywords
        keyword_conditions = []
        if keywords:
            for keyword in keywords[:5]:  # Limit 5 keywords utama
                keyword_conditions.append(
                    FieldCondition(key="content", match=MatchText(text=keyword))
                )

        # Build final filter
        # Logic: (company1 OR company2) AND (keyword1 OR keyword2 OR ...)
        query_filter = None

        if company_conditions and keyword_conditions:
            # Jika ada company dan keywords: filter by company (OR) AND keywords (OR)
            query_filter = Filter(
                should=company_conditions,  # company1 OR company2
                must=[Filter(should=keyword_conditions)]  # AND (keyword1 OR keyword2)
            )
        elif company_conditions:
            # Hanya company filter
            query_filter = Filter(should=company_conditions)
        elif keyword_conditions:
            # Hanya keyword filter
            query_filter = Filter(should=keyword_conditions)

        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold
        )

        return results.points

    def hybrid_search(self, query: str, top_k: int = TOP_K) -> list:
        """
        Hybrid search: Semantic search dengan keyword filter terintegrasi.
        Keywords digunakan sebagai filter (AND) bukan search terpisah.
        Supports multiple companies untuk perbandingan - search per company untuk balanced results.
        """
        keywords = self._extract_keywords(query)
        companies = self._extract_companies(query)  # Support multiple companies

        print(f"[Search] Query: \"{query}\"")
        print(f"[Search] Keywords: {keywords[:5] if keywords else 'None'}")
        print(f"[Search] Companies filter: {companies if companies else 'None'}")

        merged_results = []

        # Jika ada multiple companies, search per company untuk hasil yang seimbang
        if companies and len(companies) > 1:
            # Untuk perbandingan, ambil lebih banyak hasil per company
            # agar data detail (seperti tabel scope 1/2/3) tidak terpotong
            per_company_top_k = max(top_k, 30)  # Minimal 30 per company

            print(f"[Search] Multi-company mode: {len(companies)} companies, {per_company_top_k} results each")

            for company in companies:
                results = self.semantic_search_with_keywords(
                    query,
                    company_filter=[company],  # Single company
                    keywords=keywords,
                    top_k=per_company_top_k,
                    score_threshold=SCORE_THRESHOLD
                )
                print(f"[Search]   {company}: {len(results)} results")

                for result in results:
                    merged_results.append(MergedResult(
                        id=result.id,
                        score=result.score,
                        payload=result.payload
                    ))

            # Sort by score descending
            merged_results.sort(key=lambda x: x.score, reverse=True)
            print(f"[Search] Total merged: {len(merged_results)} results")

        else:
            # Single company atau tanpa company filter
            results = self.semantic_search_with_keywords(
                query,
                company_filter=companies,  # List of companies (OR logic)
                keywords=keywords,
                top_k=top_k,
                score_threshold=SCORE_THRESHOLD
            )

            if keywords and companies:
                print(f"[Search] Semantic + Keywords + Companies: {len(results)} results")
            elif keywords:
                print(f"[Search] Semantic + Keywords: {len(results)} results")
            elif companies:
                print(f"[Search] Semantic + Companies: {len(results)} results")
            else:
                print(f"[Search] Semantic only: {len(results)} results")

            # Convert ke MergedResult untuk konsistensi
            for result in results:
                merged_results.append(MergedResult(
                    id=result.id,
                    score=result.score,
                    payload=result.payload
                ))

        return merged_results[:top_k]

    def format_context(self, results: list, max_content_length: int = 1500) -> str:
        """Format hasil search menjadi context untuk LLM."""
        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            payload = result.payload
            content = payload.get("content", "")
            if len(content) > max_content_length:
                content = content[:max_content_length] + "... [dipotong]"

            context_parts.append(
                f"[Dokumen {i}]\n"
                f"Sumber: {payload.get('source_document', 'N/A')}\n"
                f"Perusahaan: {payload.get('company', 'N/A')}\n"
                f"Tahun: {payload.get('report_year', 'N/A')}\n"
                f"Halaman: {payload.get('page', 'N/A')}\n"
                f"Section: {payload.get('section_name', 'N/A')}\n"
                f"Relevansi: {result.score:.4f}\n"
                f"Konten:\n{content}\n"
            )

        return "\n---\n".join(context_parts)

    def call_llm(self, query: str, context: str) -> tuple:
        """Call ModelArk Byteplus API untuk generate response."""
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
            formatted.append({
                "rank": i,
                "score": round(result.score, 4),
                "id": str(result.id),
                "company": payload.get("company", "N/A"),
                "source_document": payload.get("source_document", "N/A"),
                "report_year": payload.get("report_year", "N/A"),
                "page": payload.get("page", "N/A"),
                "section_name": payload.get("section_name", "N/A"),
                "content": payload.get("content", "")[:500] + "..." if len(payload.get("content", "")) > 500 else payload.get("content", "")
            })
        return formatted

    def save_results(self, query: str, results: list, response: str, token_usage: dict):
        """Save search results dan response ke file JSON."""
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
        # Hybrid search (semantic + keyword)
        results = self.hybrid_search(query, top_k=TOP_K)

        if not results:
            return "Maaf, tidak ditemukan dokumen yang relevan dengan pertanyaan Anda."

        # Format context
        context = self.format_context(results)
        context_chars = len(context)
        print(f"[Context] Size: {context_chars:,} characters")

        # Call LLM
        print("[LLM] Generating response...")
        response, token_usage = self.call_llm(query, context)

        # Print token usage
        print(f"[Token Usage] Input: {token_usage['prompt_tokens']:,} | Output: {token_usage['completion_tokens']:,} | Total: {token_usage['total_tokens']:,}")

        # Save results to JSON
        self.save_results(query, results, response, token_usage)

        return response


def main():
    chatbot = ESGChatbot()

    print("=" * 60)
    print("Selamat datang di ESG Chatbot!")
    print("Saya adalah Asisten Advanced di bidang ESG.")
    print("Silakan ajukan pertanyaan tentang laporan keberlanjutan.")
    print("=" * 60)
    print("Ketik 'quit' atau 'exit' untuk keluar")

    while True:
        try:
            query = input("\nAnda: ").strip()

            if not query:
                print("Pertanyaan tidak boleh kosong!")
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("\nTerima kasih telah menggunakan ESG Chatbot. Sampai jumpa!")
                break

            response = chatbot.chat(query)

            print("\n" + "=" * 60)
            print("ESG Assistant:")
            print("=" * 60)
            print(response)
            print("=" * 60)

        except KeyboardInterrupt:
            print("\n\nTerima kasih telah menggunakan ESG Chatbot. Sampai jumpa!")
            break


if __name__ == "__main__":
    main()
