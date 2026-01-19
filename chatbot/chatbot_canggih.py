#!/usr/bin/env python3
"""
ESG RAG Chatbot Canggih
========================
Advanced RAG Chatbot untuk ESG Assistance dengan fitur:
- Multi-stage retrieval (semantic + reranking)
- Smart company detection & comparison
- Context-aware chunking
- Query understanding & expansion
- Structured response generation
"""

import os
import re
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

load_dotenv()

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "esg_chatbot")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.3))

MODEL_ARK_API_KEY = os.getenv("MODEL_ARK_API_KEY")
MODEL_ARK_API_URL = os.getenv("MODEL_ARK_API_URL")
MODEL_ARK_LLM_NAME = os.getenv("MODEL_ARK_LLM_NAME")

CHATBOT_DIR = Path(__file__).parent
OUTPUT_FILE = CHATBOT_DIR / "chat_canggih_results.json"

# Company mapping dengan variasi nama
COMPANY_MAP = {
    # Bank Jago variants
    "bank jago": "PT Bank Jago Tbk",
    "jago": "PT Bank Jago Tbk",
    "pt bank jago": "PT Bank Jago Tbk",
    # Bank Jatim variants
    "bank jatim": "PT Bank Pembangunan Daerah Jawa Timur Tbk",
    "jatim": "PT Bank Pembangunan Daerah Jawa Timur Tbk",
    "bpd jatim": "PT Bank Pembangunan Daerah Jawa Timur Tbk",
    "bank pembangunan daerah jawa timur": "PT Bank Pembangunan Daerah Jawa Timur Tbk",
    # Bank OCBC NISP variants
    "ocbc": "PT Bank OCBC NISP Tbk",
    "ocbc nisp": "PT Bank OCBC NISP Tbk",
    "bank ocbc": "PT Bank OCBC NISP Tbk",
    "bank ocbc nisp": "PT Bank OCBC NISP Tbk",
    "nisp": "PT Bank OCBC NISP Tbk",
    "pt bank ocbc nisp": "PT Bank OCBC NISP Tbk",
    # Bank Amar Indonesia variants
    "amar": "PT. Bank Amar Indonesia Tbk",
    "bank amar": "PT. Bank Amar Indonesia Tbk",
    "amar bank": "PT. Bank Amar Indonesia Tbk",
    "bank amar indonesia": "PT. Bank Amar Indonesia Tbk",
    "pt bank amar": "PT. Bank Amar Indonesia Tbk",
}

# ESG Topic keywords untuk query expansion - diperluas dengan variasi dalam dokumen
ESG_TOPICS = {
    "emisi": {
        "keywords": ["emisi", "grk", "ghg", "karbon", "carbon", "co2", "scope 1", "scope 2", "scope 3",
                    "cakupan 1", "cakupan 2", "cakupan 3", "gas rumah kaca", "greenhouse", "ton co2",
                    "intensitas emisi", "emission", "pengurangan emisi"],
        "search_queries": ["emisi GRK scope", "total emisi cakupan", "jumlah emisi GRK"],
        "related_sections": ["Kinerja Keberlanjutan", "Keberlanjutan", "Aspek Emisi"]
    },
    "energi": {
        "keywords": ["energi", "energy", "listrik", "electricity", "kwh", "mwh", "gigajoule",
                    "konsumsi energi", "bbm", "solar", "terajoule", "konsumsi listrik"],
        "search_queries": ["konsumsi energi", "penggunaan listrik"],
        "related_sections": ["Kinerja Keberlanjutan", "Keberlanjutan"]
    },
    "air": {
        "keywords": ["air", "water", "pdam", "konsumsi air", "m3", "penggunaan air"],
        "search_queries": ["konsumsi air", "penggunaan air"],
        "related_sections": ["Kinerja Keberlanjutan", "Keberlanjutan"]
    },
    "limbah": {
        "keywords": ["limbah", "waste", "sampah", "b3", "efluen", "pengelolaan limbah"],
        "search_queries": ["pengelolaan limbah", "limbah b3"],
        "related_sections": ["Kinerja Keberlanjutan", "Keberlanjutan"]
    },
    "karyawan": {
        "keywords": ["karyawan", "employee", "pegawai", "sdm", "human capital", "tenaga kerja",
                    "pekerja", "jumlah karyawan", "total karyawan", "pelatihan"],
        "search_queries": ["jumlah karyawan", "total pegawai", "komposisi karyawan"],
        "related_sections": ["Sumber Daya Manusia", "Kinerja Keberlanjutan", "Profil Perusahaan"]
    },
    "tata_kelola": {
        "keywords": ["tata kelola", "governance", "direksi", "komisaris", "komite", "audit", "gcg"],
        "search_queries": ["tata kelola perusahaan", "good corporate governance"],
        "related_sections": ["Tata Kelola", "Audit Internal", "Komite Audit"]
    },
    "keuangan": {
        "keywords": ["keuangan", "financial", "pendapatan", "laba", "aset", "modal", "rupiah"],
        "search_queries": ["kinerja keuangan", "pendapatan bunga"],
        "related_sections": ["Tinjauan Keuangan", "Ikhtisar Keuangan", "Laporan Keuangan"]
    }
}

# System prompt yang lebih detail
SYSTEM_PROMPT = """Anda adalah ESG Expert Assistant yang sangat canggih dan akurat.

KEMAMPUAN ANDA:
1. Menganalisis laporan keberlanjutan (Sustainability Report) dan laporan tahunan
2. Membandingkan data ESG antar perusahaan dengan akurat
3. Mengekstrak metrik kuantitatif (angka, satuan) dengan presisi tinggi
4. Memberikan insight berdasarkan data yang tersedia

ATURAN KETAT:
1. HANYA gunakan informasi dari konteks dokumen yang diberikan
2. Jika data tidak tersedia, katakan dengan jelas: "Data tidak tersedia dalam dokumen"
3. WAJIB menyebutkan sumber: nama perusahaan, dokumen, dan halaman
4. Untuk data numerik, SELALU sertakan satuan dan tahun
5. Jika diminta perbandingan, sajikan dalam format TABEL yang rapi
6. Jawab dalam Bahasa Indonesia yang profesional

FORMAT JAWABAN:
- Gunakan heading markdown (##) untuk struktur
- Gunakan tabel untuk data perbandingan
- Gunakan bullet points untuk daftar
- Sertakan section "Sumber" di akhir jawaban

TOPIK ESG YANG DIKUASAI:
- Environmental: Emisi GRK (Scope 1,2,3), konsumsi energi, air, limbah
- Social: Ketenagakerjaan, keragaman, pelatihan, komunitas
- Governance: Tata kelola, etika bisnis, anti korupsi, kepatuhan"""


@dataclass
class SearchResult:
    """Structured search result."""
    id: str
    score: float
    content: str
    company: str
    source: str
    page: str
    section: str
    subsection: str = ""


@dataclass
class QueryAnalysis:
    """Hasil analisis query."""
    original_query: str
    companies: List[str]
    topics: List[str]
    is_comparison: bool
    expanded_query: str


@dataclass
class TokenStats:
    """Token usage statistics untuk session."""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_queries: int = 0

    def add_usage(self, prompt_tokens: int, completion_tokens: int):
        """Add token usage dari satu query."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.total_queries += 1

    def get_summary(self) -> Dict:
        """Get summary of token usage."""
        return {
            "total_queries": self.total_queries,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_query": round(self.total_tokens / max(1, self.total_queries), 2)
        }


class ESGChatbotCanggih:
    """Advanced ESG RAG Chatbot."""

    def __init__(self):
        self._print_banner()
        self._initialize_components()
        self.token_stats = TokenStats()

    def _print_banner(self):
        print("=" * 70)
        print("  ESG CHATBOT CANGGIH - Advanced ESG Intelligence Assistant")
        print("=" * 70)
        print("  Features:")
        print("    - Smart query understanding & expansion")
        print("    - Multi-company comparison support")
        print("    - Context-aware retrieval")
        print("    - Structured response generation")
        print("=" * 70)

    def _initialize_components(self):
        print("\n[Init] Loading components...")

        print(f"  → Embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        print(f"  → Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        print(f"  → Collection: {COLLECTION_NAME}")
        print(f"  → LLM: {MODEL_ARK_LLM_NAME}")
        print("\n[Init] Ready!\n")

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analisis query untuk memahami intent dan entities."""
        query_lower = query.lower()

        # Detect companies
        companies = []
        for alias, canonical in COMPANY_MAP.items():
            if alias in query_lower:
                if canonical not in companies:
                    companies.append(canonical)

        # Detect comparison intent
        comparison_keywords = ["vs", "versus", "dibanding", "bandingkan", "perbandingan", "compare", "beda", "perbedaan"]
        is_comparison = any(kw in query_lower for kw in comparison_keywords) or len(companies) > 1

        # Detect ESG topics
        detected_topics = []
        for topic, info in ESG_TOPICS.items():
            if any(kw in query_lower for kw in info["keywords"]):
                detected_topics.append(topic)

        # Expand query dengan keywords terkait
        expanded_parts = [query]
        for topic in detected_topics:
            # Tambahkan beberapa keyword utama
            keywords = ESG_TOPICS[topic]["keywords"][:3]
            expanded_parts.extend(keywords)

        expanded_query = " ".join(expanded_parts)

        return QueryAnalysis(
            original_query=query,
            companies=companies,
            topics=detected_topics,
            is_comparison=is_comparison,
            expanded_query=expanded_query
        )

    def search(self, query: str, company_filter: Optional[str] = None, top_k: int = 15) -> List[SearchResult]:
        """Semantic search dengan optional company filter."""
        # E5 model membutuhkan prefix "query:"
        query_with_prefix = f"query: {query}"
        query_embedding = self.embedding_model.encode(query_with_prefix).tolist()

        # Build filter
        query_filter = None
        if company_filter:
            query_filter = Filter(
                must=[FieldCondition(key="nama_perusahaan", match=MatchValue(value=company_filter))]
            )

        # Gunakan score threshold yang lebih rendah untuk coverage lebih baik
        # Reranking akan memfilter hasil yang kurang relevan
        search_threshold = min(SCORE_THRESHOLD, 0.25)

        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=search_threshold
        )

        # Convert to SearchResult objects
        search_results = []
        for r in results.points:
            payload = r.payload
            metadata = payload.get("metadata", {})
            search_results.append(SearchResult(
                id=str(r.id),
                score=r.score,
                content=payload.get("content", ""),
                company=payload.get("nama_perusahaan", ""),
                source=payload.get("sumber_file", ""),
                page=metadata.get("page_range", "N/A"),
                section=metadata.get("section", "N/A"),
                subsection=metadata.get("subsection", "")
            ))

        return search_results

    def smart_retrieve(self, analysis: QueryAnalysis, top_k_per_company: int = 15) -> List[SearchResult]:
        """Smart retrieval dengan multi-query approach untuk coverage lebih baik."""
        all_results = []
        seen_ids = set()

        def add_results(results: List[SearchResult]):
            """Add results while avoiding duplicates."""
            for r in results:
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    all_results.append(r)

        # Buat multiple queries untuk coverage lebih baik
        queries_to_run = [analysis.original_query]

        # Tambahkan search queries dari detected topics
        for topic in analysis.topics:
            if topic in ESG_TOPICS and "search_queries" in ESG_TOPICS[topic]:
                queries_to_run.extend(ESG_TOPICS[topic]["search_queries"][:2])

        # Tambahkan expanded query jika berbeda
        if analysis.expanded_query != analysis.original_query:
            queries_to_run.append(analysis.expanded_query)

        # Deduplicate queries
        queries_to_run = list(dict.fromkeys(queries_to_run))

        print(f"[Retrieve] Running {len(queries_to_run)} queries")

        if analysis.is_comparison and len(analysis.companies) >= 2:
            # Comparison mode: search per company untuk hasil seimbang
            print(f"[Retrieve] Comparison mode: {len(analysis.companies)} companies")

            for company in analysis.companies:
                company_results = []
                for query in queries_to_run:
                    results = self.search(query, company_filter=company, top_k=top_k_per_company)
                    for r in results:
                        if r.id not in seen_ids:
                            seen_ids.add(r.id)
                            company_results.append(r)

                print(f"  → {company}: {len(company_results)} unique results")
                all_results.extend(company_results)

            # Sort by score
            all_results.sort(key=lambda x: x.score, reverse=True)

        elif analysis.companies:
            # Single company mode - multiple queries
            company = analysis.companies[0]
            print(f"[Retrieve] Single company: {company}")

            for query in queries_to_run:
                results = self.search(query, company_filter=company, top_k=top_k_per_company)
                add_results(results)
                print(f"  → Query '{query[:40]}...': {len(results)} results")

            print(f"  → Total unique: {len(all_results)} results")

        else:
            # No company filter - general search
            print(f"[Retrieve] General search (no company filter)")

            for query in queries_to_run:
                results = self.search(query, top_k=top_k_per_company)
                add_results(results)

            print(f"  → Total unique: {len(all_results)} results")

        return all_results

    def rerank_results(self, results: List[SearchResult], query: str, analysis: QueryAnalysis, top_k: int = 20) -> List[SearchResult]:
        """Rerank results dengan prioritas pada data numerik dan relevansi topik."""
        if not results:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())

        def relevance_score(result: SearchResult) -> float:
            content_lower = result.content.lower()
            score = result.score

            # 1. Keyword matches dari query (+0.01 per match)
            keyword_matches = sum(1 for word in query_words if word in content_lower)
            score += keyword_matches * 0.01

            # 2. Bonus untuk konten dengan data numerik/tabel (+0.05)
            numeric_indicators = ["ton co2", "kwh", "mwh", "gigajoule", "terajoule", "liter",
                                  "m3", "rupiah", "miliar", "juta", "karyawan", "employee"]
            has_numeric = any(ind in content_lower for ind in numeric_indicators)
            if has_numeric:
                score += 0.05

            # 3. Bonus untuk section yang relevan (+0.03)
            for topic in analysis.topics:
                if topic in ESG_TOPICS:
                    related_sections = ESG_TOPICS[topic].get("related_sections", [])
                    if any(sec.lower() in result.section.lower() for sec in related_sections):
                        score += 0.03
                        break

            # 4. Bonus untuk konten yang mengandung "tabel" atau format data (+0.02)
            if "tabel" in content_lower or "table" in content_lower:
                score += 0.02

            # 5. Bonus untuk topic keywords match (+0.02 per topic)
            for topic in analysis.topics:
                if topic in ESG_TOPICS:
                    topic_keywords = ESG_TOPICS[topic].get("keywords", [])
                    topic_match = sum(1 for kw in topic_keywords if kw in content_lower)
                    score += topic_match * 0.005

            return score

        reranked = sorted(results, key=relevance_score, reverse=True)
        return reranked[:top_k]

    def format_context(self, results: List[SearchResult]) -> str:
        """Format search results menjadi context untuk LLM."""
        if not results:
            return "Tidak ada dokumen yang ditemukan."

        context_parts = []
        for i, r in enumerate(results, 1):
            section_info = f"{r.section}"
            if r.subsection:
                section_info += f" > {r.subsection}"

            context_parts.append(
                f"[DOKUMEN {i}]\n"
                f"Perusahaan: {r.company}\n"
                f"Sumber: {r.source}\n"
                f"Halaman: {r.page}\n"
                f"Section: {section_info}\n"
                f"Relevansi: {r.score:.4f}\n"
                f"---\n"
                f"{r.content}\n"
            )

        return "\n" + "="*50 + "\n".join(context_parts)

    def call_llm(self, query: str, context: str, analysis: QueryAnalysis) -> Tuple[str, Dict]:
        """Call LLM dengan context dan analysis."""

        # Build enhanced prompt
        comparison_hint = ""
        if analysis.is_comparison:
            companies_str = " dan ".join(analysis.companies)
            comparison_hint = f"\n\nINSTRUKSI KHUSUS: Ini adalah pertanyaan PERBANDINGAN antara {companies_str}. Sajikan data dalam format TABEL yang membandingkan kedua perusahaan."

        user_message = f"""KONTEKS DOKUMEN:
{context}

PERTANYAAN: {query}
{comparison_hint}

Berikan jawaban yang lengkap, akurat, dan terstruktur berdasarkan konteks di atas."""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MODEL_ARK_API_KEY}"
        }

        payload = {
            "model": MODEL_ARK_LLM_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.3,  # Lower temperature untuk akurasi
            "max_tokens": 3000
        }

        try:
            response = requests.post(
                f"{MODEL_ARK_API_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=90
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

    def save_results(self, query: str, analysis: QueryAnalysis, results: List[SearchResult],
                     response: str, token_usage: Dict):
        """Save hasil ke JSON file."""
        output_data = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "analysis": {
                "companies": analysis.companies,
                "topics": analysis.topics,
                "is_comparison": analysis.is_comparison,
                "expanded_query": analysis.expanded_query
            },
            "config": {
                "embedding_model": EMBEDDING_MODEL,
                "llm_model": MODEL_ARK_LLM_NAME,
                "collection": COLLECTION_NAME,
                "score_threshold": SCORE_THRESHOLD
            },
            "token_usage": {
                "query": token_usage,
                "session": self.token_stats.get_summary()
            },
            "total_results": len(results),
            "response": response,
            "search_results": [
                {
                    "rank": i+1,
                    "score": round(r.score, 4),
                    "company": r.company,
                    "source": r.source,
                    "page": r.page,
                    "section": r.section,
                    "content_preview": r.content[:300] + "..." if len(r.content) > 300 else r.content
                }
                for i, r in enumerate(results)
            ]
        }

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    def chat(self, query: str) -> str:
        """Main chat function."""
        print(f"\n{'='*60}")
        print(f"[Query] {query}")
        print(f"{'='*60}")

        # Step 1: Analyze query
        print("\n[Step 1] Analyzing query...")
        analysis = self.analyze_query(query)
        print(f"  → Companies: {analysis.companies if analysis.companies else 'None detected'}")
        print(f"  → Topics: {analysis.topics if analysis.topics else 'General'}")
        print(f"  → Comparison: {'Yes' if analysis.is_comparison else 'No'}")

        # Step 2: Smart retrieval
        print("\n[Step 2] Retrieving documents...")
        results = self.smart_retrieve(analysis, top_k_per_company=10)

        if not results:
            return "Maaf, tidak ditemukan dokumen yang relevan dengan pertanyaan Anda."

        # Step 3: Rerank
        print("\n[Step 3] Reranking results...")
        reranked = self.rerank_results(results, query, analysis, top_k=20)
        print(f"  → Top {len(reranked)} results selected")

        # Step 4: Format context
        context = self.format_context(reranked)
        print(f"\n[Step 4] Context size: {len(context):,} characters")

        # Step 5: Generate response
        print("\n[Step 5] Generating response...")
        response, token_usage = self.call_llm(query, context, analysis)

        # Update token statistics
        self.token_stats.add_usage(
            token_usage['prompt_tokens'],
            token_usage['completion_tokens']
        )

        # Print token usage untuk query ini
        print(f"\n[Token Usage - Query ini]")
        print(f"  → Input tokens  : {token_usage['prompt_tokens']:,}")
        print(f"  → Output tokens : {token_usage['completion_tokens']:,}")
        print(f"  → Total tokens  : {token_usage['total_tokens']:,}")

        # Print session statistics
        stats = self.token_stats.get_summary()
        print(f"\n[Token Usage - Session Total]")
        print(f"  → Total queries      : {stats['total_queries']}")
        print(f"  → Total input tokens : {stats['total_prompt_tokens']:,}")
        print(f"  → Total output tokens: {stats['total_completion_tokens']:,}")
        print(f"  → Total tokens       : {stats['total_tokens']:,}")
        print(f"  → Avg tokens/query   : {stats['avg_tokens_per_query']:,}")

        # Step 6: Save results
        self.save_results(query, analysis, reranked, response, token_usage)
        print(f"\n[Output] Saved to: {OUTPUT_FILE}")

        return response

    def get_token_stats(self) -> Dict:
        """Get current token statistics."""
        return self.token_stats.get_summary()

    def reset_token_stats(self):
        """Reset token statistics."""
        self.token_stats = TokenStats()
        print("[Info] Token statistics telah di-reset.")


def print_final_stats(chatbot: ESGChatbotCanggih):
    """Print final session statistics."""
    stats = chatbot.get_token_stats()
    if stats['total_queries'] > 0:
        print("\n" + "="*70)
        print("  SESSION TOKEN USAGE SUMMARY")
        print("="*70)
        print(f"  Total queries        : {stats['total_queries']}")
        print(f"  Total input tokens   : {stats['total_prompt_tokens']:,}")
        print(f"  Total output tokens  : {stats['total_completion_tokens']:,}")
        print(f"  Total tokens         : {stats['total_tokens']:,}")
        print(f"  Average tokens/query : {stats['avg_tokens_per_query']:,}")
        print("="*70)


def main():
    chatbot = ESGChatbotCanggih()

    print("\n" + "="*70)
    print("  Selamat datang di ESG Chatbot Canggih!")
    print("  ")
    print("  Contoh pertanyaan:")
    print("    - Berapa emisi GRK Bank Jatim tahun 2024?")
    print("    - Bandingkan emisi Bank Jago vs Bank Jatim")
    print("    - Apa strategi keberlanjutan Bank Jago?")
    print("    - Berapa jumlah karyawan Bank Jatim?")
    print("  ")
    print("  Commands:")
    print("    - 'quit' atau 'exit' untuk keluar")
    print("    - 'stats' untuk melihat token usage")
    print("    - 'reset' untuk reset token statistics")
    print("="*70)

    while True:
        try:
            query = input("\n🧑 Anda: ").strip()

            if not query:
                print("   ⚠️  Pertanyaan tidak boleh kosong!")
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print_final_stats(chatbot)
                print("\n   👋 Terima kasih telah menggunakan ESG Chatbot Canggih!")
                break

            if query.lower() == 'stats':
                stats = chatbot.get_token_stats()
                print("\n[Token Statistics]")
                print(f"  Total queries        : {stats['total_queries']}")
                print(f"  Total input tokens   : {stats['total_prompt_tokens']:,}")
                print(f"  Total output tokens  : {stats['total_completion_tokens']:,}")
                print(f"  Total tokens         : {stats['total_tokens']:,}")
                print(f"  Average tokens/query : {stats['avg_tokens_per_query']:,}")
                continue

            if query.lower() == 'reset':
                chatbot.reset_token_stats()
                continue

            response = chatbot.chat(query)

            print("\n" + "="*70)
            print("🤖 ESG Assistant:")
            print("="*70)
            print(response)
            print("="*70)

        except KeyboardInterrupt:
            print_final_stats(chatbot)
            print("\n\n   👋 Terima kasih! Sampai jumpa!")
            break


if __name__ == "__main__":
    main()
