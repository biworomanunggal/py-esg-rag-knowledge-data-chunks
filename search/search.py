import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText, MatchValue
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "esg_chatbot")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.575))

SEARCH_DIR = Path(__file__).parent
TOP_K = 10

KEYWORD_SYNONYMS = {
    "emisi": ["emisi", "emission", "grk", "ghg", "karbon", "carbon", "co2", "scope 1", "scope 2", "scope 3", "cakupan 1", "cakupan 2", "cakupan 3", "tco2", "kgco2", "gas rumah kaca"],
    "karyawan": ["karyawan", "employee", "pegawai", "sdm", "human capital", "workforce", "laki-laki", "perempuan", "male", "female", "tenaga kerja"],
    "energi": ["energi", "energy", "listrik", "electricity", "kwh", "mwh", "konsumsi energi"],
    "air": ["air", "water", "pdam", "konsumsi air"],
    "limbah": ["limbah", "waste", "sampah", "b3"],
    "lingkungan": ["lingkungan", "environment", "environmental", "biaya lingkungan", "pengelolaan lingkungan"],
    "biaya": ["biaya", "cost", "anggaran", "budget", "alokasi", "allocation", "investasi", "investment", "pengeluaran", "expenditure"],
}

COMPANY_ALIASES = {
    "bank jago": "PT Bank Jago Tbk",
    "jago": "PT Bank Jago Tbk",
    "bank jatim": "PT Bank Pembangunan Daerah Jawa Timur Tbk",
    "jatim": "PT Bank Pembangunan Daerah Jawa Timur Tbk",
}


class MergedResult:
    """Wrapper untuk hasil merge dengan score."""
    def __init__(self, id, score, payload):
        self.id = id
        self.score = score
        self.payload = payload


class ESGSearch:
    """ESG Document Search dengan Hybrid Search."""

    def __init__(self):
        print("=" * 60)
        print("ESG Document Search - Hybrid Search")
        print("=" * 60)
        print("Initializing...")

        print(f"  Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        print(f"  Connecting to Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        print(f"  Collection: {COLLECTION_NAME}")
        print("  Ready!\n")

    def _extract_keywords(self, query: str) -> list:
        """Extract keywords dari query dan expand dengan synonyms."""
        query_lower = query.lower()
        keywords = set()

        for _, synonyms in KEYWORD_SYNONYMS.items():
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
        query_with_prefix = f"query: {query}"
        query_embedding = self.embedding_model.encode(query_with_prefix).tolist()

        company_conditions = []
        if company_filter and len(company_filter) > 0:
            for company in company_filter:
                company_conditions.append(
                    FieldCondition(key="nama_perusahaan", match=MatchValue(value=company))
                )

        keyword_conditions = []
        if keywords:
            for keyword in keywords[:5]:
                keyword_conditions.append(
                    FieldCondition(key="content", match=MatchText(text=keyword))
                )

        query_filter = None

        if company_conditions and keyword_conditions:
            query_filter = Filter(
                should=company_conditions,
                must=[Filter(should=keyword_conditions)]
            )
        elif company_conditions:
            query_filter = Filter(should=company_conditions)
        elif keyword_conditions:
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
        Supports multiple companies untuk perbandingan.
        """
        keywords = self._extract_keywords(query)
        companies = self._extract_companies(query)

        print(f"[Search] Query: \"{query}\"")
        print(f"[Search] Keywords: {keywords[:5] if keywords else 'None'}")
        print(f"[Search] Companies filter: {companies if companies else 'None'}")

        merged_results = []

        if companies and len(companies) > 1:
            per_company_top_k = max(top_k, 30)

            print(f"[Search] Multi-company mode: {len(companies)} companies, {per_company_top_k} results each")

            for company in companies:
                results = self.semantic_search_with_keywords(
                    query,
                    company_filter=[company],
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

            merged_results.sort(key=lambda x: x.score, reverse=True)
            print(f"[Search] Total merged: {len(merged_results)} results")

        else:
            results = self.semantic_search_with_keywords(
                query,
                company_filter=companies,
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

            for result in results:
                merged_results.append(MergedResult(
                    id=result.id,
                    score=result.score,
                    payload=result.payload
                ))

        return merged_results[:top_k]

    def format_results(self, results: list) -> list:
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
                "content": payload.get("content", "")
            })
        return formatted

    def save_results(self, query: str, results: list):
        """Save search results to JSON file."""
        formatted_results = self.format_results(results)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"search_result_{timestamp}.json"
        filepath = SEARCH_DIR / filename

        output = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "embedding_model": EMBEDDING_MODEL,
            "collection": COLLECTION_NAME,
            "total_results": len(formatted_results),
            "score_threshold": SCORE_THRESHOLD,
            "results": formatted_results
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"[Output] Results saved to: {filepath}")
        return filepath


def main():
    searcher = ESGSearch()

    print("=" * 60)
    print("Ketik 'quit' atau 'exit' untuk keluar")
    print("=" * 60)

    while True:
        try:
            query = input("\nMasukkan query: ").strip()

            if not query:
                print("Query tidak boleh kosong!")
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("Bye!")
                break

            results = searcher.hybrid_search(query, top_k=TOP_K)

            print(f"\n[Result] Found {len(results)} results\n")

            for i, result in enumerate(results, 1):
                payload = result.payload
                metadata = payload.get("metadata", {})
                print(f"--- Result {i} (score: {result.score:.4f}) ---")
                print(f"Perusahaan: {payload.get('nama_perusahaan', 'N/A')}")
                print(f"Sumber: {payload.get('sumber_file', 'N/A')}")
                print(f"Halaman: {metadata.get('page_range', 'N/A')}")
                print(f"Content: {payload.get('content', '')[:200]}...")
                print()

            searcher.save_results(query, results)
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\nBye!")
            break


if __name__ == "__main__":
    main()
