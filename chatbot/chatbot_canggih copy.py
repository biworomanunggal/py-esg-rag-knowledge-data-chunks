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

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "esg_reports")
COLLECTION_DATA_NAME = os.getenv("QDRANT_COLLECTION_DATA_NAME", "esg_data_reports")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.3))
RERANK_TOP_K = 10
RERANK_DATA_TOP_K = 5  # Top K untuk data collection (Excel)

MODEL_ARK_API_KEY = os.getenv("MODEL_ARK_API_KEY")
MODEL_ARK_API_URL = os.getenv("MODEL_ARK_API_URL")
MODEL_ARK_LLM_NAME = os.getenv("MODEL_ARK_LLM_NAME")

CHATBOT_DIR = Path(__file__).parent
OUTPUT_FILE = CHATBOT_DIR / "chat_canggih_results.json"

# Company mapping dengan variasi nama - organized by sector
COMPANY_MAP = {
    # ============== FINANCE SECTOR ==============
    # Bank Jago
    "bank jago": "Bank Jago",
    "jago": "Bank Jago",
    "pt bank jago": "Bank Jago",
    # Bank Jatim
    "bank jatim": "PT Bank Pembangunan Daerah Jawa Timur Tbk",
    "jatim": "PT Bank Pembangunan Daerah Jawa Timur Tbk",
    "bpd jatim": "PT Bank Pembangunan Daerah Jawa Timur Tbk",
    "bank pembangunan daerah jawa timur": "PT Bank Pembangunan Daerah Jawa Timur Tbk",
    # Bank OCBC NISP
    "ocbc": "PT Bank OCBC NISP Tbk",
    "ocbc nisp": "PT Bank OCBC NISP Tbk",
    "bank ocbc": "PT Bank OCBC NISP Tbk",
    "bank ocbc nisp": "PT Bank OCBC NISP Tbk",
    "nisp": "PT Bank OCBC NISP Tbk",
    "pt bank ocbc nisp": "PT Bank OCBC NISP Tbk",
    # Bank Amar Indonesia
    "amar": "PT. Bank Amar Indonesia Tbk",
    "bank amar": "PT. Bank Amar Indonesia Tbk",
    "amar bank": "PT. Bank Amar Indonesia Tbk",
    "bank amar indonesia": "PT. Bank Amar Indonesia Tbk",
    "pt bank amar": "PT. Bank Amar Indonesia Tbk",

    # ============== MINING SECTOR ==============
    # Adaro Energy
    "adaro": "PT Adaro Energy Indonesia Tbk",
    "adaro energy": "PT Adaro Energy Indonesia Tbk",
    "pt adaro": "PT Adaro Energy Indonesia Tbk",
    "aadi": "PT Adaro Energy Indonesia Tbk",
    # Archi Indonesia
    "archi": "PT Archi Indonesia Tbk",
    "archi indonesia": "PT Archi Indonesia Tbk",
    "pt archi": "PT Archi Indonesia Tbk",
    # Astrindo Nusantara Infrastruktur
    "astrindo": "PT Astrindo Nusantara Infrastruktur Tbk",
    "astrindo nusantara": "PT Astrindo Nusantara Infrastruktur Tbk",
    "pt astrindo": "PT Astrindo Nusantara Infrastruktur Tbk",

    # ============== ENERGY SECTOR ==============
    # Apexindo Pratama Duta
    "apexindo": "Apexindo Pratama Duta Tbk",
    "apex": "Apexindo Pratama Duta Tbk",
    "apexindo pratama": "Apexindo Pratama Duta Tbk",
    # Elnusa
    "elnusa": "PT Elnusa Tbk.",
    "pt elnusa": "PT Elnusa Tbk.",
    "elsa": "PT Elnusa Tbk.",

    # ============== PLANTATION SECTOR ==============
    # Andira Agro
    "andira": "PT Andira Agro Tbk",
    "andira agro": "PT Andira Agro Tbk",
    "pt andira": "PT Andira Agro Tbk",
    # Astra Agro Lestari
    "astra agro": "PT Astra Agro Lestari Tbk",
    "astra agro lestari": "PT Astra Agro Lestari Tbk",
    "aal": "PT Astra Agro Lestari Tbk",
    "pt astra agro": "PT Astra Agro Lestari Tbk",
    # Austindo Nusantara Jaya
    "austindo": "PT Austindo Nusantara Jaya Tbk",
    "anj": "PT Austindo Nusantara Jaya Tbk",
    "austindo nusantara": "PT Austindo Nusantara Jaya Tbk",

    # ============== TRANSPORTATION SECTOR ==============
    # Adi Sarana Armada
    "assa": "PT. Adi Sarana Armada Tbk",
    "adi sarana": "PT. Adi Sarana Armada Tbk",
    "adi sarana armada": "PT. Adi Sarana Armada Tbk",
    # AirAsia Indonesia
    "airasia": "PT. AirAsia Indonesia Tbk",
    "airasia indonesia": "PT. AirAsia Indonesia Tbk",
    "air asia": "PT. AirAsia Indonesia Tbk",
    # Armada Berjaya Trans
    "armada berjaya": "PT. Armada Berjaya Trans Tbk",
    "armada berjaya trans": "PT. Armada Berjaya Trans Tbk",
    # Batavia Prosperindo Trans
    "batavia": "PT. Batavia Prosperindo Trans Tbk",
    "batavia prosperindo": "PT. Batavia Prosperindo Trans Tbk",

    # ============== F&B SECTOR ==============
    # Delta Djakarta
    "delta": "PT Delta Djakarta",
    "delta djakarta": "PT Delta Djakarta",
    "pt delta": "PT Delta Djakarta",
    # Akasha Wira International
    "akasha": "PT. Akasha Wira International Tbk",
    "akasha wira": "PT. Akasha Wira International Tbk",
    "ades": "PT. Akasha Wira International Tbk",
    # Aman Agrindo
    "aman agrindo": "PT. Aman Agrindo Tbk",
    "pt aman agrindo": "PT. Aman Agrindo Tbk",
    # Budi Starch & Sweetener
    "budi starch": "PT. Budi Starch & Sweetener Tbk",
    "budi": "PT. Budi Starch & Sweetener Tbk",
    "budi sweetener": "PT. Budi Starch & Sweetener Tbk",

    # ============== CHEMICALS SECTOR ==============
    # Avia Avian
    "avian": "PT. Avia Avian Tbk",
    "avia avian": "PT. Avia Avian Tbk",
    "avia": "PT. Avia Avian Tbk",
    # Barito Pacific
    "barito": "PT. Barito Pacific Tbk",
    "barito pacific": "PT. Barito Pacific Tbk",
    "brpt": "PT. Barito Pacific Tbk",
    # Chandra Asri Pacific
    "chandra asri": "PT. Chandra Asri Pacific Tbk",
    "chandra asri pacific": "PT. Chandra Asri Pacific Tbk",
    "tpia": "PT. Chandra Asri Pacific Tbk",
    # Chemstar Indonesia
    "chemstar": "PT. Chemstar Indonesia Tbk",
    "chemstar indonesia": "PT. Chemstar Indonesia Tbk",

    # ============== HOSPITALITY SECTOR ==============
    # Mandarine Oriental (note: typo in folder name "Hostpitality")
    "mandarine": "Mandarine Oriental",
    "mandarine oriental": "Mandarine Oriental",
    "moil": "Mandarine Oriental",
    # MNC Land
    "mnc land": "MNC Land",
    "mnc": "MNC Land",
    # Andalan Perkasa Abadi
    "andalan perkasa": "PT Andalan Perkasa Abadi Tbk",
    "andalan perkasa abadi": "PT Andalan Perkasa Abadi Tbk",
    # Arthavest
    "arthavest": "PT Arthavest Tbk",
    "pt arthavest": "PT Arthavest Tbk",
}

# Mapping company to sector untuk filter
COMPANY_SECTOR_MAP = {
    # Finance
    "Bank Jago": "Finance",
    "PT Bank Pembangunan Daerah Jawa Timur Tbk": "Finance",
    "PT Bank OCBC NISP Tbk": "Finance",
    "PT. Bank Amar Indonesia Tbk": "Finance",
    # Mining
    "PT Adaro Energy Indonesia Tbk": "Mining",
    "PT Archi Indonesia Tbk": "Mining",
    "PT Astrindo Nusantara Infrastruktur Tbk": "Mining",
    # Energy
    "Apexindo Pratama Duta Tbk": "Energy",
    "PT Elnusa Tbk.": "Energy",
    # Plantation
    "PT Andira Agro Tbk": "Plantation",
    "PT Astra Agro Lestari Tbk": "Plantation",
    "PT Austindo Nusantara Jaya Tbk": "Plantation",
    # Transportation
    "PT. Adi Sarana Armada Tbk": "Transportation",
    "PT. AirAsia Indonesia Tbk": "Transportation",
    "PT. Armada Berjaya Trans Tbk": "Transportation",
    "PT. Batavia Prosperindo Trans Tbk": "Transportation",
    # F&B
    "PT Delta Djakarta": "F&B",
    "PT. Akasha Wira International Tbk": "F&B",
    "PT. Aman Agrindo Tbk": "F&B",
    "PT. Budi Starch & Sweetener Tbk": "F&B",
    # Chemicals
    "PT. Avia Avian Tbk": "Chemicals",
    "PT. Barito Pacific Tbk": "Chemicals",
    "PT. Chandra Asri Pacific Tbk": "Chemicals",
    "PT. Chemstar Indonesia Tbk": "Chemicals",
    # Hospitality
    "Mandarine Oriental": "Hostpitality",
    "MNC Land": "Hostpitality",
    "PT Andalan Perkasa Abadi Tbk": "Hostpitality",
    "PT Arthavest Tbk": "Hostpitality",
}

# ESG Topic keywords untuk query expansion - diperluas dengan variasi dalam dokumen
ESG_TOPICS = {
    "emisi": {
        "keywords": ["emisi", "grk", "ghg", "karbon", "carbon", "co2", "scope 1", "scope 2", "scope 3",
                    "cakupan 1", "cakupan 2", "cakupan 3", "gas rumah kaca", "greenhouse", "ton co2",
                    "intensitas emisi", "emission", "pengurangan emisi", "net zero", "carbon footprint",
                    "jejak karbon", "nol emisi", "dekarbonisasi", "ton coeq", "tco2e", "metana", "ch4", "n2o"],
        "search_queries": ["emisi GRK scope", "total emisi cakupan", "jumlah emisi GRK", "intensitas emisi"],
        "related_sections": ["Kinerja Keberlanjutan", "Keberlanjutan", "Aspek Emisi", "Kinerja Lingkungan"]
    },
    "energi": {
        "keywords": ["energi", "energy", "listrik", "electricity", "kwh", "mwh", "gigajoule", "gj",
                    "konsumsi energi", "bbm", "solar", "terajoule", "konsumsi listrik", "energi terbarukan",
                    "renewable", "panel surya", "solar panel", "efisiensi energi", "intensitas energi"],
        "search_queries": ["konsumsi energi", "penggunaan listrik", "total energi"],
        "related_sections": ["Kinerja Keberlanjutan", "Keberlanjutan", "Kinerja Lingkungan"]
    },
    "air": {
        "keywords": ["air", "water", "pdam", "konsumsi air", "m3", "penggunaan air", "air bersih",
                    "air tanah", "daur ulang air", "water recycling", "intensitas air", "efisiensi air"],
        "search_queries": ["konsumsi air", "penggunaan air", "pengelolaan air"],
        "related_sections": ["Kinerja Keberlanjutan", "Keberlanjutan", "Kinerja Lingkungan"]
    },
    "limbah": {
        "keywords": ["limbah", "waste", "sampah", "b3", "efluen", "pengelolaan limbah", "hazardous waste",
                    "limbah berbahaya", "daur ulang", "recycling", "reduce", "reuse", "circular economy",
                    "3r", "pengolahan limbah", "tpa", "landfill"],
        "search_queries": ["pengelolaan limbah", "limbah b3", "total limbah"],
        "related_sections": ["Kinerja Keberlanjutan", "Keberlanjutan", "Kinerja Lingkungan"]
    },
    "karyawan": {
        "keywords": ["karyawan", "employee", "pegawai", "sdm", "human capital", "tenaga kerja",
                    "pekerja", "jumlah karyawan", "total karyawan", "pelatihan", "training",
                    "turnover", "rekrutmen", "recruitment", "keberagaman", "diversity", "inklusi",
                    "gender", "perempuan", "pria", "jam pelatihan", "training hours"],
        "search_queries": ["jumlah karyawan", "total pegawai", "komposisi karyawan", "demografi karyawan"],
        "related_sections": ["Sumber Daya Manusia", "Kinerja Keberlanjutan", "Profil Perusahaan", "Kinerja Sosial"]
    },
    "k3": {
        "keywords": ["k3", "keselamatan", "kesehatan kerja", "ohs", "occupational health", "safety",
                    "kecelakaan kerja", "fatality", "ltir", "trir", "lost time injury", "zero accident",
                    "nihil kecelakaan", "cidera", "injury", "near miss"],
        "search_queries": ["keselamatan kerja", "kecelakaan kerja", "kinerja k3"],
        "related_sections": ["Kinerja K3", "Keselamatan Kerja", "HSE"]
    },
    "tata_kelola": {
        "keywords": ["tata kelola", "governance", "direksi", "komisaris", "komite", "audit", "gcg",
                    "corporate governance", "transparansi", "akuntabilitas", "responsibility",
                    "independensi", "fairness", "anti korupsi", "etika bisnis"],
        "search_queries": ["tata kelola perusahaan", "good corporate governance", "struktur governance"],
        "related_sections": ["Tata Kelola", "Audit Internal", "Komite Audit", "GCG"]
    },
    "keuangan": {
        "keywords": ["keuangan", "financial", "pendapatan", "laba", "aset", "modal", "rupiah",
                    "revenue", "profit", "ekuitas", "equity", "triliun", "miliar", "billion",
                    "pertumbuhan", "growth", "dividen"],
        "search_queries": ["kinerja keuangan", "pendapatan", "laba bersih"],
        "related_sections": ["Tinjauan Keuangan", "Ikhtisar Keuangan", "Laporan Keuangan", "Kinerja Ekonomi"]
    },
    "sosial_masyarakat": {
        "keywords": ["csr", "tanggung jawab sosial", "community", "masyarakat", "pemberdayaan",
                    "social investment", "investasi sosial", "donasi", "bantuan", "penerima manfaat",
                    "beneficiaries", "pengembangan komunitas", "community development"],
        "search_queries": ["tanggung jawab sosial", "pengembangan masyarakat", "csr"],
        "related_sections": ["Kinerja Sosial", "CSR", "Community Development", "Tanggung Jawab Sosial"]
    },
    "biodiversitas": {
        "keywords": ["biodiversitas", "biodiversity", "keanekaragaman hayati", "flora", "fauna",
                    "ekosistem", "ecosystem", "konservasi", "conservation", "hutan", "forest",
                    "spesies", "species", "habitat", "rewilding"],
        "search_queries": ["keanekaragaman hayati", "konservasi", "biodiversitas"],
        "related_sections": ["Kinerja Lingkungan", "Biodiversitas", "Konservasi"]
    }
}

# Sector-specific ESG topics dengan keywords dan metrics khusus per industri
SECTOR_SPECIFIC_TOPICS = {
    "Finance": {
        "keywords": ["pembiayaan hijau", "green financing", "sustainable finance", "kredit hijau",
                    "green bond", "obligasi hijau", "portofolio berkelanjutan", "sustainable portfolio",
                    "kredit umkm", "inklusi keuangan", "financial inclusion", "literasi keuangan",
                    "digital banking", "fintech", "npl", "car", "bopo", "nim"],
        "metrics": ["pembiayaan berkelanjutan", "portofolio hijau", "kredit usaha rakyat"],
        "esg_focus": ["sustainable lending", "green portfolio", "financial inclusion"]
    },
    "Mining": {
        "keywords": ["reklamasi", "revegetasi", "tambang", "mine", "batubara", "coal", "mineral",
                    "overburden", "stripping ratio", "produksi tambang", "cadangan", "reserves",
                    "debu", "dust", "air asam tambang", "acid mine drainage", "rehabilitasi lahan",
                    "post mining", "pasca tambang", "izin pinjam pakai kawasan hutan", "ippkh"],
        "metrics": ["luas reklamasi", "produksi batubara", "cadangan mineral", "revegetasi"],
        "esg_focus": ["land rehabilitation", "mine closure", "community resettlement"]
    },
    "Energy": {
        "keywords": ["migas", "oil and gas", "minyak bumi", "gas alam", "pengeboran", "drilling",
                    "eksplorasi", "exploration", "produksi minyak", "oil production", "barrel",
                    "bph", "mmscfd", "energi terbarukan", "renewable energy", "ebt", "geothermal",
                    "panas bumi", "plts", "energi surya", "transisi energi", "energy transition"],
        "metrics": ["produksi migas", "lifting minyak", "kapasitas energi terbarukan"],
        "esg_focus": ["energy transition", "flaring reduction", "methane reduction"]
    },
    "Plantation": {
        "keywords": ["sawit", "palm oil", "cpo", "crude palm oil", "kebun", "plantation", "tbs",
                    "fresh fruit bunch", "rendemen", "rspo", "ispo", "sustainable palm oil",
                    "ndpe", "no deforestation", "hcv", "high conservation value", "hcs",
                    "high carbon stock", "smallholder", "petani plasma", "petani swadaya",
                    "lahan gambut", "peatland", "kebakaran lahan", "land fire", "replanting"],
        "metrics": ["produksi cpo", "luas kebun", "produktivitas tbs", "sertifikasi rspo"],
        "esg_focus": ["sustainable palm oil", "no deforestation", "smallholder inclusion"]
    },
    "Transportation": {
        "keywords": ["armada", "fleet", "kendaraan", "vehicle", "logistik", "logistics",
                    "pengiriman", "delivery", "bbm", "fuel", "efisiensi bahan bakar", "fuel efficiency",
                    "kendaraan listrik", "electric vehicle", "ev", "emisi transportasi", "transport emission",
                    "aviation", "penerbangan", "pesawat", "aircraft", "rute", "route"],
        "metrics": ["jumlah armada", "efisiensi bbm", "armada listrik", "emisi per km"],
        "esg_focus": ["fleet electrification", "fuel efficiency", "green logistics"]
    },
    "F&B": {
        "keywords": ["makanan", "food", "minuman", "beverage", "kemasan", "packaging", "plastik",
                    "plastic", "daur ulang kemasan", "packaging recycling", "food safety",
                    "keamanan pangan", "halal", "nutrisi", "nutrition", "gula", "sugar",
                    "bahan baku", "raw material", "pertanian", "agriculture", "pasokan berkelanjutan",
                    "sustainable sourcing", "food waste", "sampah makanan"],
        "metrics": ["produksi makanan", "kemasan daur ulang", "sertifikasi halal"],
        "esg_focus": ["sustainable packaging", "food safety", "responsible sourcing"]
    },
    "Chemicals": {
        "keywords": ["petrokimia", "petrochemical", "olefin", "polyethylene", "polypropylene",
                    "polymer", "resin", "cat", "paint", "coating", "bahan kimia", "chemical",
                    "emisi proses", "process emission", "gas buang", "flue gas", "efluen",
                    "effluent", "spill", "tumpahan", "bahan berbahaya", "hazardous material",
                    "product stewardship", "responsible care"],
        "metrics": ["produksi petrokimia", "emisi industri", "limbah kimia"],
        "esg_focus": ["process safety", "chemical management", "circular economy"]
    },
    "Hostpitality": {
        "keywords": ["hotel", "resort", "hospitality", "properti", "property", "kamar", "room",
                    "okupansi", "occupancy", "tamu", "guest", "pariwisata", "tourism",
                    "green hotel", "hotel hijau", "single use plastic", "plastik sekali pakai",
                    "food waste hotel", "limbah hotel", "linen", "laundry", "air minum kemasan",
                    "amenities", "energy hotel"],
        "metrics": ["tingkat okupansi", "konsumsi energi per kamar", "pengurangan plastik"],
        "esg_focus": ["green building", "waste reduction", "sustainable tourism"]
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
    source_type: str = "pdf"  # "pdf" or "data" (Excel)


@dataclass
class QueryAnalysis:
    """Hasil analisis query."""
    original_query: str
    companies: List[str]
    topics: List[str]
    is_comparison: bool
    expanded_query: str
    sectors: List[str] = None  # Detected sectors

    def __post_init__(self):
        if self.sectors is None:
            self.sectors = []


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

        print(f"  → Collection (PDF): {COLLECTION_NAME}")
        print(f"  → Collection (Data): {COLLECTION_DATA_NAME}")
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

        # Detect sectors from detected companies
        sectors = []
        for company in companies:
            if company in COMPANY_SECTOR_MAP:
                sector = COMPANY_SECTOR_MAP[company]
                if sector not in sectors:
                    sectors.append(sector)

        # Also detect sectors from query keywords
        sector_keywords = {
            "Finance": ["bank", "keuangan", "kredit", "financial", "financing"],
            "Mining": ["tambang", "batubara", "coal", "mining", "mineral"],
            "Energy": ["migas", "energi", "oil", "gas", "pengeboran", "drilling"],
            "Plantation": ["sawit", "palm", "perkebunan", "plantation", "cpo"],
            "Transportation": ["transportasi", "logistik", "armada", "fleet", "kendaraan"],
            "F&B": ["makanan", "minuman", "food", "beverage", "pangan"],
            "Chemicals": ["kimia", "chemical", "petrokimia", "petrochemical"],
            "Hostpitality": ["hotel", "hospitality", "pariwisata", "tourism", "properti"]
        }
        for sector, keywords in sector_keywords.items():
            if any(kw in query_lower for kw in keywords):
                if sector not in sectors:
                    sectors.append(sector)

        # Detect comparison intent
        comparison_keywords = ["vs", "versus", "dibanding", "bandingkan", "perbandingan", "compare", "beda", "perbedaan"]
        is_comparison = any(kw in query_lower for kw in comparison_keywords) or len(companies) > 1

        # Detect ESG topics
        detected_topics = []
        for topic, info in ESG_TOPICS.items():
            if any(kw in query_lower for kw in info["keywords"]):
                detected_topics.append(topic)

        # Also check sector-specific topics
        for sector in sectors:
            if sector in SECTOR_SPECIFIC_TOPICS:
                sector_info = SECTOR_SPECIFIC_TOPICS[sector]
                if any(kw in query_lower for kw in sector_info.get("keywords", [])):
                    topic_name = f"sector_{sector.lower()}"
                    if topic_name not in detected_topics:
                        detected_topics.append(topic_name)

        # Expand query dengan keywords terkait
        expanded_parts = [query]

        # Add general ESG topic keywords
        for topic in detected_topics:
            if topic in ESG_TOPICS:
                keywords = ESG_TOPICS[topic]["keywords"][:3]
                expanded_parts.extend(keywords)

        # Add sector-specific keywords
        for sector in sectors:
            if sector in SECTOR_SPECIFIC_TOPICS:
                sector_keywords = SECTOR_SPECIFIC_TOPICS[sector].get("keywords", [])[:3]
                expanded_parts.extend(sector_keywords)

        expanded_query = " ".join(expanded_parts)

        return QueryAnalysis(
            original_query=query,
            companies=companies,
            topics=detected_topics,
            is_comparison=is_comparison,
            expanded_query=expanded_query,
            sectors=sectors
        )

    def search(self, query: str, company_filter: Optional[str] = None,
               sector_filter: Optional[str] = None, top_k: int = 15,
               collection: str = None) -> List[SearchResult]:
        """Semantic search dengan optional company and sector filter.

        Args:
            query: Search query
            company_filter: Filter by company name
            sector_filter: Filter by sector
            top_k: Number of results to return
            collection: Collection to search in (default: COLLECTION_NAME)
        """
        if collection is None:
            collection = COLLECTION_NAME

        # E5 model membutuhkan prefix "query:"
        query_with_prefix = f"query: {query}"
        query_embedding = self.embedding_model.encode(query_with_prefix).tolist()

        # Build filter conditions
        filter_conditions = []
        if company_filter:
            filter_conditions.append(
                FieldCondition(key="nama_perusahaan", match=MatchValue(value=company_filter))
            )
        if sector_filter:
            filter_conditions.append(
                FieldCondition(key="sector", match=MatchValue(value=sector_filter))
            )

        query_filter = None
        if filter_conditions:
            query_filter = Filter(must=filter_conditions)

        # Gunakan score threshold yang lebih rendah untuk coverage lebih baik
        # Reranking akan memfilter hasil yang kurang relevan
        search_threshold = min(SCORE_THRESHOLD, 0.25)

        results = self.qdrant.query_points(
            collection_name=collection,
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

            # Determine source type based on collection
            source_type = "pdf" if collection == COLLECTION_NAME else "data"

            search_results.append(SearchResult(
                id=str(r.id),
                score=r.score,
                content=payload.get("content", ""),
                company=payload.get("nama_perusahaan", ""),
                source=payload.get("sumber_file", ""),
                page=metadata.get("page_range", "N/A"),
                section=metadata.get("section", "N/A"),
                subsection=metadata.get("subsection", ""),
                source_type=source_type
            ))

        return search_results

    def smart_retrieve(self, analysis: QueryAnalysis, top_k_per_company: int = 15) -> Tuple[List[SearchResult], List[SearchResult]]:
        """Smart retrieval dengan multi-query approach untuk coverage lebih baik.

        Returns:
            Tuple of (pdf_results, data_results) - results from both collections
        """
        pdf_results = []
        data_results = []
        seen_pdf_ids = set()
        seen_data_ids = set()

        def add_pdf_results(results: List[SearchResult]):
            """Add PDF results while avoiding duplicates."""
            for r in results:
                if r.id not in seen_pdf_ids:
                    seen_pdf_ids.add(r.id)
                    pdf_results.append(r)

        def add_data_results(results: List[SearchResult]):
            """Add data results while avoiding duplicates."""
            for r in results:
                if r.id not in seen_data_ids:
                    seen_data_ids.add(r.id)
                    data_results.append(r)

        # Buat multiple queries untuk coverage lebih baik
        queries_to_run = [analysis.original_query]

        # Tambahkan search queries dari detected topics
        for topic in analysis.topics:
            if topic in ESG_TOPICS and "search_queries" in ESG_TOPICS[topic]:
                queries_to_run.extend(ESG_TOPICS[topic]["search_queries"][:2])

        # Tambahkan sector-specific search queries
        for sector in analysis.sectors:
            if sector in SECTOR_SPECIFIC_TOPICS:
                sector_metrics = SECTOR_SPECIFIC_TOPICS[sector].get("metrics", [])
                queries_to_run.extend(sector_metrics[:2])

        # Tambahkan expanded query jika berbeda
        if analysis.expanded_query != analysis.original_query:
            queries_to_run.append(analysis.expanded_query)

        # Deduplicate queries
        queries_to_run = list(dict.fromkeys(queries_to_run))

        print(f"[Retrieve] Running {len(queries_to_run)} queries on both collections")
        if analysis.sectors:
            print(f"[Retrieve] Detected sectors: {', '.join(analysis.sectors)}")

        # ============== SEARCH PDF COLLECTION (Primary) ==============
        print(f"\n[Retrieve PDF] Searching {COLLECTION_NAME}...")

        if analysis.is_comparison and len(analysis.companies) >= 2:
            # Comparison mode: search per company untuk hasil seimbang
            print(f"  Mode: Comparison ({len(analysis.companies)} companies)")

            for company in analysis.companies:
                company_results = []
                company_sector = COMPANY_SECTOR_MAP.get(company)

                for query in queries_to_run:
                    results = self.search(query, company_filter=company, top_k=top_k_per_company,
                                         collection=COLLECTION_NAME)
                    for r in results:
                        if r.id not in seen_pdf_ids:
                            seen_pdf_ids.add(r.id)
                            company_results.append(r)

                print(f"  → {company}: {len(company_results)} results")
                pdf_results.extend(company_results)

            pdf_results.sort(key=lambda x: x.score, reverse=True)

        elif analysis.companies:
            # Single company mode
            company = analysis.companies[0]
            company_sector = COMPANY_SECTOR_MAP.get(company, "Unknown")
            print(f"  Mode: Single company - {company} ({company_sector})")

            for query in queries_to_run:
                results = self.search(query, company_filter=company, top_k=top_k_per_company,
                                     collection=COLLECTION_NAME)
                add_pdf_results(results)

            print(f"  → Total: {len(pdf_results)} results")

        elif analysis.sectors:
            # Sector-based search
            print(f"  Mode: Sector-based - {', '.join(analysis.sectors)}")

            for sector in analysis.sectors:
                sector_results = []
                for query in queries_to_run:
                    results = self.search(query, sector_filter=sector, top_k=top_k_per_company,
                                         collection=COLLECTION_NAME)
                    for r in results:
                        if r.id not in seen_pdf_ids:
                            seen_pdf_ids.add(r.id)
                            sector_results.append(r)

                print(f"  → Sector {sector}: {len(sector_results)} results")
                pdf_results.extend(sector_results)

            pdf_results.sort(key=lambda x: x.score, reverse=True)

        else:
            # No filter - general search
            print(f"  Mode: General search")

            for query in queries_to_run:
                results = self.search(query, top_k=top_k_per_company, collection=COLLECTION_NAME)
                add_pdf_results(results)

            print(f"  → Total: {len(pdf_results)} results")

        # ============== SEARCH DATA COLLECTION (Supplement) ==============
        print(f"\n[Retrieve Data] Searching {COLLECTION_DATA_NAME}...")

        # For data collection, use simpler search (usually has less data)
        data_top_k = min(top_k_per_company, 10)

        if analysis.is_comparison and len(analysis.companies) >= 2:
            print(f"  Mode: Comparison ({len(analysis.companies)} companies)")

            for company in analysis.companies:
                company_results = []

                for query in queries_to_run[:3]:  # Limit queries for data collection
                    results = self.search(query, company_filter=company, top_k=data_top_k,
                                         collection=COLLECTION_DATA_NAME)
                    for r in results:
                        if r.id not in seen_data_ids:
                            seen_data_ids.add(r.id)
                            company_results.append(r)

                print(f"  → {company}: {len(company_results)} results")
                data_results.extend(company_results)

            data_results.sort(key=lambda x: x.score, reverse=True)

        elif analysis.companies:
            company = analysis.companies[0]
            print(f"  Mode: Single company - {company}")

            for query in queries_to_run[:3]:
                results = self.search(query, company_filter=company, top_k=data_top_k,
                                     collection=COLLECTION_DATA_NAME)
                add_data_results(results)

            print(f"  → Total: {len(data_results)} results")

        elif analysis.sectors:
            print(f"  Mode: Sector-based - {', '.join(analysis.sectors)}")

            for sector in analysis.sectors:
                sector_results = []
                for query in queries_to_run[:3]:
                    results = self.search(query, sector_filter=sector, top_k=data_top_k,
                                         collection=COLLECTION_DATA_NAME)
                    for r in results:
                        if r.id not in seen_data_ids:
                            seen_data_ids.add(r.id)
                            sector_results.append(r)

                print(f"  → Sector {sector}: {len(sector_results)} results")
                data_results.extend(sector_results)

            data_results.sort(key=lambda x: x.score, reverse=True)

        else:
            print(f"  Mode: General search")

            for query in queries_to_run[:3]:
                results = self.search(query, top_k=data_top_k, collection=COLLECTION_DATA_NAME)
                add_data_results(results)

            print(f"  → Total: {len(data_results)} results")

        print(f"\n[Retrieve Summary] PDF: {len(pdf_results)}, Data: {len(data_results)}")

        return pdf_results, data_results

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
                                  "m3", "rupiah", "miliar", "juta", "karyawan", "employee",
                                  "ton coeq", "tco2e", "triliun", "billion", "gj"]
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

            # 5. Bonus untuk topic keywords match (+0.005 per match)
            for topic in analysis.topics:
                if topic in ESG_TOPICS:
                    topic_keywords = ESG_TOPICS[topic].get("keywords", [])
                    topic_match = sum(1 for kw in topic_keywords if kw in content_lower)
                    score += topic_match * 0.005

            # 6. NEW: Bonus untuk sector-specific keywords match (+0.005 per match)
            for sector in analysis.sectors:
                if sector in SECTOR_SPECIFIC_TOPICS:
                    sector_keywords = SECTOR_SPECIFIC_TOPICS[sector].get("keywords", [])
                    sector_match = sum(1 for kw in sector_keywords if kw in content_lower)
                    score += sector_match * 0.005

                    # Additional bonus for sector-specific metrics
                    sector_metrics = SECTOR_SPECIFIC_TOPICS[sector].get("metrics", [])
                    if any(metric.lower() in content_lower for metric in sector_metrics):
                        score += 0.03

            return score

        reranked = sorted(results, key=relevance_score, reverse=True)
        return reranked[:top_k]

    def format_context(self, pdf_results: List[SearchResult], data_results: List[SearchResult] = None) -> str:
        """Format search results menjadi context untuk LLM.

        Args:
            pdf_results: Results from PDF collection (primary)
            data_results: Results from data collection (supplementary)
        """
        if not pdf_results and not data_results:
            return "Tidak ada dokumen yang ditemukan."

        context_parts = []
        doc_num = 1

        # Format PDF results (primary source)
        if pdf_results:
            context_parts.append("=" * 50)
            context_parts.append("SUMBER UTAMA (Laporan PDF)")
            context_parts.append("=" * 50)

            for r in pdf_results:
                section_info = f"{r.section}"
                if r.subsection:
                    section_info += f" > {r.subsection}"

                context_parts.append(
                    f"\n[DOKUMEN {doc_num}]\n"
                    f"Perusahaan: {r.company}\n"
                    f"Sumber: {r.source}\n"
                    f"Halaman: {r.page}\n"
                    f"Section: {section_info}\n"
                    f"Relevansi: {r.score:.4f}\n"
                    f"---\n"
                    f"{r.content}\n"
                )
                doc_num += 1

        # Format data results (supplementary source)
        if data_results:
            context_parts.append("\n" + "=" * 50)
            context_parts.append("DATA TAMBAHAN (Data Kuantitatif)")
            context_parts.append("=" * 50)

            for r in data_results:
                section_info = f"{r.section}"
                if r.subsection:
                    section_info += f" > {r.subsection}"

                context_parts.append(
                    f"\n[DATA {doc_num}]\n"
                    f"Perusahaan: {r.company}\n"
                    f"Sumber: {r.source}\n"
                    f"Relevansi: {r.score:.4f}\n"
                    f"---\n"
                    f"{r.content}\n"
                )
                doc_num += 1

        return "\n".join(context_parts)

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
        # Separate PDF and data results for logging
        pdf_results = [r for r in results if r.source_type == "pdf"]
        data_results = [r for r in results if r.source_type == "data"]

        output_data = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "analysis": {
                "companies": analysis.companies,
                "topics": analysis.topics,
                "is_comparison": analysis.is_comparison,
                "expanded_query": analysis.expanded_query,
                "sectors": analysis.sectors
            },
            "config": {
                "embedding_model": EMBEDDING_MODEL,
                "llm_model": MODEL_ARK_LLM_NAME,
                "collection_pdf": COLLECTION_NAME,
                "collection_data": COLLECTION_DATA_NAME,
                "score_threshold": SCORE_THRESHOLD,
                "rerank_top_k_pdf": RERANK_TOP_K,
                "rerank_top_k_data": RERANK_DATA_TOP_K
            },
            "token_usage": {
                "query": token_usage,
                "session": self.token_stats.get_summary()
            },
            "results_summary": {
                "total": len(results),
                "pdf_results": len(pdf_results),
                "data_results": len(data_results)
            },
            "response": response,
            "search_results_pdf": [
                {
                    "rank": i+1,
                    "score": round(r.score, 4),
                    "company": r.company,
                    "source": r.source,
                    "page": r.page,
                    "section": r.section,
                    "source_type": r.source_type,
                    "content_preview": r.content[:300] + "..." if len(r.content) > 300 else r.content
                }
                for i, r in enumerate(pdf_results)
            ],
            "search_results_data": [
                {
                    "rank": i+1,
                    "score": round(r.score, 4),
                    "company": r.company,
                    "source": r.source,
                    "section": r.section,
                    "source_type": r.source_type,
                    "content_preview": r.content[:300] + "..." if len(r.content) > 300 else r.content
                }
                for i, r in enumerate(data_results)
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

        # Step 2: Smart retrieval (returns tuple of pdf_results, data_results)
        print("\n[Step 2] Retrieving documents...")
        pdf_results, data_results = self.smart_retrieve(analysis, top_k_per_company=10)

        if not pdf_results and not data_results:
            return "Maaf, tidak ditemukan dokumen yang relevan dengan pertanyaan Anda."

        # Step 3: Rerank both result sets
        print("\n[Step 3] Reranking results...")
        reranked_pdf = self.rerank_results(pdf_results, query, analysis, top_k=RERANK_TOP_K)
        reranked_data = self.rerank_results(data_results, query, analysis, top_k=RERANK_DATA_TOP_K)
        print(f"  → PDF results: {len(reranked_pdf)} selected")
        print(f"  → Data results: {len(reranked_data)} selected")

        # Step 4: Format context with both sources
        context = self.format_context(reranked_pdf, reranked_data)
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

        # Step 6: Save results (combine both result sets for logging)
        all_results = reranked_pdf + reranked_data
        self.save_results(query, analysis, all_results, response, token_usage)
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
