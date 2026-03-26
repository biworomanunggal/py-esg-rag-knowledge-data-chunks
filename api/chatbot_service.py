#!/usr/bin/env python3
"""
ESG RAG Chatbot Service
========================
Adapted from chatbot_canggih.py for API usage.
All logic is exactly the same as the original chatbot.
"""

import os
import re
import json
import uuid
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
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

API_DIR = Path(__file__).parent
HISTORY_DIR = API_DIR / "chat_history"

# Chat history configuration
MAX_HISTORY_MESSAGES = 10  # Maximum messages to keep in context (to avoid token overflow)

# Company mapping dengan variasi nama - organized by sector
# Nama perusahaan harus sama persis dengan yang ada di chunked_data
COMPANY_MAP = {
    # ============== FINANCE SECTOR ==============
    # Bank Jago
    "bank jago": "PT Bank Jago Tbk",
    "jago": "PT Bank Jago Tbk",
    "pt bank jago": "PT Bank Jago Tbk",
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
    "amar": "PT Bank Amar Indonesia Tbk",
    "bank amar": "PT Bank Amar Indonesia Tbk",
    "amar bank": "PT Bank Amar Indonesia Tbk",
    "bank amar indonesia": "PT Bank Amar Indonesia Tbk",
    "pt bank amar": "PT Bank Amar Indonesia Tbk",

    # ============== MINING SECTOR ==============
    # Adaro Andalan Indonesia (dari chunked_data)
    "adaro": "PT Adaro Andalan Indonesia Tbk",
    "adaro energy": "PT Adaro Andalan Indonesia Tbk",
    "adaro andalan": "PT Adaro Andalan Indonesia Tbk",
    "pt adaro": "PT Adaro Andalan Indonesia Tbk",
    "aadi": "PT Adaro Andalan Indonesia Tbk",
    # Archi Indonesia (uppercase di chunked_data)
    "archi": "PT ARCHI INDONESIA Tbk",
    "archi indonesia": "PT ARCHI INDONESIA Tbk",
    "pt archi": "PT ARCHI INDONESIA Tbk",
    # Astrindo Nusantara Infrastruktur
    "astrindo": "PT Astrindo Nusantara Infrastruktur Tbk",
    "astrindo nusantara": "PT Astrindo Nusantara Infrastruktur Tbk",
    "pt astrindo": "PT Astrindo Nusantara Infrastruktur Tbk",

    # ============== ENERGY SECTOR ==============
    # Apexindo Pratama Duta
    "apexindo": "PT Apexindo Pratama Duta Tbk",
    "apex": "PT Apexindo Pratama Duta Tbk",
    "apexindo pratama": "PT Apexindo Pratama Duta Tbk",
    # Elnusa (uppercase di chunked_data)
    "elnusa": "PT ELNUSA Tbk",
    "pt elnusa": "PT ELNUSA Tbk",
    "elsa": "PT ELNUSA Tbk",

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
    "assa": "PT Adi Sarana Armada Tbk",
    "adi sarana": "PT Adi Sarana Armada Tbk",
    "adi sarana armada": "PT Adi Sarana Armada Tbk",
    # AirAsia Indonesia
    "airasia": "PT AirAsia Indonesia Tbk",
    "airasia indonesia": "PT AirAsia Indonesia Tbk",
    "air asia": "PT AirAsia Indonesia Tbk",
    # Armada Berjaya Trans
    "armada berjaya": "PT Armada Berjaya Trans Tbk",
    "armada berjaya trans": "PT Armada Berjaya Trans Tbk",
    # Batavia Prosperindo Trans
    "batavia": "PT Batavia Prosperindo Trans Tbk",
    "batavia prosperindo": "PT Batavia Prosperindo Trans Tbk",

    # ============== F&B SECTOR ==============
    # Delta Djakarta
    "delta": "PT Delta Djakarta Tbk",
    "delta djakarta": "PT Delta Djakarta Tbk",
    "pt delta": "PT Delta Djakarta Tbk",
    # Akasha Wira International
    "akasha": "PT Akasha Wira International Tbk",
    "akasha wira": "PT Akasha Wira International Tbk",
    "ades": "PT Akasha Wira International Tbk",
    # Aman Agrindo
    "aman agrindo": "PT Aman Agrindo Tbk",
    "pt aman agrindo": "PT Aman Agrindo Tbk",
    # Budi Starch & Sweetener
    "budi starch": "PT Budi Starch & Sweetener Tbk",
    "budi": "PT Budi Starch & Sweetener Tbk",
    "budi sweetener": "PT Budi Starch & Sweetener Tbk",

    # ============== CHEMICALS SECTOR ==============
    # Avia Avian
    "avian": "PT Avia Avian Tbk",
    "avia avian": "PT Avia Avian Tbk",
    "avia": "PT Avia Avian Tbk",
    # Barito Pacific
    "barito": "PT Barito Pacific Tbk",
    "barito pacific": "PT Barito Pacific Tbk",
    "brpt": "PT Barito Pacific Tbk",
    # Chandra Asri Pacific
    "chandra asri": "PT Chandra Asri Pacific Tbk",
    "chandra asri pacific": "PT Chandra Asri Pacific Tbk",
    "tpia": "PT Chandra Asri Pacific Tbk",
    # Chemstar Indonesia (perhatikan: ChemStar dengan S besar)
    "chemstar": "PT ChemStar Indonesia Tbk",
    "chemstar indonesia": "PT ChemStar Indonesia Tbk",

    # ============== HOSPITALITY SECTOR ==============
    # Mandarine Oriental
    "mandarine": "Mandarine Oriental",
    "mandarine oriental": "Mandarine Oriental",
    "moil": "Mandarine Oriental",
    # MNC Land
    "mnc land": "PT MNC Land Tbk",
    "mnc": "PT MNC Land Tbk",
    # Andalan Perkasa Abadi
    "andalan perkasa": "PT Andalan Perkasa Abadi Tbk",
    "andalan perkasa abadi": "PT Andalan Perkasa Abadi Tbk",
    # Arthavest (uppercase di chunked_data)
    "arthavest": "PT ARTHAVEST Tbk",
    "pt arthavest": "PT ARTHAVEST Tbk",
}

# Mapping company to sector untuk filter
# Nama perusahaan harus sama persis dengan yang ada di chunked_data
COMPANY_SECTOR_MAP = {
    # Finance
    "PT Bank Jago Tbk": "Finance",
    "PT Bank Pembangunan Daerah Jawa Timur Tbk": "Finance",
    "PT Bank OCBC NISP Tbk": "Finance",
    "PT Bank Amar Indonesia Tbk": "Finance",
    # Mining
    "PT Adaro Andalan Indonesia Tbk": "Mining",
    "PT ARCHI INDONESIA Tbk": "Mining",
    "PT Astrindo Nusantara Infrastruktur Tbk": "Mining",
    # Energy
    "PT Apexindo Pratama Duta Tbk": "Energy",
    "PT ELNUSA Tbk": "Energy",
    # Plantation
    "PT Andira Agro Tbk": "Plantation",
    "PT Astra Agro Lestari Tbk": "Plantation",
    "PT Austindo Nusantara Jaya Tbk": "Plantation",
    # Transportation
    "PT Adi Sarana Armada Tbk": "Transportation",
    "PT AirAsia Indonesia Tbk": "Transportation",
    "PT Armada Berjaya Trans Tbk": "Transportation",
    "PT Batavia Prosperindo Trans Tbk": "Transportation",
    # F&B
    "PT Delta Djakarta Tbk": "F&B",
    "PT Akasha Wira International Tbk": "F&B",
    "PT Aman Agrindo Tbk": "F&B",
    "PT Budi Starch & Sweetener Tbk": "F&B",
    # Chemicals
    "PT Avia Avian Tbk": "Chemicals",
    "PT Barito Pacific Tbk": "Chemicals",
    "PT Chandra Asri Pacific Tbk": "Chemicals",
    "PT ChemStar Indonesia Tbk": "Chemicals",
    # Hospitality
    "Mandarine Oriental": "Hospitality",
    "PT MNC Land Tbk": "Hospitality",
    "PT Andalan Perkasa Abadi Tbk": "Hospitality",
    "PT ARTHAVEST Tbk": "Hospitality",
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
    "Hospitality": {
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
class QueryAnalysisResult:
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


@dataclass
class ChatMessage:
    """Single chat message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ChatSession:
    """Chat session dengan unique ID dan history."""
    session_id: str
    created_at: str
    messages: List[ChatMessage] = field(default_factory=list)

    def add_message(self, role: str, content: str):
        """Add a message to the session."""
        self.messages.append(ChatMessage(role=role, content=content))

    def get_history_for_llm(self, max_messages: int = MAX_HISTORY_MESSAGES) -> List[Dict]:
        """Get recent messages formatted for LLM API."""
        recent_messages = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        return [{"role": msg.role, "content": msg.content} for msg in recent_messages]

    def to_dict(self) -> Dict:
        """Convert session to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "messages": [
                {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp}
                for msg in self.messages
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ChatSession":
        """Create session from dictionary."""
        session = cls(
            session_id=data["session_id"],
            created_at=data["created_at"]
        )
        for msg_data in data.get("messages", []):
            msg = ChatMessage(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=msg_data.get("timestamp", datetime.now().isoformat())
            )
            session.messages.append(msg)
        return session


@dataclass
class ChatResult:
    """Result dari chat function untuk API response."""
    response: str
    session_id: str
    token_usage: Dict
    session_token_usage: Dict
    analysis: QueryAnalysisResult
    pdf_count: int
    data_count: int


class ESGChatbotService:
    """ESG RAG Chatbot Service for API usage."""

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern untuk share instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not ESGChatbotService._initialized:
            self._initialize_components()
            self.sessions: Dict[str, ChatSession] = {}
            self.session_token_stats: Dict[str, TokenStats] = {}
            ESGChatbotService._initialized = True

    def _initialize_components(self):
        """Initialize ML components."""
        print("[Init] Loading ESG Chatbot Service components...")

        print(f"  → Embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        print(f"  → Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        print(f"  → Collection (PDF): {COLLECTION_NAME}")
        print(f"  → Collection (Data): {COLLECTION_DATA_NAME}")
        print(f"  → LLM: {MODEL_ARK_LLM_NAME}")
        print("[Init] Ready!\n")

    # ==================== SESSION MANAGEMENT ====================

    def _get_session_path(self, session_id: str) -> Path:
        """Get the file path for a session."""
        return HISTORY_DIR / f"session_{session_id}.json"

    def _load_session_from_file(self, session_id: str) -> Optional[ChatSession]:
        """Load a session from file."""
        session_path = self._get_session_path(session_id)
        if session_path.exists():
            try:
                with open(session_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return ChatSession.from_dict(data)
            except Exception as e:
                print(f"[Session] Error loading session: {e}")
        return None

    def _save_session_to_file(self, session: ChatSession):
        """Save session to file."""
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        session_path = self._get_session_path(session.session_id)
        try:
            with open(session_path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Session] Error saving session: {e}")

    def get_or_create_session(self, session_id: Optional[str] = None, new_session: bool = False) -> ChatSession:
        """Get existing session or create new one."""
        # If new_session is True, always create new session
        if new_session:
            new_id = session_id or str(uuid.uuid4())[:8]
            session = ChatSession(
                session_id=new_id,
                created_at=datetime.now().isoformat()
            )
            self.sessions[new_id] = session
            self.session_token_stats[new_id] = TokenStats()
            return session

        # If session_id provided, try to load it
        if session_id:
            # Check in-memory cache first
            if session_id in self.sessions:
                return self.sessions[session_id]

            # Try to load from file
            session = self._load_session_from_file(session_id)
            if session:
                self.sessions[session_id] = session
                if session_id not in self.session_token_stats:
                    self.session_token_stats[session_id] = TokenStats()
                return session

        # Create new session
        new_id = session_id or str(uuid.uuid4())[:8]
        session = ChatSession(
            session_id=new_id,
            created_at=datetime.now().isoformat()
        )
        self.sessions[new_id] = session
        self.session_token_stats[new_id] = TokenStats()
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a session by ID."""
        if session_id in self.sessions:
            return self.sessions[session_id]
        return self._load_session_from_file(session_id)

    def list_sessions(self) -> List[Dict]:
        """List all available sessions."""
        sessions = []
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)

        for session_file in HISTORY_DIR.glob("session_*.json"):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data["session_id"],
                    "created_at": data["created_at"],
                    "message_count": len(data.get("messages", []))
                })
            except Exception:
                pass

        return sorted(sessions, key=lambda x: x["created_at"], reverse=True)

    # ==================== COMPANY LIST HELPER ====================

    def _is_company_list_query(self, query: str) -> bool:
        """Check if query is asking about available companies/data."""
        query_lower = query.lower()

        # Keywords yang mengindikasikan pertanyaan tentang daftar perusahaan
        list_keywords = [
            "perusahaan apa saja",
            "daftar perusahaan",
            "list perusahaan",
            "perusahaan yang tersedia",
            "data apa saja",
            "data perusahaan apa",
            "perusahaan mana saja",
            "company apa saja",
            "companies available",
            "available companies",
            "sektor apa saja",
            "sector apa saja",
            "ada perusahaan apa",
            "punya data apa",
            "data yang tersedia",
        ]

        return any(kw in query_lower for kw in list_keywords)

    def _get_company_list_response(self) -> str:
        """Generate formatted response with available companies by sector."""
        # Group companies by sector
        sector_companies = {}
        for company, sector in COMPANY_SECTOR_MAP.items():
            if sector not in sector_companies:
                sector_companies[sector] = []
            sector_companies[sector].append(company)

        # Sort sectors alphabetically
        sorted_sectors = sorted(sector_companies.keys())

        # Build response
        response_parts = [
            "Berikut adalah daftar perusahaan yang datanya tersedia dalam sistem:\n"
        ]

        total_companies = 0
        for sector in sorted_sectors:
            companies = sorted(sector_companies[sector])
            total_companies += len(companies)
            response_parts.append(f"\n**{sector}** ({len(companies)} perusahaan):")
            for company in companies:
                response_parts.append(f"  - {company}")

        response_parts.append(f"\n---\n**Total: {total_companies} perusahaan** dari {len(sorted_sectors)} sektor")
        response_parts.append("\nAnda dapat bertanya tentang data ESG, emisi, keberlanjutan, dan informasi lainnya dari perusahaan-perusahaan di atas.")

        return "\n".join(response_parts)

    def get_companies_by_sector(self) -> Dict[str, List[str]]:
        """Get companies grouped by sector."""
        sector_companies = {}
        for company, sector in COMPANY_SECTOR_MAP.items():
            if sector not in sector_companies:
                sector_companies[sector] = []
            sector_companies[sector].append(company)

        # Sort companies within each sector
        for sector in sector_companies:
            sector_companies[sector] = sorted(sector_companies[sector])

        return sector_companies

    # ==================== QUERY ANALYSIS ====================

    def analyze_query(self, query: str) -> QueryAnalysisResult:
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
            "Hospitality": ["hotel", "hospitality", "pariwisata", "tourism", "properti"]
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

        return QueryAnalysisResult(
            original_query=query,
            companies=companies,
            topics=detected_topics,
            is_comparison=is_comparison,
            expanded_query=expanded_query,
            sectors=sectors
        )

    # ==================== SEARCH ====================

    def search(self, query: str, company_filter: Optional[str] = None,
               sector_filter: Optional[str] = None, top_k: int = 15,
               collection: str = None) -> List[SearchResult]:
        """Semantic search dengan optional company and sector filter."""
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

    def smart_retrieve(self, analysis: QueryAnalysisResult, top_k_per_company: int = 15) -> Tuple[List[SearchResult], List[SearchResult]]:
        """Smart retrieval dengan multi-query approach untuk coverage lebih baik."""
        pdf_results = []
        data_results = []
        seen_pdf_ids = set()
        seen_data_ids = set()

        def add_pdf_results(results: List[SearchResult]):
            for r in results:
                if r.id not in seen_pdf_ids:
                    seen_pdf_ids.add(r.id)
                    pdf_results.append(r)

        def add_data_results(results: List[SearchResult]):
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

        # ============== SEARCH PDF COLLECTION (Primary) ==============
        if analysis.is_comparison and len(analysis.companies) >= 2:
            # Comparison mode: search per company untuk hasil seimbang
            for company in analysis.companies:
                company_results = []
                for query in queries_to_run:
                    results = self.search(query, company_filter=company, top_k=top_k_per_company,
                                         collection=COLLECTION_NAME)
                    for r in results:
                        if r.id not in seen_pdf_ids:
                            seen_pdf_ids.add(r.id)
                            company_results.append(r)
                pdf_results.extend(company_results)
            pdf_results.sort(key=lambda x: x.score, reverse=True)

        elif analysis.companies:
            # Single company mode
            company = analysis.companies[0]
            for query in queries_to_run:
                results = self.search(query, company_filter=company, top_k=top_k_per_company,
                                     collection=COLLECTION_NAME)
                add_pdf_results(results)

        elif analysis.sectors:
            # Sector-based search
            for sector in analysis.sectors:
                sector_results = []
                for query in queries_to_run:
                    results = self.search(query, sector_filter=sector, top_k=top_k_per_company,
                                         collection=COLLECTION_NAME)
                    for r in results:
                        if r.id not in seen_pdf_ids:
                            seen_pdf_ids.add(r.id)
                            sector_results.append(r)
                pdf_results.extend(sector_results)
            pdf_results.sort(key=lambda x: x.score, reverse=True)

        else:
            # No filter - general search
            for query in queries_to_run:
                results = self.search(query, top_k=top_k_per_company, collection=COLLECTION_NAME)
                add_pdf_results(results)

        # ============== SEARCH DATA COLLECTION (Supplement) ==============
        data_top_k = min(top_k_per_company, 10)

        if analysis.is_comparison and len(analysis.companies) >= 2:
            for company in analysis.companies:
                company_results = []
                for query in queries_to_run[:3]:
                    results = self.search(query, company_filter=company, top_k=data_top_k,
                                         collection=COLLECTION_DATA_NAME)
                    for r in results:
                        if r.id not in seen_data_ids:
                            seen_data_ids.add(r.id)
                            company_results.append(r)
                data_results.extend(company_results)
            data_results.sort(key=lambda x: x.score, reverse=True)

        elif analysis.companies:
            company = analysis.companies[0]
            for query in queries_to_run[:3]:
                results = self.search(query, company_filter=company, top_k=data_top_k,
                                     collection=COLLECTION_DATA_NAME)
                add_data_results(results)

        elif analysis.sectors:
            for sector in analysis.sectors:
                sector_results = []
                for query in queries_to_run[:3]:
                    results = self.search(query, sector_filter=sector, top_k=data_top_k,
                                         collection=COLLECTION_DATA_NAME)
                    for r in results:
                        if r.id not in seen_data_ids:
                            seen_data_ids.add(r.id)
                            sector_results.append(r)
                data_results.extend(sector_results)
            data_results.sort(key=lambda x: x.score, reverse=True)

        else:
            for query in queries_to_run[:3]:
                results = self.search(query, top_k=data_top_k, collection=COLLECTION_DATA_NAME)
                add_data_results(results)

        return pdf_results, data_results

    def rerank_results(self, results: List[SearchResult], query: str, analysis: QueryAnalysisResult, top_k: int = 20) -> List[SearchResult]:
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

            # 6. Bonus untuk sector-specific keywords match (+0.005 per match)
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
        """Format search results menjadi context untuk LLM."""
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

    def call_llm(self, query: str, context: str, analysis: QueryAnalysisResult, session: ChatSession) -> Tuple[str, Dict]:
        """Call LLM dengan context, analysis, dan conversation history."""

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

        # Build messages with history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add conversation history (limited to avoid token overflow)
        history = session.get_history_for_llm(MAX_HISTORY_MESSAGES)
        if history:
            messages.extend(history)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": MODEL_ARK_LLM_NAME,
            "messages": messages,
            "temperature": 0.3,
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

    # ==================== MAIN CHAT ====================

    def chat(self, query: str, session_id: Optional[str] = None, new_session: bool = False) -> ChatResult:
        """Main chat function."""
        # Get or create session
        session = self.get_or_create_session(session_id, new_session)
        session_id = session.session_id

        # Ensure token stats exist for this session
        if session_id not in self.session_token_stats:
            self.session_token_stats[session_id] = TokenStats()

        # Check if query is asking about available companies/data
        if self._is_company_list_query(query):
            company_list_response = self._get_company_list_response()
            # Save to history
            session.add_message("user", query)
            session.add_message("assistant", company_list_response)
            self._save_session_to_file(session)

            # Create dummy analysis for company list query
            analysis = QueryAnalysisResult(
                original_query=query,
                companies=[],
                topics=[],
                is_comparison=False,
                expanded_query=query,
                sectors=[]
            )

            return ChatResult(
                response=company_list_response,
                session_id=session_id,
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                session_token_usage=self.session_token_stats[session_id].get_summary(),
                analysis=analysis,
                pdf_count=0,
                data_count=0
            )

        # Step 1: Analyze query
        analysis = self.analyze_query(query)

        # Validasi: Batasi perbandingan maksimal 2 perusahaan
        MAX_COMPARISON_COMPANIES = 2
        if analysis.is_comparison and len(analysis.companies) > MAX_COMPARISON_COMPANIES:
            companies_str = ", ".join(analysis.companies)
            reject_msg = (
                f"Mohon maaf, saat ini sistem hanya mendukung perbandingan maksimal {MAX_COMPARISON_COMPANIES} perusahaan sekaligus. "
                f"Anda menyebutkan {len(analysis.companies)} perusahaan: {companies_str}.\n\n"
                f"Silakan ajukan pertanyaan perbandingan dengan maksimal 2 perusahaan, misalnya:\n"
                f"  - \"Bandingkan emisi {analysis.companies[0]} dan {analysis.companies[1]}\"\n"
                f"  - \"Apa perbedaan keberlanjutan {analysis.companies[0]} vs {analysis.companies[1]}?\""
            )
            # Save to history
            session.add_message("user", query)
            session.add_message("assistant", reject_msg)
            self._save_session_to_file(session)

            return ChatResult(
                response=reject_msg,
                session_id=session_id,
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                session_token_usage=self.session_token_stats[session_id].get_summary(),
                analysis=analysis,
                pdf_count=0,
                data_count=0
            )

        # Step 2: Smart retrieval
        pdf_results, data_results = self.smart_retrieve(analysis, top_k_per_company=10)

        if not pdf_results and not data_results:
            no_result_msg = "Maaf, tidak ditemukan dokumen yang relevan dengan pertanyaan Anda."
            session.add_message("user", query)
            session.add_message("assistant", no_result_msg)
            self._save_session_to_file(session)

            return ChatResult(
                response=no_result_msg,
                session_id=session_id,
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                session_token_usage=self.session_token_stats[session_id].get_summary(),
                analysis=analysis,
                pdf_count=0,
                data_count=0
            )

        # Step 3: Rerank both result sets
        reranked_pdf = self.rerank_results(pdf_results, query, analysis, top_k=RERANK_TOP_K)
        reranked_data = self.rerank_results(data_results, query, analysis, top_k=RERANK_DATA_TOP_K)

        # Step 4: Format context with both sources
        context = self.format_context(reranked_pdf, reranked_data)

        # Step 5: Generate response
        response, token_usage = self.call_llm(query, context, analysis, session)

        # Step 6: Save to chat history
        session.add_message("user", query)
        session.add_message("assistant", response)
        self._save_session_to_file(session)

        # Update token statistics
        self.session_token_stats[session_id].add_usage(
            token_usage['prompt_tokens'],
            token_usage['completion_tokens']
        )

        return ChatResult(
            response=response,
            session_id=session_id,
            token_usage=token_usage,
            session_token_usage=self.session_token_stats[session_id].get_summary(),
            analysis=analysis,
            pdf_count=len(reranked_pdf),
            data_count=len(reranked_data)
        )

    # ==================== HEALTH CHECK ====================

    def health_check(self) -> Dict[str, str]:
        """Check health of all components."""
        components = {}

        # Check Qdrant
        try:
            collections = self.qdrant.get_collections()
            components["qdrant"] = "healthy"
        except Exception as e:
            components["qdrant"] = f"unhealthy: {str(e)}"

        # Check embedding model
        try:
            _ = self.embedding_model.encode("test")
            components["embedding_model"] = "healthy"
        except Exception as e:
            components["embedding_model"] = f"unhealthy: {str(e)}"

        # Check LLM endpoint
        try:
            response = requests.get(MODEL_ARK_API_URL, timeout=5)
            components["llm_endpoint"] = "healthy"
        except Exception as e:
            components["llm_endpoint"] = f"unhealthy: {str(e)}"

        return components


# Singleton instance
chatbot_service = ESGChatbotService()
