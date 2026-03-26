#!/usr/bin/env python3
"""
Pydantic Models for ESG Chatbot API
====================================
Request and response models for the API endpoints.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


# ============== Request Models ==============

class ChatRequest(BaseModel):
    """Request body untuk endpoint /chat."""
    query: str = Field(..., description="Pertanyaan pengguna", min_length=1)
    session_id: Optional[str] = Field(None, description="Session ID untuk melanjutkan percakapan")
    new_session: bool = Field(False, description="Jika True, buat session baru meskipun session_id sudah ada")


# ============== Response Models ==============

class TokenUsage(BaseModel):
    """Token usage untuk satu query."""
    prompt_tokens: int = Field(0, description="Jumlah token input")
    completion_tokens: int = Field(0, description="Jumlah token output")
    total_tokens: int = Field(0, description="Total token")


class SessionTokenUsage(BaseModel):
    """Token usage kumulatif untuk session."""
    total_queries: int = Field(0, description="Total pertanyaan dalam session")
    total_prompt_tokens: int = Field(0, description="Total token input session")
    total_completion_tokens: int = Field(0, description="Total token output session")
    total_tokens: int = Field(0, description="Total token session")
    avg_tokens_per_query: float = Field(0.0, description="Rata-rata token per pertanyaan")


class QueryAnalysis(BaseModel):
    """Hasil analisis query."""
    companies_detected: List[str] = Field(default_factory=list, description="Perusahaan yang terdeteksi")
    topics_detected: List[str] = Field(default_factory=list, description="Topik ESG yang terdeteksi")
    sectors: List[str] = Field(default_factory=list, description="Sektor yang terdeteksi")
    is_comparison: bool = Field(False, description="Apakah query perbandingan")


class SourcesSummary(BaseModel):
    """Ringkasan sumber dokumen yang digunakan."""
    pdf_count: int = Field(0, description="Jumlah sumber PDF")
    data_count: int = Field(0, description="Jumlah sumber data (Excel)")


class ChatResponse(BaseModel):
    """Response body untuk endpoint /chat."""
    success: bool = Field(True, description="Status keberhasilan request")
    session_id: str = Field(..., description="Session ID yang digunakan")
    response: str = Field(..., description="Jawaban dari chatbot")
    token_usage: TokenUsage = Field(default_factory=TokenUsage, description="Token usage untuk query ini")
    session_token_usage: SessionTokenUsage = Field(
        default_factory=SessionTokenUsage,
        description="Token usage kumulatif session"
    )
    analysis: QueryAnalysis = Field(default_factory=QueryAnalysis, description="Hasil analisis query")
    sources: SourcesSummary = Field(default_factory=SourcesSummary, description="Ringkasan sumber")
    processing_time_ms: int = Field(0, description="Waktu pemrosesan dalam milidetik")


class ErrorResponse(BaseModel):
    """Response untuk error."""
    success: bool = Field(False)
    error: str = Field(..., description="Pesan error")
    error_code: str = Field("INTERNAL_ERROR", description="Kode error")


# ============== Session Models ==============

class ChatMessage(BaseModel):
    """Single chat message."""
    role: str = Field(..., description="Role: 'user' atau 'assistant'")
    content: str = Field(..., description="Isi pesan")
    timestamp: str = Field(..., description="Waktu pesan dalam ISO format")


class SessionInfo(BaseModel):
    """Informasi session."""
    session_id: str
    created_at: str
    message_count: int


class SessionDetail(BaseModel):
    """Detail session dengan history."""
    session_id: str
    created_at: str
    messages: List[ChatMessage]


class SessionListResponse(BaseModel):
    """Response untuk list sessions."""
    success: bool = True
    sessions: List[SessionInfo]


class SessionDetailResponse(BaseModel):
    """Response untuk session detail."""
    success: bool = True
    session: SessionDetail


# ============== Company Models ==============

class CompanyInfo(BaseModel):
    """Informasi perusahaan."""
    name: str
    sector: str


class CompanyListResponse(BaseModel):
    """Response untuk daftar perusahaan."""
    success: bool = True
    total_companies: int
    total_sectors: int
    companies_by_sector: Dict[str, List[str]]


# ============== Health Check ==============

class HealthResponse(BaseModel):
    """Response untuk health check."""
    status: str = "healthy"
    version: str = "1.0.0"
    components: Dict[str, str] = Field(default_factory=dict)
