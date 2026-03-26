#!/usr/bin/env python3
"""
ESG Chatbot API
================
FastAPI application for ESG RAG Chatbot.
All logic is exactly the same as chatbot_canggih.py.
"""

import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    TokenUsage,
    SessionTokenUsage,
    QueryAnalysis,
    SourcesSummary,
    SessionListResponse,
    SessionDetailResponse,
    SessionInfo,
    SessionDetail,
    ChatMessage,
    CompanyListResponse,
    HealthResponse,
)
from chatbot_service import chatbot_service, COMPANY_SECTOR_MAP

# Initialize FastAPI app
app = FastAPI(
    title="ESG Chatbot API",
    description="API untuk ESG RAG Chatbot - Intelligent ESG Data Assistant",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Health Check ==============

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check health status of all components."""
    components = chatbot_service.health_check()
    overall_status = "healthy" if all(v == "healthy" for v in components.values()) else "degraded"
    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        components=components
    )


# ============== Chat Endpoint ==============

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main chat endpoint untuk bertanya tentang ESG.

    - **query**: Pertanyaan pengguna (wajib)
    - **session_id**: ID session untuk melanjutkan percakapan (opsional)
    - **new_session**: Jika True, buat session baru meskipun session_id sudah ada

    Contoh pertanyaan:
    - "Berapa emisi GRK Bank Jago tahun 2024?"
    - "Bandingkan emisi Bank Jago vs Bank Jatim"
    - "Apa strategi keberlanjutan Bank Jago?"
    - "Perusahaan apa saja yang tersedia?"
    """
    start_time = time.time()

    try:
        # Call chatbot service
        result = chatbot_service.chat(
            query=request.query,
            session_id=request.session_id,
            new_session=request.new_session
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        return ChatResponse(
            success=True,
            session_id=result.session_id,
            response=result.response,
            token_usage=TokenUsage(
                prompt_tokens=result.token_usage.get("prompt_tokens", 0),
                completion_tokens=result.token_usage.get("completion_tokens", 0),
                total_tokens=result.token_usage.get("total_tokens", 0)
            ),
            session_token_usage=SessionTokenUsage(
                total_queries=result.session_token_usage.get("total_queries", 0),
                total_prompt_tokens=result.session_token_usage.get("total_prompt_tokens", 0),
                total_completion_tokens=result.session_token_usage.get("total_completion_tokens", 0),
                total_tokens=result.session_token_usage.get("total_tokens", 0),
                avg_tokens_per_query=result.session_token_usage.get("avg_tokens_per_query", 0.0)
            ),
            analysis=QueryAnalysis(
                companies_detected=result.analysis.companies,
                topics_detected=result.analysis.topics,
                sectors=result.analysis.sectors,
                is_comparison=result.analysis.is_comparison
            ),
            sources=SourcesSummary(
                pdf_count=result.pdf_count,
                data_count=result.data_count
            ),
            processing_time_ms=processing_time_ms
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Session Endpoints ==============

@app.get("/sessions", response_model=SessionListResponse, tags=["Sessions"])
async def list_sessions():
    """
    List all available chat sessions.
    """
    sessions = chatbot_service.list_sessions()
    return SessionListResponse(
        success=True,
        sessions=[
            SessionInfo(
                session_id=s["session_id"],
                created_at=s["created_at"],
                message_count=s["message_count"]
            )
            for s in sessions
        ]
    )


@app.get("/sessions/{session_id}", response_model=SessionDetailResponse, tags=["Sessions"])
async def get_session(session_id: str):
    """
    Get detail of a specific session including chat history.
    """
    session = chatbot_service.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return SessionDetailResponse(
        success=True,
        session=SessionDetail(
            session_id=session.session_id,
            created_at=session.created_at,
            messages=[
                ChatMessage(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp
                )
                for msg in session.messages
            ]
        )
    )


# ============== Company Endpoints ==============

@app.get("/companies", response_model=CompanyListResponse, tags=["Companies"])
async def list_companies():
    """
    Get list of all available companies grouped by sector.
    """
    companies_by_sector = chatbot_service.get_companies_by_sector()

    total_companies = sum(len(companies) for companies in companies_by_sector.values())
    total_sectors = len(companies_by_sector)

    return CompanyListResponse(
        success=True,
        total_companies=total_companies,
        total_sectors=total_sectors,
        companies_by_sector=companies_by_sector
    )


# ============== Root Endpoint ==============

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint dengan informasi API.
    """
    return {
        "name": "ESG Chatbot API",
        "version": "1.0.0",
        "description": "API untuk ESG RAG Chatbot - Intelligent ESG Data Assistant",
        "endpoints": {
            "chat": "POST /chat - Main chat endpoint",
            "sessions": "GET /sessions - List all sessions",
            "session_detail": "GET /sessions/{id} - Get session detail",
            "companies": "GET /companies - List available companies",
            "health": "GET /health - Health check",
            "docs": "GET /docs - API documentation (Swagger UI)",
            "redoc": "GET /redoc - API documentation (ReDoc)"
        }
    }


# ============== Run Application ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
