#!/usr/bin/env python3
"""
⚠️ DEPRECATED: 이 서버는 유지보수 종료되었습니다. 운영 환경에서는 `enhanced_api_server.py`를 사용하세요.

🤖 한국어 지원 Adaptive RAG API 서버

OpenAI 호환 API 형식으로 LangGraph 기반 Adaptive RAG 시스템을 제공합니다.
- FAISS 벡터 스토어를 통한 문서 검색
- 한국어 최적화된 프롬프트 시스템  
- 자가 수정 및 품질 검증 워크플로우
- Open WebUI 및 모든 OpenAI 호환 클라이언트 지원

주요 엔드포인트:
- GET /v1/models: 사용 가능한 모델 목록 (adaptive-rag)
- POST /v1/chat/completions: OpenAI 호환 채팅 완료
- GET /: 서버 상태 및 파이프라인 정보
- POST /api/documents: 새로운 문서 추가

사용 예시:
    python web_api_server.py
    curl -X POST "http://localhost:8000/v1/chat/completions" \\
         -H "Content-Type: application/json" \\
         -d '{"model": "adaptive-rag", "messages": [{"role": "user", "content": "안녕하세요!"}]}'
"""
import os
import sys
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from pipelines.adaptive_rag_pipeline import Pipe

# FastAPI 앱 초기화
app = FastAPI(title="Adaptive RAG API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 파이프라인 초기화
pipeline = Pipe()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "adaptive-rag"
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 1000


class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]


@app.get("/")
async def root():
    return {
        "message": "Adaptive RAG Pipeline API Server",
        "status": "running",
        "pipeline_status": pipeline.get_status(),
    }


@app.get("/api/models")
@app.get("/v1/models")
async def get_models():
    """OpenAI 호환 모델 목록"""
    return {
        "data": [
            {
                "id": "adaptive-rag",
                "object": "model",
                "created": 1677610602,
                "owned_by": "adaptive-rag",
            }
        ]
    }


@app.post("/api/chat")
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI 호환 채팅 완료 엔드포인트"""
    try:
        # 마지막 메시지 추출
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        user_message = request.messages[-1].content

        # 파이프라인 실행
        response_content = pipeline.pipe(
            user_message=user_message,
            model_id=request.model,
            messages=[msg.dict() for msg in request.messages],
            body={"temperature": request.temperature, "max_tokens": request.max_tokens},
        )

        # OpenAI 호환 응답 형식
        return ChatResponse(
            id="chatcmpl-adaptive-rag",
            object="chat.completion",
            created=1677610602,
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_content},
                    "finish_reason": "stop",
                }
            ],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.get("/api/status")
async def get_pipeline_status():
    """파이프라인 상태 확인"""
    return pipeline.get_status()


@app.post("/api/documents")
async def add_documents(documents: List[str]):
    """문서 추가"""
    try:
        result = pipeline.add_documents(documents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document add error: {str(e)}")


if __name__ == "__main__":
    print("⚠️ [DEPRECATED] web_api_server.py는 더 이상 권장되지 않습니다. enhanced_api_server.py를 사용하세요.")
    print("🚀 Adaptive RAG API Server Starting (legacy)...")
    print("📊 Pipeline Status:", pipeline.get_status())
    print("🌐 Server URL: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")

    uvicorn.run(
        "web_api_server:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
