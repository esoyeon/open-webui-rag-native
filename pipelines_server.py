#!/usr/bin/env python3
"""
Open WebUI Pipelines Plugin Framework 서버
"""
import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from pipelines.adaptive_rag_pipeline import Pipe

app = FastAPI(title="Adaptive RAG Pipelines Server", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 파이프라인 인스턴스
pipeline_instance = Pipe()


@app.get("/pipelines")
async def get_pipelines():
    """사용 가능한 파이프라인 목록 반환"""
    return {
        "data": [
            {
                "id": "adaptive-rag",
                "name": "Adaptive RAG Pipeline",
                "type": "pipe",
                "description": "LangGraph 기반 Adaptive RAG with FAISS",
                "manifest": {"required_open_webui_version": "0.1.0"},
            }
        ]
    }


@app.post("/adaptive-rag/pipe")
async def pipe_adaptive_rag(request_data: dict):
    """Adaptive RAG 파이프라인 실행"""
    try:
        user_message = request_data.get("messages", [])[-1].get("content", "")
        model_id = request_data.get("model", "adaptive-rag")

        response = pipeline_instance.pipe(
            user_message=user_message,
            model_id=model_id,
            messages=request_data.get("messages", []),
            body=request_data.get("body", {}),
        )

        return {"response": response}
    except Exception as e:
        return {"error": str(e)}


@app.get("/adaptive-rag/valves")
async def get_valves():
    """파이프라인 설정값 반환"""
    return {
        "openai_api_key": "",
        "vector_store_path": "./vector_stores/spri_ai_brief",
        "max_retries": 3,
        "temperature": 0.7,
    }


@app.post("/adaptive-rag/valves/update")
async def update_valves(valves_data: dict):
    """파이프라인 설정값 업데이트"""
    # TODO: 실제 설정값 업데이트 로직 구현
    return {"status": "updated", "valves": valves_data}


if __name__ == "__main__":
    print("🔧 Open WebUI Pipelines Server Starting...")
    print("🌐 Server URL: http://localhost:9099")

    uvicorn.run(
        "pipelines_server:app", host="0.0.0.0", port=9099, reload=True, log_level="info"
    )
