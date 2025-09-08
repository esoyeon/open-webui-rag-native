# 🔗 Open WebUI ↔ RAG 시스템 연결을 위한 핵심 파일 가이드

Adaptive RAG LangGraph 베이스라인 모듈이 있다고 가정할 때, Open WebUI와 연결하기 위해 수정/생성해야 할 핵심 파일들을 정리합니다.

## 📋 **수정해야 할 핵심 파일 목록**

### **🔧 1. API 서버 파일 (필수 생성)**
```python
# web_api_server.py - 새로 생성 필요
```

**역할**: OpenAI 호환 API 서버로 Open WebUI와 통신

**핵심 기능**:
- `/v1/models` 엔드포인트 (모델 목록)
- `/v1/chat/completions` 엔드포인트 (채팅 완료)
- CORS 설정으로 Open WebUI 접근 허용

**핵심 코드 구조**:
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pipelines.adaptive_rag_pipeline import Pipe

app = FastAPI(title="Adaptive RAG API")

# CORS 설정
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# 파이프라인 초기화
pipeline = Pipe()

@app.get("/v1/models")
async def get_models():
    return {"data": [{"id": "adaptive-rag", "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    response = pipeline.pipe(
        user_message=request.messages[-1]["content"],
        model_id=request.model,
        messages=request.messages,
        body=request.dict()
    )
    return {"choices": [{"message": {"content": response}}]}
```

### **🔧 2. 파이프라인 어댑터 파일 (필수 생성)**
```python
# pipelines/adaptive_rag_pipeline.py - 새로 생성 필요
```

**역할**: LangGraph RAG 시스템을 Open WebUI Pipe 인터페이스로 래핑

**핵심 기능**:
- `Pipe` 클래스로 Open WebUI 인터페이스 구현
- 벡터 스토어 초기화 및 로드
- `pipe()` 메서드로 질의응답 처리

**핵심 코드 구조**:
```python
class Pipe:
    def __init__(self):
        # 벡터 스토어 초기화
        self.vector_store = FAISSVectorStore(embeddings)
        self.vector_store.load('data/vector_store')
        
        # LangGraph RAG 시스템 초기화
        self.rag_graph = AdaptiveRAGGraph(
            vector_store=self.vector_store,
            model_name="gpt-3.5-turbo"
        )
    
    def pipe(self, user_message, model_id, messages, body):
        # LangGraph 실행
        result = self.rag_graph.run({"question": user_message})
        return result["answer"]
    
    def get_status(self):
        return {"initialized": True, "total_documents": len(self.vector_store.documents)}
```

### **🔧 3. 환경 설정 파일 (필수 생성)**
```python
# .env - 새로 생성 필요
```

**역할**: API 키 및 설정 관리

**필수 키**:
```bash
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### **🔧 4. 의존성 관리 파일 (수정 필요)**
```python
# pyproject.toml - 수정 필요
```

**역할**: Python 패키지 의존성 관리

**추가 필요 패키지**:
```toml
[tool.uv]
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "langchain>=0.1.0",
    "langgraph>=0.0.40",
    "faiss-cpu>=1.7.4",
    "openai>=1.0.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
]
```

## 🛠️ **구체적인 수정 가이드**

### **1단계: API 서버 생성**
1. `web_api_server.py` 파일 생성
2. FastAPI 앱 설정
3. OpenAI 호환 엔드포인트 구현
4. CORS 미들웨어 추가

### **2단계: 파이프라인 어댑터 생성**
1. `pipelines/` 디렉토리 생성
2. `adaptive_rag_pipeline.py` 파일 생성
3. `Pipe` 클래스 구현
4. LangGraph RAG 시스템과 연결

### **3단계: 환경 설정**
1. `.env` 파일 생성
2. API 키 설정
3. `pyproject.toml` 의존성 추가

### **4단계: 벡터 스토어 준비**
1. `data/vector_store/` 디렉토리 생성
2. FAISS 인덱스 파일 생성
3. 문서 임베딩 완료

## 🎯 **핵심 연결 포인트**

### **Open WebUI → API 서버**
```
Open WebUI (Frontend)
    ↓ HTTP Request
API Server (web_api_server.py)
    ↓ Function Call
Pipeline Adapter (adaptive_rag_pipeline.py)
    ↓ Method Call
LangGraph RAG System (adaptive_rag/)
```

### **데이터 흐름**
```
사용자 질문 → Open WebUI → API 서버 → 파이프라인 → LangGraph → 벡터 스토어 → 답변 생성
```

## ✅ **검증 체크리스트**

- [ ] API 서버가 `/v1/models` 엔드포인트 제공
- [ ] API 서버가 `/v1/chat/completions` 엔드포인트 제공
- [ ] 파이프라인이 LangGraph RAG 시스템과 연결됨
- [ ] 벡터 스토어가 정상 로드됨
- [ ] 환경 변수가 올바르게 설정됨
- [ ] CORS 설정으로 Open WebUI 접근 허용됨

## 🚀 **실행 순서**

1. **환경 설정**: `.env` 파일 생성 및 API 키 설정
2. **의존성 설치**: `uv pip install -e .`
3. **벡터 스토어 준비**: 문서 인덱싱 완료
4. **API 서버 실행**: `python web_api_server.py`
5. **Open WebUI 연결**: `http://localhost:8000/v1` 백엔드 설정

이 가이드를 따라하면 기존 LangGraph RAG 시스템을 Open WebUI와 완벽하게 연결할 수 있습니다!
