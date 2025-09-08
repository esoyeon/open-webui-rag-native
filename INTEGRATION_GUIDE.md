# 🔗 Open WebUI + 자체 RAG/DB/Embedding API 완전 통합 가이드

> **Open WebUI를 자체 데이터베이스, RAG 시스템, Embedding API와 통합하는 단계별 실전 가이드**

## 🎯 **가이드 목적**

이 가이드는 **Open WebUI**를 기존의 OpenAI API 대신 **자체적인 RAG 시스템, 데이터베이스, Embedding API**와 연결하려는 모든 개발자를 위한 완전한 매뉴얼입니다.

### **적용 가능한 시나리오**
- 🏢 **기업 내부 문서** 기반 AI 챗봇
- 📚 **전문 도메인 지식** 기반 질의응답 시스템
- 🔒 **프라이빗 데이터** 보안이 중요한 환경
- 💰 **비용 최적화**를 위한 자체 인프라 활용
- 🌐 **다국어 지원**이 필요한 특화 시스템

---

## 📋 **통합 방법 개요**

Open WebUI와 자체 시스템을 통합하는 **3가지 주요 방법**:

### **1. OpenAI 호환 API 서버 방식** ⭐ (권장)
- ✅ **가장 안정적이고 범용적**
- ✅ 기존 OpenAI 클라이언트와 완전 호환
- ✅ Docker 환경에서 안정적 작동
- ✅ 다른 도구들과도 연동 가능

### **2. Pipelines Plugin Framework 방식**
- ✅ Open WebUI 네이티브 통합
- ✅ 더 깊은 레벨의 커스터마이징 가능
- ⚠️ Open WebUI 전용, 상대적으로 복잡

### **3. 직접 FastAPI 통합 방식**
- ✅ 완전한 제어 및 커스터마이징
- ⚠️ 높은 개발 복잡도
- ⚠️ 유지보수 부담

---

## 🚀 **방법 1: OpenAI 호환 API 서버** (권장)

### **핵심 아이디어**
OpenAI API와 동일한 엔드포인트(`/v1/chat/completions`, `/v1/models`)를 제공하는 FastAPI 서버를 만들어, Open WebUI가 마치 OpenAI를 사용하는 것처럼 자체 시스템을 사용하게 합니다.

### **1단계: 기본 FastAPI 서버 구조**

```python
# web_api_server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

app = FastAPI(title="Custom RAG API Server")

# CORS 설정 (Open WebUI 연동을 위해 필수)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI 호환 데이터 모델
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2000

class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-custom"
    object: str = "chat.completion"
    created: int = 1677610602
    model: str
    choices: List[Dict[str, Any]]

# 필수 엔드포인트 1: 모델 목록
@app.get("/v1/models")
async def get_models():
    return {
        "data": [
            {
                "id": "your-custom-model",
                "object": "model", 
                "created": 1677610602,
                "owned_by": "custom-api",
            }
        ]
    }

# 필수 엔드포인트 2: 채팅 완료
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        # 여기에 자체 RAG 로직 연결
        answer = process_with_your_rag_system(request.messages[-1].content)
        
        return ChatCompletionResponse(
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant", 
                    "content": answer
                },
                "finish_reason": "stop"
            }]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_with_your_rag_system(user_question: str) -> str:
    """여기에 자체 RAG 시스템 로직을 구현"""
    # 예시: 벡터 검색 + LLM 생성
    documents = your_vector_store.search(user_question)
    context = "\n".join([doc.content for doc in documents])
    
    prompt = f"Context: {context}\n\nQuestion: {user_question}\n\nAnswer:"
    answer = your_llm.generate(prompt)
    
    return answer

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### **2단계: RAG 시스템 통합**

#### **A. 벡터 스토어 연결**
```python
# vector_store.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class CustomVectorStore:
    def __init__(self, index_path: str, documents_path: str):
        self.index = faiss.read_index(index_path)
        self.documents = self.load_documents(documents_path)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def search(self, query: str, k: int = 5):
        # 쿼리 임베딩
        query_vector = self.encoder.encode([query])
        
        # FAISS 검색
        scores, indices = self.index.search(query_vector, k)
        
        # 문서 반환
        return [self.documents[idx] for idx in indices[0]]
```

#### **B. LLM 생성기 연결**
```python
# llm_generator.py
from openai import OpenAI
# 또는 from transformers import pipeline

class CustomLLMGenerator:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.client = OpenAI()  # 또는 로컬 모델
        self.model_name = model_name
    
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
```

### **3단계: Open WebUI 설정**

#### **Docker로 Open WebUI 실행**
```bash
docker run -d \
  --name open-webui \
  -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=dummy-key \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

#### **브라우저에서 수동 설정**
1. **http://localhost:3000** 접속
2. **Admin Panel** → **Settings** → **Connections**
3. **OpenAI API** 섹션에서:
   - **API Base URL**: `http://host.docker.internal:8000/v1`
   - **API Key**: `dummy-key` (아무 값)

### **4단계: 고급 기능 추가**

#### **A. 스트리밍 응답 지원**
```python
from fastapi.responses import StreamingResponse
import json

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        return StreamingResponse(
            stream_response(request), 
            media_type="text/plain"
        )
    # ... 기존 로직

async def stream_response(request):
    """스트리밍 응답 생성"""
    answer = process_with_your_rag_system(request.messages[-1].content)
    
    # 토큰별로 스트리밍
    for token in answer.split():
        chunk = {
            "id": "chatcmpl-custom",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"content": token + " "},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    
    # 종료 신호
    yield f"data: [DONE]\n\n"
```

#### **B. 문서 업로드 지원**
```python
from fastapi import UploadFile, File

@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    """PDF 문서 업로드 및 벡터화"""
    content = await file.read()
    
    # PDF 처리 및 청킹
    documents = process_pdf(content)
    
    # 벡터화 및 인덱스 업데이트  
    embeddings = embed_documents(documents)
    update_vector_index(embeddings, documents)
    
    return {"message": f"{file.filename} 업로드 완료"}
```

---

## 🔧 **방법 2: Pipelines Plugin Framework**

### **핵심 아이디어**
Open WebUI의 Pipelines Plugin 시스템을 사용하여 더 깊은 레벨에서 통합합니다.

### **1단계: Pipe 클래스 구현**

```python
# pipelines/custom_rag_pipeline.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class Pipeline:
    class Valves(BaseModel):
        priority: int = 0
        temperature: float = 0.7
        
    def __init__(self):
        self.type = "manifold"
        self.name = "Custom RAG Pipeline"
        self.valves = self.Valves()
        
        # 자체 RAG 시스템 초기화
        self.vector_store = load_vector_store()
        self.llm = load_llm_model()
    
    async def on_startup(self):
        """파이프라인 시작 시 초기화"""
        print("Custom RAG Pipeline 시작")
    
    async def on_shutdown(self):
        """파이프라인 종료 시 정리"""
        print("Custom RAG Pipeline 종료")
    
    def pipe(
        self, 
        user_message: str, 
        model_id: str, 
        messages: List[Dict[str, str]], 
        body: Dict[str, Any]
    ) -> str:
        """메인 처리 함수"""
        try:
            # 벡터 검색
            documents = self.vector_store.search(user_message)
            
            # 컨텍스트 구성
            context = "\n".join([doc.content for doc in documents])
            
            # 프롬프트 생성
            prompt = f"Context: {context}\n\nQuestion: {user_message}\n\nAnswer:"
            
            # LLM 생성
            answer = self.llm.generate(prompt)
            
            return answer
            
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"
```

### **2단계: Pipelines 서버 실행**

```python
# pipelines_server.py
from fastapi import FastAPI
from pipelines.custom_rag_pipeline import Pipeline

app = FastAPI()

# 파이프라인 등록
pipeline = Pipeline()

@app.get("/")
async def get_status():
    return {"status": "Pipelines Server Running"}

@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    messages = request.get("messages", [])
    user_message = messages[-1]["content"] if messages else ""
    
    response = pipeline.pipe(
        user_message=user_message,
        model_id=request.get("model", ""),
        messages=messages,
        body=request
    )
    
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": response
            }
        }]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9099)
```

### **3단계: Open WebUI에서 Pipelines 설정**
1. **Admin Panel** → **Settings** → **Pipelines**
2. **Pipeline URL**: `http://host.docker.internal:9099`
3. 파이프라인 목록에서 **Custom RAG Pipeline** 선택

---

## 🏗️ **방법 3: 완전 커스텀 통합**

### **고급 사용자를 위한 완전한 제어**

```python
# custom_integration.py
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI()

# 정적 파일 서빙 (프론트엔드)
app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")

@app.get("/")
async def get_frontend():
    """커스텀 프론트엔드 제공"""
    return HTMLResponse(open("frontend/dist/index.html").read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """실시간 채팅 WebSocket"""
    await websocket.accept()
    
    while True:
        # 사용자 메시지 수신
        user_message = await websocket.receive_text()
        
        # 자체 RAG 처리
        response = process_message(user_message)
        
        # 응답 전송
        await websocket.send_text(response)

@app.post("/api/chat")
async def chat_api(request: dict):
    """REST API 엔드포인트"""
    message = request.get("message")
    return {"response": process_message(message)}

def process_message(message: str) -> str:
    """자체 RAG 로직"""
    # 벡터 검색 + LLM 생성
    return "AI 응답"
```

---

## 🔍 **핵심 구현 패턴**

### **1. 문서 처리 파이프라인**
```python
def create_document_pipeline():
    """문서 → 청킹 → 임베딩 → 인덱싱"""
    
    # PDF 로더
    loader = PyPDFLoader()
    
    # 텍스트 분할기  
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # 임베딩 모델
    embeddings = OpenAIEmbeddings()
    # 또는 로컬: HuggingFaceEmbeddings()
    
    # 벡터 스토어
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    return vector_store
```

### **2. 검색 최적화**
```python
def hybrid_search(query: str, vector_store, bm25_retriever):
    """하이브리드 검색: 벡터 + 키워드"""
    
    # 벡터 검색
    vector_docs = vector_store.similarity_search(query, k=5)
    
    # BM25 키워드 검색
    bm25_docs = bm25_retriever.get_relevant_documents(query)
    
    # 결과 결합 및 재랭킹
    combined_docs = combine_and_rerank(vector_docs, bm25_docs)
    
    return combined_docs
```

### **3. 응답 품질 개선**
```python
def generate_enhanced_response(query: str, documents: List[Document]):
    """구조화된 고품질 응답 생성"""
    
    context = "\n".join([doc.page_content for doc in documents])
    
    prompt = f"""
    당신은 전문 AI 어시스턴트입니다.
    다음 문서들을 바탕으로 질문에 답변해주세요.
    
    **답변 구조:**
    ## 📋 핵심 내용
    [주요 내용]
    
    ## 🔍 상세 분석
    [구체적 분석]
    
    ## 📊 관련 데이터
    [수치, 통계 등]
    
    ## 💡 결론
    [요약 및 시사점]
    
    문서: {context}
    
    질문: {query}
    
    답변:
    """
    
    return llm.invoke(prompt)
```

---

## 📊 **성능 최적화 가이드**

### **1. 벡터 스토어 최적화**
```python
# FAISS 인덱스 최적화
index = faiss.IndexFlatIP(dimension)  # 내적 기반 (더 빠름)
# index = faiss.IndexHNSWFlat(dimension, 32)  # 메모리 효율적

# 인덱스 훈련 및 압축
if len(documents) > 10000:
    index.train(embeddings)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
```

### **2. 캐싱 전략**
```python
from functools import lru_cache
import redis

# 메모리 캐시
@lru_cache(maxsize=1000)
def cached_search(query: str):
    return vector_store.search(query)

# Redis 캐시  
redis_client = redis.Redis()

def search_with_cache(query: str):
    cache_key = f"search:{hash(query)}"
    
    # 캐시 확인
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # 검색 수행
    results = vector_store.search(query)
    
    # 캐시 저장 (1시간)
    redis_client.setex(cache_key, 3600, json.dumps(results))
    
    return results
```

### **3. 비동기 처리**
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def async_rag_processing(query: str):
    """비동기 RAG 처리"""
    
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=4)
    
    # 병렬 처리
    tasks = [
        loop.run_in_executor(executor, vector_search, query),
        loop.run_in_executor(executor, keyword_search, query),
        loop.run_in_executor(executor, web_search, query)
    ]
    
    vector_docs, keyword_docs, web_docs = await asyncio.gather(*tasks)
    
    # 결과 결합
    all_docs = combine_results(vector_docs, keyword_docs, web_docs)
    
    # 답변 생성
    response = await generate_response(query, all_docs)
    
    return response
```

---

## 🛡️ **보안 및 인증**

### **API 키 인증**
```python
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer

security = HTTPBearer()

def verify_api_key(token: str = Depends(security)):
    if token.credentials != "your-secret-api-key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return token

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    token: str = Depends(verify_api_key)
):
    # ... 처리 로직
```

### **사용량 제한**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/v1/chat/completions")
@limiter.limit("10/minute")  # 분당 10회 제한
async def chat_completions(request: Request, ...):
    # ... 처리 로직
```

---

## 🚀 **배포 가이드**

### **Docker Compose로 전체 시스템 배포**
```yaml
# docker-compose.yml
version: '3.8'

services:
  # 자체 RAG API 서버
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    networks:
      - ai-network

  # Open WebUI
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    environment:
      - OPENAI_API_BASE_URL=http://rag-api:8000/v1
      - OPENAI_API_KEY=dummy-key
    depends_on:
      - rag-api
    networks:
      - ai-network
    volumes:
      - open-webui-data:/app/backend/data

  # Redis (캐싱)
  redis:
    image: redis:alpine
    networks:
      - ai-network

networks:
  ai-network:
    driver: bridge

volumes:
  open-webui-data:
```

### **Kubernetes 배포**
```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: your-registry/rag-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
---
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
spec:
  selector:
    app: rag-api
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

---

## 📈 **모니터링 및 로깅**

### **상세 로깅**
```python
import logging
from datetime import datetime

# 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    start_time = datetime.now()
    
    logger.info(f"질문 수신: {request.messages[-1].content[:100]}...")
    
    try:
        response = process_rag(request)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"응답 완료 - 소요시간: {duration:.2f}초")
        
        return response
        
    except Exception as e:
        logger.error(f"처리 실패: {str(e)}")
        raise
```

### **메트릭 수집**
```python
from prometheus_client import Counter, Histogram, generate_latest

# 메트릭 정의
REQUEST_COUNT = Counter('rag_requests_total', '총 요청 수')
REQUEST_DURATION = Histogram('rag_request_duration_seconds', '요청 처리 시간')
ERROR_COUNT = Counter('rag_errors_total', '오류 수')

@app.get("/metrics")
async def get_metrics():
    """Prometheus 메트릭 엔드포인트"""
    return Response(generate_latest(), media_type="text/plain")

@REQUEST_DURATION.time()
async def timed_rag_processing(query: str):
    REQUEST_COUNT.inc()
    
    try:
        result = process_rag(query)
        return result
    except Exception as e:
        ERROR_COUNT.inc()
        raise
```

---

## 🎯 **결론**

이 가이드를 통해 **Open WebUI를 자체 RAG/DB/Embedding API와 완벽하게 통합**할 수 있습니다.

### **권장 접근법**
1. **시작**: OpenAI 호환 API 서버로 프로토타입 구축
2. **발전**: 성능 최적화 및 고급 기능 추가  
3. **운영**: Docker/K8s로 프로덕션 배포

### **성공을 위한 핵심 포인트**
- ✅ **표준 준수**: OpenAI API 스펙 완벽 구현
- ✅ **성능 최적화**: 캐싱, 비동기 처리, 인덱스 튜닝
- ✅ **안정성**: 에러 처리, 로깅, 모니터링
- ✅ **확장성**: 컨테이너화, 로드밸런싱

**🚀 이제 자신만의 AI 통합 시스템을 구축해보세요!**
