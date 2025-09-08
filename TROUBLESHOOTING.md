# 🛠️ 문제 해결 및 FAQ

> **Open WebUI + 자체 RAG 시스템 통합 시 발생할 수 있는 모든 문제와 해결책**

## 🆘 **긴급 문제 해결**

### **🚫 서버가 시작되지 않는 경우**
```bash
# 1. 포트 충돌 확인
netstat -tulpn | grep :8000
# 또는 
lsof -i :8000

# 2. 다른 포트로 실행
python web_api_server.py --port 8001

# 3. 프로세스 강제 종료
pkill -f web_api_server
```

### **🔌 Open WebUI에서 "Connection Failed" 에러**
```bash
# Docker 컨테이너 내부에서 확인
docker exec -it open-webui curl http://host.docker.internal:8000/v1/models

# 호스트 네트워크 사용 (임시 해결책)
docker run --network host -p 3000:8080 ghcr.io/open-webui/open-webui:main
```

### **💥 "Internal Server Error" 발생**
```bash
# 로그 확인
tail -f rag_system.log

# 의존성 재설치
pip uninstall -y langchain langchain-openai faiss-cpu
pip install langchain langchain-openai faiss-cpu

# 환경 변수 확인
echo $OPENAI_API_KEY
```

---

## 📋 **일반적인 문제들**

### **🔑 API 키 관련 문제**

#### **문제**: `Error code: 401 - Incorrect API key`
```bash
# 해결 1: .env 파일 확인
cat .env
# OPENAI_API_KEY가 올바른지 확인

# 해결 2: 환경 변수 재로드
source .env
export $(cat .env | xargs)

# 해결 3: 쉘 변수 덮어쓰기 확인
unset OPENAI_API_KEY
source .env
```

#### **문제**: `.env` 파일이 로드되지 않음
```python
# adaptive_rag/nodes.py 또는 web_api_server.py 상단에 추가
from dotenv import load_dotenv
load_dotenv()  # 이 줄이 있는지 확인
```

### **🐳 Docker 관련 문제**

#### **문제**: Docker가 실행되지 않음
```bash
# macOS
open -a Docker

# Linux
sudo systemctl start docker

# Windows
# Docker Desktop 수동 실행
```

#### **문제**: `host.docker.internal` 연결 불가
```bash
# Linux에서는 다음 사용
--add-host=host.docker.internal:host-gateway

# 또는 호스트 IP 직접 사용
ip route show default | awk '/default/ {print $3}'
# 해당 IP를 API Base URL에 입력
```

#### **문제**: 컨테이너 포트 충돌
```bash
# 포트 변경
docker run -p 3001:8080 ghcr.io/open-webui/open-webui:main

# 실행 중인 컨테이너 정리
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
```

### **📊 성능 관련 문제**

#### **문제**: 응답이 너무 느림 (10초 이상)
```python
# adaptive_rag/nodes.py에서 검색 문서 수 줄이기
documents = self.vector_store.similarity_search(question, k=5)  # k=10에서 k=5로 변경
```

```python
# 캐싱 추가
import functools
import time

@functools.lru_cache(maxsize=100)
def cached_search(query_hash):
    return expensive_search(query)
```

#### **문제**: 메모리 사용량 급증
```python
# FAISS 인덱스 메모리 최적화
import faiss
import gc

# 인덱스 압축
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, 100)

# 정기적 가비지 컬렉션
gc.collect()
```

#### **문제**: 환상(Hallucination) 검증 무한 루프
```python
# adaptive_rag/grader.py에서 재시도 횟수 제한
class HallucinationGrader:
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
        self.retry_count = 0
    
    def grade(self, documents, generation):
        if self.retry_count >= self.max_retries:
            return "yes"  # 강제로 통과시킴
        
        # ... 기존 로직
```

### **🌐 네트워크 관련 문제**

#### **문제**: Tavily API 연결 실패
```python
# adaptive_rag/nodes.py에서 fallback 로직 확인
def search(self, query: str, max_results: int = 3) -> List[dict]:
    try:
        results = self.tavily_tool.invoke({"query": query})
        return formatted_results
    except Exception as e:
        logger.error(f"❌ Tavily 검색 실패: {e}")
        # Fallback: 빈 결과 또는 기본 메시지 반환
        return [{"content": f"죄송합니다. '{query}'에 대한 실시간 정보를 가져올 수 없습니다."}]
```

#### **문제**: OpenAI API 요청 제한 초과
```python
# 요청 제한 처리
import time
import random
from openai import RateLimitError

def api_call_with_retry(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait_time)
    raise Exception("API rate limit exceeded after all retries")
```

### **📄 PDF 처리 관련 문제**

#### **문제**: PDF 인덱싱 실패
```bash
# 디버깅 모드로 실행
python scripts/index_documents.py --docs-dir data --verbose

# 특정 PDF 파일 확인
python -c "
import PyPDF2
with open('data/your-file.pdf', 'rb') as f:
    pdf = PyPDF2.PdfReader(f)
    print(f'Pages: {len(pdf.pages)}')
    print(f'First page: {pdf.pages[0].extract_text()[:200]}')
"
```

#### **문제**: 한글 PDF 인식 불가
```python
# rag/pdf.py에서 인코딩 처리
import pdfplumber

def extract_text_with_encoding(pdf_path):
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # 한글 처리 개선
                text = text.encode('utf-8').decode('utf-8')
                texts.append(text)
    return texts
```

#### **문제**: 벡터 스토어 업데이트 실패
```bash
# 기존 인덱스 백업 후 재생성
cp -r data/vector_store data/vector_store_backup
rm -rf data/vector_store/*
python scripts/index_documents.py --docs-dir data --force-rebuild
```

---

## 🔧 **고급 문제 해결**

### **⚡ 성능 최적화**

#### **벡터 검색 속도 개선**
```python
# FAISS 인덱스 타입 변경
import faiss

# 기존: IndexFlatL2 (정확하지만 느림)
# 개선: IndexHNSWFlat (빠르지만 근사치)
dimension = 1536
index = faiss.IndexHNSWFlat(dimension, 32)
index.hnsw.efConstruction = 40
index.hnsw.efSearch = 16
```

#### **LLM 응답 속도 개선**
```python
# 1. 모델을 더 빠른 버전으로 변경
# GPT-4 → GPT-3.5-turbo
# GPT-3.5-turbo → GPT-3.5-turbo-1106

# 2. 응답 길이 제한
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    max_tokens=1000,  # 토큰 수 제한
    temperature=0.1   # 창의성 낮춰서 속도 향상
)

# 3. 스트리밍 활성화
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    stream=True  # 스트리밍으로 점진적 응답
)
```

### **🔒 보안 강화**

#### **API 키 보안**
```python
# 환경 변수 암호화
from cryptography.fernet import Fernet

def encrypt_api_key(api_key: str, encryption_key: bytes) -> bytes:
    f = Fernet(encryption_key)
    return f.encrypt(api_key.encode())

def decrypt_api_key(encrypted_key: bytes, encryption_key: bytes) -> str:
    f = Fernet(encryption_key)
    return f.decrypt(encrypted_key).decode()

# 사용법
encryption_key = Fernet.generate_key()  # 안전한 곳에 저장
encrypted_key = encrypt_api_key(os.getenv("OPENAI_API_KEY"), encryption_key)
```

#### **요청 제한 및 인증**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/v1/chat/completions")
@limiter.limit("10/minute")  # 분당 10회 제한
async def chat_completions(request: Request, ...):
    # API 키 검증
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not verify_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # ... 처리 로직
```

### **📊 모니터링 개선**

#### **상세한 로깅 시스템**
```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_request(self, user_query: str, response: str, duration: float):
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "request",
            "query_length": len(user_query),
            "response_length": len(response),
            "duration_seconds": duration,
            "query_preview": user_query[:100] + "..." if len(user_query) > 100 else user_query
        }
        self.logger.info(json.dumps(log_data, ensure_ascii=False))
    
    def log_error(self, error: Exception, context: dict = None):
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        self.logger.error(json.dumps(log_data, ensure_ascii=False))
```

#### **Prometheus 메트릭**
```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# 메트릭 정의
REQUEST_COUNT = Counter('rag_requests_total', 'Total requests', ['method', 'status'])
REQUEST_DURATION = Histogram('rag_request_duration_seconds', 'Request duration')
ACTIVE_SESSIONS = Gauge('rag_active_sessions', 'Active user sessions')
VECTOR_STORE_SIZE = Gauge('rag_vector_store_documents', 'Number of documents in vector store')

@app.get("/metrics")
async def get_metrics():
    return Response(generate_latest(), media_type="text/plain")

# 사용법
@REQUEST_DURATION.time()
async def process_query(query: str):
    REQUEST_COUNT.labels(method='POST', status='success').inc()
    # ... 처리 로직
```

---

## 🔍 **디버깅 도구**

### **🕵️ 시스템 상태 진단**

```python
# scripts/diagnose_system.py
import os
import sys
import psutil
import docker
import requests
from typing import Dict, Any

class SystemDiagnostics:
    def __init__(self):
        self.results = {}
    
    def check_environment(self) -> Dict[str, Any]:
        """환경 변수 및 파이썬 환경 확인"""
        env_check = {
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "environment_variables": {
                "OPENAI_API_KEY": "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Missing",
                "TAVILY_API_KEY": "✅ Set" if os.getenv("TAVILY_API_KEY") else "❌ Missing",
            }
        }
        return env_check
    
    def check_dependencies(self) -> Dict[str, Any]:
        """의존성 패키지 확인"""
        required_packages = [
            "langchain", "langchain-openai", "faiss-cpu", 
            "fastapi", "uvicorn", "pydantic"
        ]
        
        dependency_status = {}
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                dependency_status[package] = "✅ Installed"
            except ImportError:
                dependency_status[package] = "❌ Missing"
        
        return dependency_status
    
    def check_ports(self) -> Dict[str, Any]:
        """포트 사용 상태 확인"""
        ports_to_check = [8000, 3000, 9099]
        port_status = {}
        
        for port in ports_to_check:
            try:
                response = requests.get(f"http://localhost:{port}", timeout=2)
                port_status[f"port_{port}"] = f"✅ Active (Status: {response.status_code})"
            except requests.exceptions.RequestException:
                port_status[f"port_{port}"] = "❌ Not responding"
        
        return port_status
    
    def check_docker(self) -> Dict[str, Any]:
        """Docker 상태 확인"""
        try:
            client = docker.from_env()
            containers = client.containers.list()
            
            docker_info = {
                "docker_available": "✅ Running",
                "containers": []
            }
            
            for container in containers:
                if "open-webui" in container.name:
                    docker_info["containers"].append({
                        "name": container.name,
                        "status": container.status,
                        "ports": container.attrs["NetworkSettings"]["Ports"]
                    })
            
            return docker_info
            
        except Exception as e:
            return {"docker_available": f"❌ Error: {str(e)}"}
    
    def check_vector_store(self) -> Dict[str, Any]:
        """벡터 스토어 상태 확인"""
        vector_store_path = "data/vector_store"
        
        if os.path.exists(vector_store_path):
            files = os.listdir(vector_store_path)
            file_sizes = {}
            
            for file in files:
                file_path = os.path.join(vector_store_path, file)
                if os.path.isfile(file_path):
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    file_sizes[file] = f"{size_mb:.2f} MB"
            
            return {
                "vector_store_exists": "✅ Found",
                "files": file_sizes
            }
        else:
            return {"vector_store_exists": "❌ Not found"}
    
    def generate_report(self) -> str:
        """전체 진단 보고서 생성"""
        report = []
        report.append("🔍 시스템 진단 보고서")
        report.append("=" * 50)
        
        # 환경 확인
        env_check = self.check_environment()
        report.append("\n📋 환경 변수:")
        for key, value in env_check["environment_variables"].items():
            report.append(f"  {key}: {value}")
        
        # 의존성 확인
        deps = self.check_dependencies()
        report.append("\n📦 의존성 패키지:")
        for package, status in deps.items():
            report.append(f"  {package}: {status}")
        
        # 포트 확인
        ports = self.check_ports()
        report.append("\n🌐 포트 상태:")
        for port, status in ports.items():
            report.append(f"  {port}: {status}")
        
        # Docker 확인
        docker_status = self.check_docker()
        report.append(f"\n🐳 Docker: {docker_status.get('docker_available', 'Unknown')}")
        
        # 벡터 스토어 확인
        vs_status = self.check_vector_store()
        report.append(f"\n📚 Vector Store: {vs_status.get('vector_store_exists', 'Unknown')}")
        
        return "\n".join(report)

# 실행
if __name__ == "__main__":
    diagnostics = SystemDiagnostics()
    print(diagnostics.generate_report())
```

### **🧪 API 테스터**

```python
# scripts/test_api.py
import asyncio
import aiohttp
import json
from typing import List, Dict, Any

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
    
    async def test_models_endpoint(self) -> Dict[str, Any]:
        """모델 엔드포인트 테스트"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/v1/models") as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "✅ Success",
                            "models": [model["id"] for model in data.get("data", [])]
                        }
                    else:
                        return {"status": f"❌ HTTP {response.status}"}
            except Exception as e:
                return {"status": f"❌ Error: {str(e)}"}
    
    async def test_chat_endpoint(self, query: str) -> Dict[str, Any]:
        """채팅 엔드포인트 테스트"""
        payload = {
            "model": "adaptive-rag",
            "messages": [{"role": "user", "content": query}]
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                start_time = asyncio.get_event_loop().time()
                
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self.headers,
                    json=payload
                ) as response:
                    duration = asyncio.get_event_loop().time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        answer = data["choices"][0]["message"]["content"]
                        
                        return {
                            "status": "✅ Success",
                            "duration": f"{duration:.2f}s",
                            "response_length": len(answer),
                            "response_preview": answer[:100] + "..." if len(answer) > 100 else answer
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "status": f"❌ HTTP {response.status}",
                            "error": error_text[:200]
                        }
                        
            except Exception as e:
                return {"status": f"❌ Error: {str(e)}"}
    
    async def run_comprehensive_test(self) -> str:
        """종합 API 테스트"""
        report = []
        report.append("🧪 API 테스트 보고서")
        report.append("=" * 40)
        
        # 모델 엔드포인트 테스트
        models_result = await self.test_models_endpoint()
        report.append(f"\n🔧 Models Endpoint: {models_result['status']}")
        if "models" in models_result:
            report.append(f"   Available models: {', '.join(models_result['models'])}")
        
        # 채팅 엔드포인트 테스트
        test_queries = [
            "안녕하세요",
            "문서에서 AI 정책에 대해 알려주세요",
            "오늘 날씨는 어떤가요?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            result = await self.test_chat_endpoint(query)
            report.append(f"\n💬 Chat Test {i}: {result['status']}")
            if "duration" in result:
                report.append(f"   Query: {query}")
                report.append(f"   Duration: {result['duration']}")
                report.append(f"   Response: {result['response_preview']}")
        
        return "\n".join(report)

# 실행
async def main():
    tester = APITester()
    report = await tester.run_comprehensive_test()
    print(report)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## ❓ **자주 묻는 질문 (FAQ)**

### **🤖 시스템 구성**

**Q: 이 시스템은 오프라인에서 작동할 수 있나요?**
A: 부분적으로 가능합니다. 벡터 검색은 로컬에서 작동하지만, OpenAI API 호출이 필요합니다. 완전 오프라인을 원한다면 Ollama나 HuggingFace 로컬 모델로 교체해야 합니다.

**Q: 다른 언어 문서도 처리할 수 있나요?**
A: 네, 가능합니다. 다만 시스템 프롬프트를 해당 언어로 수정하고, 적절한 임베딩 모델을 선택해야 합니다.

**Q: 문서 수에 제한이 있나요?**
A: FAISS는 수백만 개의 문서도 처리할 수 있습니다. 다만 메모리와 검색 속도를 고려하여 적절한 인덱스 타입을 선택해야 합니다.

### **🔧 기술적 질문**

**Q: GPT-4 대신 Claude를 사용할 수 있나요?**
A: 네, `adaptive_rag/nodes.py`에서 OpenAI 클라이언트를 Anthropic 클라이언트로 교체하면 됩니다.

```python
from anthropic import Anthropic

class RAGNodes:
    def __init__(self):
        self.llm = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def generate_response(self, prompt):
        response = self.llm.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
```

**Q: 더 빠른 임베딩 모델로 교체할 수 있나요?**
A: 네, `adaptive_rag/vector_store.py`에서 임베딩 모델을 변경할 수 있습니다.

```python
from sentence_transformers import SentenceTransformer

# OpenAI 대신 로컬 모델 사용
embeddings = SentenceTransformer('all-MiniLM-L6-v2')
```

**Q: 스트리밍 응답을 지원하나요?**
A: 현재는 기본적으로 지원하지 않지만, `web_api_server.py`에서 스트리밍 로직을 추가할 수 있습니다.

### **🚀 운영 및 배포**

**Q: 프로덕션 환경에서 주의할 점은?**
A: 
1. API 키를 안전하게 관리
2. 로드밸런싱 및 헬스체크 구성
3. 로그 모니터링 및 알림 설정
4. 정기적인 벡터 스토어 백업

**Q: 비용을 줄이려면 어떻게 해야 하나요?**
A:
1. 캐싱을 적극적으로 활용
2. GPT-4 대신 GPT-3.5-turbo 사용
3. 응답 길이 제한
4. 로컬 임베딩 모델 사용

---

## 📞 **추가 지원**

### **🔍 문제가 해결되지 않는 경우**

1. **로그 확인**: 모든 에러 메시지를 자세히 확인
2. **버전 호환성**: 의존성 패키지 버전 확인
3. **환경 재구성**: 가상환경을 새로 만들어 테스트
4. **단계별 디버깅**: 각 컴포넌트를 개별적으로 테스트

### **🛠️ 디버깅 명령어 모음**

```bash
# 전체 시스템 진단
python scripts/diagnose_system.py

# API 종합 테스트
python scripts/test_api.py

# 의존성 확인
pip list | grep -E "(langchain|faiss|fastapi|openai)"

# 로그 실시간 모니터링
tail -f rag_system.log | grep -E "(ERROR|WARNING)"

# 시스템 리소스 확인
ps aux | grep -E "(python|docker)"
df -h  # 디스크 사용량
free -m  # 메모리 사용량
```

### **📋 이슈 리포팅 템플릿**

문제 발생 시 다음 정보를 포함해서 이슈를 등록해주세요:

```markdown
## 문제 설명
[구체적인 문제 상황 설명]

## 재현 단계
1. [단계 1]
2. [단계 2]
3. [단계 3]

## 예상 결과
[어떻게 작동해야 하는지]

## 실제 결과
[실제로 무엇이 발생했는지]

## 환경 정보
- OS: [운영체제]
- Python 버전: [버전]
- 주요 패키지 버전: [langchain, openai, faiss 등]
- Docker 사용 여부: [예/아니오]

## 로그 정보
```
[관련 에러 로그 복사]
```

## 시도한 해결책
[이미 시도해본 해결 방법들]
```

---

**🎯 대부분의 문제는 이 가이드를 통해 해결할 수 있습니다. 추가 도움이 필요하면 언제든 문의해주세요!**

**💡 팁: 문제 발생 시 당황하지 말고 차근차근 로그를 확인하는 것이 가장 중요합니다.**
