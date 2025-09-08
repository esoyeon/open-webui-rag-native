# 📁 프로젝트 구조 가이드

> **정리된 프로젝트 구조와 각 컴포넌트의 역할 설명**

## 🏗️ **전체 구조**

```
llm_chatbot/
├── 📚 **Core Modules**
│   ├── adaptive_rag/           # 🧠 메인 RAG 엔진 (LangGraph 기반)
│   ├── document_processing/    # 📄 문서 처리 모듈
│   └── pipelines/             # 🔌 Open WebUI 통합 레이어
│
├── 🌐 **API Servers**
│   ├── web_api_server.py      # OpenAI 호환 API 서버 (메인)
│   └── pipelines_server.py    # Pipelines Plugin 서버 (대안)
│
├── 🛠️ **Scripts & Tools**
│   └── scripts/               # 유틸리티 스크립트들
│
├── 📊 **Data & Storage**
│   └── data/                  # PDF 문서 및 벡터 스토어
│
└── 📖 **Documentation**
    ├── README.md              # 메인 프로젝트 설명
    ├── INTEGRATION_GUIDE.md   # 통합 완전 가이드
    ├── METHODOLOGY.md         # 방법론 문서
    ├── TROUBLESHOOTING.md     # 문제 해결
    └── HOW_TO_RUN.md          # 빠른 실행 가이드
```

---

## 🧠 **Core Modules**

### **adaptive_rag/** - 메인 RAG 엔진
```
adaptive_rag/
├── __init__.py       # 모듈 진입점
├── router.py         # 쿼리 라우팅 (벡터스토어 vs 웹검색)
├── nodes.py          # LangGraph 노드 (검색, 생성, 검증)
├── grader.py         # 품질 검증 (문서 관련성, 환상 검증)
├── rewriter.py       # 쿼리 재작성 (검색 최적화)
├── vector_store.py   # FAISS 벡터 스토어
└── graph.py          # LangGraph 워크플로우 오케스트레이션
```

**역할:** LangGraph 기반 고도화된 RAG 워크플로우
- ✅ 자가 수정 (Self-correction)
- ✅ 환상 검증 (Hallucination checking)  
- ✅ 지능형 라우팅 (문서 검색 vs 웹 검색)
- ✅ 품질 보장 (Multiple grading layers)

### **document_processing/** - 문서 처리
```
document_processing/
├── __init__.py       # 모듈 진입점  
├── base.py           # 기본 RetrievalChain 추상 클래스
├── pdf.py            # PDF 처리 및 청킹
└── utils.py          # 유틸리티 함수들
```

**역할:** PDF 문서 로딩, 청킹, 기본 RAG 체인
- 📄 PDF 파일 로딩 (PDFPlumber)
- ✂️ 텍스트 청킹 (RecursiveCharacterTextSplitter)
- 🔧 기본 RAG 유틸리티

### **pipelines/** - Open WebUI 통합
```
pipelines/
└── adaptive_rag_pipeline.py   # Pipe 인터페이스 구현
```

**역할:** Open WebUI Pipelines Plugin Framework 통합
- 🔌 Open WebUI Pipe 클래스 구현
- 📊 문서 추가/삭제 API
- 🎛️ 파이프라인 상태 관리

---

## 🌐 **API Servers**

### **web_api_server.py** - OpenAI 호환 API (메인)
**엔드포인트:**
- `GET /v1/models` - 사용 가능한 모델 목록
- `POST /v1/chat/completions` - 채팅 완료 (메인)
- `GET /` - 서버 상태 정보
- `POST /api/documents` - 새 문서 추가

**특징:**
- ✅ OpenAI API 완전 호환
- ✅ Docker 환경 지원
- ✅ CORS 설정 완료
- ✅ 에러 처리 포함

### **pipelines_server.py** - Pipelines 서버 (대안)
**용도:** Open WebUI Pipelines Plugin Framework 전용 서버

---

## 🛠️ **Scripts & Tools**

### **scripts/**
```
scripts/
├── index_documents.py         # PDF 문서 인덱싱 및 벡터화
├── start_webui_integration.py # 전체 시스템 자동 실행
├── test_pipeline.py           # 개별 컴포넌트 테스트
└── start_server.py            # API 서버 단독 실행
```

**주요 스크립트:**
- **index_documents.py**: 새로운 PDF → 벡터 스토어 구축
- **start_webui_integration.py**: 원클릭 전체 시스템 실행
- **test_pipeline.py**: 개별 모듈 테스트

---

## 📊 **Data & Storage**

### **data/**
```
data/
├── *.pdf                     # 소스 PDF 문서들
├── documents/                # 처리된 문서 임시 저장
└── vector_store/             # FAISS 벡터 스토어
    ├── faiss.index           # FAISS 인덱스 파일
    └── documents.pkl         # 문서 메타데이터
```

---

## 📖 **Documentation**

| 파일명 | 용도 | 대상 |
|--------|------|------|
| **README.md** | 프로젝트 전체 개요 | 모든 사용자 |
| **HOW_TO_RUN.md** | 5분 빠른 실행 | 초보자 |
| **INTEGRATION_GUIDE.md** | Open WebUI 통합 완전 가이드 | 개발자 |
| **METHODOLOGY.md** | 다른 프로젝트 적용 방법론 | 고급 사용자 |
| **TROUBLESHOOTING.md** | 문제 해결 및 FAQ | 문제 발생 시 |

---

## 🔄 **데이터 플로우**

### **문서 기반 질문 처리**
```
사용자 질문 → Open WebUI → web_api_server.py → adaptive_rag/router.py
→ adaptive_rag/nodes.py (retrieve) → adaptive_rag/vector_store.py 
→ adaptive_rag/grader.py (relevance) → adaptive_rag/nodes.py (generate)
→ adaptive_rag/grader.py (hallucination) → 최종 응답
```

### **웹 검색 기반 질문 처리**
```
사용자 질문 → Open WebUI → web_api_server.py → adaptive_rag/router.py
→ adaptive_rag/nodes.py (web_search) → Tavily API
→ adaptive_rag/nodes.py (generate) → adaptive_rag/grader.py → 최종 응답
```

### **새 문서 추가 플로우**
```
PDF 파일 → scripts/index_documents.py → document_processing/pdf.py
→ adaptive_rag/vector_store.py → data/vector_store/ 저장
→ 서버 재시작 → 새 문서로 검색 가능
```

---

## 🎯 **각 모듈의 핵심 역할**

| 모듈 | 핵심 역할 | 의존성 |
|------|-----------|--------|
| **adaptive_rag** | 고도화된 RAG 워크플로우 | LangGraph, OpenAI |
| **document_processing** | PDF 처리 및 기본 유틸리티 | LangChain, PDFPlumber |
| **pipelines** | Open WebUI 통합 어댑터 | adaptive_rag, document_processing |
| **scripts** | 자동화 및 유틸리티 | 모든 모듈 |

---

## 💡 **모듈 추가/수정 가이드**

### **새로운 문서 처리기 추가**
```python
# document_processing/new_processor.py
from document_processing.base import RetrievalChain

class NewDocumentProcessor(RetrievalChain):
    def load_documents(self, source_uris):
        # 새로운 문서 타입 처리 로직
        pass
```

### **새로운 RAG 노드 추가**
```python  
# adaptive_rag/new_node.py
def new_processing_node(state: dict) -> dict:
    # 새로운 RAG 처리 단계
    return {"processed_data": result}

# adaptive_rag/graph.py에서 노드 추가
graph.add_node("new_process", new_processing_node)
```

### **새로운 API 엔드포인트 추가**
```python
# web_api_server.py
@app.post("/v1/new-endpoint")
async def new_endpoint(request: CustomRequest):
    # 새로운 기능 구현
    return response
```

---

## 🚀 **확장 포인트**

1. **멀티모달 지원**: 이미지, 음성 처리 모듈 추가
2. **다양한 임베딩**: Sentence Transformers, 로컬 모델 지원
3. **캐싱 레이어**: Redis 기반 결과 캐싱
4. **모니터링**: Prometheus 메트릭, 로깅 개선
5. **A/B 테스팅**: 다양한 RAG 전략 비교

**🎯 이제 프로젝트 구조가 명확하고 확장 가능한 형태로 정리되었습니다!**
