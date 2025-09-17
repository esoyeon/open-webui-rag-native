## Open WebUI + LangGraph RAG 통합 가이드 (Unified, Full)

이 문서는 입문(실습 중심) → 중급(인덱싱/검색) → 운영(세션/캐시/큐)까지 한 번에 제공합니다. 기존 Student/Integration/Technical 가이드를 모두 대체할 수 있도록 충분히 상세하게 구성했습니다.

- 대상: 웹 초심자부터 실무자까지
- 목표: Open WebUI에 LangGraph RAG를 붙이고, 운영형 Enhanced RAG로 확장 및 커스터마이징

---

## 0. 준비물 및 설치

필수
- Python 3.11
- Node.js 20~22 (권장 22.x)
- Redis (macOS: `brew install redis`)
- OpenAI API Key (`OPENAI_API_KEY` 환경변수)

설치 절차
```bash
cd /Users/<you>/projects
git clone <repo>
cd open-webui-rag-native
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# (선택) 프론트엔드 빌드
cd open-webui
npm ci --legacy-peer-deps
npm run build
```
참고: undici 관련 에러가 나면 `open-webui/package.json`의 `undici`가 6.x인지 확인하세요(본 저장소는 6.x 고정).

프로젝트 구조(핵심)
```
enhanced_api_server.py           # 운영형 OpenAI 호환 API 서버(FastAPI)
enhanced_rag/                    # 세션/캐시/큐 등 운영 컴포넌트
pipelines/adaptive_rag_pipeline.py# WebUI 파이프라인 진입점(Pipe)
adaptive_rag/                    # 예시 LangGraph/벡터 스토어
document_processing/             # PDF 등 문서 인덱싱 체인
scripts/                         # 빌드/인덱싱/서버 실행 스크립트
open-webui/                      # Open WebUI 프론트엔드
```

---

## 1단계(입문) — 파이프라인 플러그인으로 빠르게 붙이기

핵심 파일: `pipelines/adaptive_rag_pipeline.py`
- `Pipe` 클래스가 Open WebUI가 로드하는 진입점입니다.
- 여러분의 LangGraph(예: `my_graph.py`)를 만들어 `Pipe`에 연결합니다.

절차
1) 그래프 작성
- 생성자에서 벡터 스토어 주입, `create_graph()` 또는 `run(question: str)` 구현
- 참고 구현: `adaptive_rag/graph.py`의 `AdaptiveRAGGraph`

2) 파이프라인 연결
- `pipelines/adaptive_rag_pipeline.py` 상단에서 import
- `_initialize_pipeline()`에서 임베딩/FAISS 초기화 후 그래프 생성
- `pipe()`에서 `self.rag_graph.run(user_message)` 호출해 문자열 답변 반환

3) 실행
- WebUI 백엔드/프론트 실행 시 파이프 자동 로드
- “Model not found” 발생 시 파이프 초기화(`is_initialized`)와 로그 확인
- LangGraph import 오류는 `adaptive_rag/__init__.py`에서 불필요 import 제거로 해결(본 저장소 적용)

(선택) 문서 인덱싱
```bash
source venv/bin/activate
python scripts/index_documents.py
```
출력물: `data/vector_store/`

---

## 2단계(중급) — 문서 임베딩/검색 붙이기(FAISS 로컬)

목표
- 로컬 PDF를 임베딩해 검색·회수 품질을 체감하고, 프롬프트/답변 개선에 반영

절차
1) 문서 배치: `data/documents/*.pdf`
2) 인덱싱 실행: `python scripts/index_documents.py`
3) 파이프/엔진이 `data/vector_store/`를 로드하는지 확인

실험 아이디어
- 문서 기반 질의 vs 비문서 질의 응답 차이 비교
- top_k, threshold 등의 검색 파라미터 조정

---

## 3단계(운영) — Enhanced RAG API로 확장

핵심 파일: `enhanced_api_server.py`
- OpenAI 호환 API(`/v1/chat/completions`, `/v1/models`) 제공
- 세션 메모리/캐시/태스크 큐 내장

3.1 로컬 기동·헬스
```bash
source venv/bin/activate
python scripts/start_enhanced_system.py
curl -s http://localhost:8000/health | jq .
```
- Redis → RQ → API 순으로 기동. degraded라도 엔드포인트가 응답하면 실습 가능

3.2 WebUI 연동 테스트
```bash
curl -s http://localhost:8000/v1/models | jq .   # id: enhanced-rag
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"enhanced-rag","messages":[{"role":"user","content":"Ping?"}]}' | jq .
```

3.3 세션 메모리 검증(요약/번역 시 초기 대화 포함)
- 날씨→번역→단위차이→요약 흐름 시나리오로 확인
- `enhanced_rag/session_manager.py`의 `max_messages_per_session` 조정

3.4 캐시 체감
- 동일 질문 2회 호출 시간 비교
```bash
curl -s http://localhost:8000/admin/cache/clear | jq .
```

3.5 태스크 큐(RQ)
```bash
curl -s http://localhost:8000/admin/tasks | jq .
```

3.6 안정성/운용 팁
- uvicorn reload 비활성화: `export ENHANCED_API_RELOAD=0`
- 시작 스크립트는 PIPE 블로킹 방지 적용
- API 다운 시 WebUI "Model not found" → API 헬스/로그 확인

3.A 엔진 교체(핵심 확장 포인트)
- `enhanced_api_server.py`의 `initialize_rag_engine()`에서 `OptimizedRAGEngine` 대신 여러분의 엔진을 할당
- 벡터 스토어 초기화/로드 코드는 그대로 재사용 가능

---

## 시스템 아키텍처 및 컴포넌트

요약
```
사용자 → Open WebUI(프론트/백엔드) → RAG API → LangGraph/엔진 → Vector Store → LLM
```

주요 컴포넌트
- Open WebUI Frontend/Backend: UI와 게이트웨이
- RAG API Server: OpenAI 호환 엔드포인트
- LangGraph RAG: 라우팅/검색/생성 워크플로우
- Vector Store: FAISS 임베딩 저장소

API 서버 구조(핵심 파일)
```
web_api_server.py / enhanced_api_server.py    # OpenAI 호환 API
adaptive_rag/graph.py, nodes.py               # LangGraph 워크플로우/노드
adaptive_rag/vector_store.py                  # FAISS 벡터 스토어
pipelines/adaptive_rag_pipeline.py            # WebUI 파이프라인 래퍼
```

---

## 환경변수/설정(요약)

일반
```
OPENAI_API_KEY=...
ENHANCED_API_RELOAD=0         # uvicorn reload 비활성화 권장
```

Open WebUI → 외부 API 연결(예)
```
OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=adaptive-rag-local-key
```

추가(선택)
```
# 벡터/검색/생성 파라미터 (엔진/그래프에서 소비)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
TEMPERATURE=0.7
SIMILARITY_THRESHOLD=0.7
MAX_TOKENS=2000
```

---

## API 명세 개요(OpenAI 호환)

엔드포인트
- POST `/v1/chat/completions`
- GET `/v1/models`
- GET `/health`
- Admin: GET `/admin/tasks`, POST `/admin/cache/clear`

요청 예시
```json
{
  "model": "enhanced-rag",
  "messages": [{"role": "user", "content": "사용자 질문"}],
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false
}
```

응답 예시
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1699999999,
  "model": "enhanced-rag",
  "choices": [
    {"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}
  ],
  "usage": {"prompt_tokens": 50, "completion_tokens": 200, "total_tokens": 250}
}
```

---

## 커스터마이징 가이드(핵심 코드 위치)

프롬프트/파이프라인
- `adaptive_rag/graph.py`, `adaptive_rag/nodes.py`
- 검색 파라미터(top_k/threshold), 생성 파라미터(temperature/max_tokens) 조정

문서 처리 확장
- `document_processing/*`에 새로운 포맷 체인 추가(예: Word, HTML)

벡터 스토어 교체
- `adaptive_rag/vector_store.py`에 Chroma/Pinecone 등 도입 가능

엔진 주입(운영형)
- `enhanced_api_server.py`의 `initialize_rag_engine()`에서 교체

---

## 디버깅/운영 팁

자주 겪는 에러
1) Node 빌드 실패(undici)
- 에러: `Cannot find module '../cache/memory-cache-store'`
- 해결: undici 6.x 고정, `npm ci --legacy-peer-deps`, 재빌드

2) venv pip 경로 오류
- 에러: `pip: bad interpreter`
- 해결: venv 재생성 후 `pip install -r requirements.txt`

3) langgraph import 오류
- 해결: `adaptive_rag/__init__.py`에서 그래프 임포트 제거, 필요한 곳에서 직접 import(본 저장소 적용)

헬스/성능 관찰
- `/health`의 status와 세션/캐시 통계 확인
- 동일 질의 반복 호출로 캐시 효과를 체감
- `/admin/tasks`로 RQ 큐 상태 확인

로그/재시작 안정화
- `ENHANCED_API_RELOAD=0`으로 불필요한 재시작 방지
- 시작 스크립트는 stdout/stderr PIPE 미사용으로 데드락 방지 적용

---

## 데이터 저장소(WebUI)

위치: `data/webui/webui.db`
- 대화(`chat`), 사용자(`user`), 메모리(`memory`), 파일(`file`) 등 저장

빠른 확인(예시)
```bash
sqlite3 data/webui/webui.db ".tables"
sqlite3 data/webui/webui.db "SELECT id, title FROM chat LIMIT 5;"
```

백업/초기화
```bash
cp data/webui/webui.db data/webui/webui_backup_$(date +%Y%m%d).db
rm data/webui/webui.db   # 주의: 초기화
```

---

## 체크리스트
- [ ] `OPENAI_API_KEY` 설정
- [ ] Python 의존성 설치
- [ ] (선택) 프론트 빌드 성공
- [ ] 1단계: 파이프라인에서 그래프 호출 성공
- [ ] 2단계: 문서 인덱싱 후 검색 반영 확인
- [ ] 3단계: Enhanced API 기동 및 `/v1/chat/completions` 응답 확인
- [ ] 요약 시 초기 대화 포함 확인(세션 메모리)
- [ ] 캐시/태스크 큐 동작 체감

---

## 참고
- Open WebUI, LangChain, LangGraph, FAISS, FastAPI 공식 문서
- 본 저장소 주석: `enhanced_api_server.py`, `enhanced_rag/*`, `pipelines/*`, `scripts/*`
