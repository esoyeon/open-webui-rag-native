## Open WebUI + LangGraph RAG 통합 가이드 (Unified)

이 문서는 입문자용 실습(학생 가이드)부터 기술 설명, 실제 적용/운영까지 한 번에 제공합니다.

- 대상 독자: 웹을 처음 공부하는 수강생, 실무 적용을 준비하는 개발자
- 목표: Open WebUI에 본인 LangGraph 기반 RAG를 붙이고, 운영형(세션 메모리/Redis/RQ)으로 확장

---

### 0. 준비물
- Python 3.11
- Node.js 20~22 (권장 22.x)
- Redis 서버 (macOS: `brew install redis`)
- OpenAI API Key 환경변수 `OPENAI_API_KEY`

---

### 1. 설치
1) 저장소/가상환경
```bash
cd /Users/<you>/projects
git clone <repo>
cd open-webui-rag-native
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
2) 프론트엔드(선택)
```bash
cd open-webui
npm ci --legacy-peer-deps
npm run build
```
- undici 에러 시 `package.json`의 `undici`가 6.x로 고정되었는지 확인 (본 저장소는 6.x).

---

### 2. 빠른 통합(입문, 파이프라인 플러그인)
핵심: `pipelines/adaptive_rag_pipeline.py` 의 `Pipe` 클래스가 WebUI가 인식하는 진입점.

절차
- 여러분의 그래프(예: `my_graph.py`) 작성: 생성자에서 벡터 스토어, `create_graph()` 또는 `run(question)` 제공
- `pipelines/adaptive_rag_pipeline.py`에서 import → `_initialize_pipeline()`에 주입
- 기본 흐름: OpenAI 임베딩 → FAISS 초기화/로드 → 그래프 호출 → `pipe()`에서 답변 리턴

실행/확인
- Open WebUI 백엔드/프론트엔드 실행 시 자동 로드됨
- “Model not found” 시 파이프 초기화 여부(`is_initialized`)와 로그 확인
- LangGraph import 오류는 루트 `adaptive_rag/__init__.py`에서 불필요한 import 제거로 해결(본 저장소 적용)

문서 인덱싱(선택)
```bash
source venv/bin/activate
python scripts/index_documents.py
```
- 데이터 경로: `data/vector_store/`

---

### 3. 운영형 확장(Enhanced RAG API)
핵심 파일: `enhanced_api_server.py`
- FastAPI 기반 OpenAI 호환 API(`/v1/chat/completions`, `/v1/models`)
- 구성: 세션 메모리(`enhanced_rag/session_manager.py`), 캐시(`enhanced_rag/cache_manager.py`), 태스크 큐(`enhanced_rag/task_queue.py`)

3.1 로컬 기동·헬스
```bash
source venv/bin/activate
python scripts/start_enhanced_system.py
curl -s http://localhost:8000/health | jq .
```
- Redis → RQ → API 순으로 기동, degraded여도 엔드포인트 응답 정상이면 실습 가능

3.2 WebUI 연동 확인
```bash
curl -s http://localhost:8000/v1/models | jq .   # id: enhanced-rag
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"enhanced-rag","messages":[{"role":"user","content":"Ping?"}]}' | jq .
```

3.3 세션 메모리 검증
- 날씨→번역→단위→요약 시나리오로 요약에 초기 대화 포함되는지 확인
- `max_messages_per_session` 변경으로 영향 관찰

3.4 캐시 체감
- 동일 질문 2회 응답 시간 비교
```bash
curl -s http://localhost:8000/admin/cache/clear | jq .
```

3.5 태스크 큐
```bash
curl -s http://localhost:8000/admin/tasks | jq .
```

3.6 안정성/운용 팁
- 불필요한 reload 비활성화: `export ENHANCED_API_RELOAD=0`
- `scripts/start_enhanced_system.py`는 PIPE 블로킹 방지로 안정성 강화
- API 다운 시 WebUI “Model not found” → API 헬스/로그 확인

엔진 교체(핵심 확장 포인트)
- `enhanced_api_server.py` → `initialize_rag_engine()`에서 `OptimizedRAGEngine` 대신 본인 엔진 주입
- 벡터 스토어 초기화 코드는 재사용 가능

---

### 4. 시스템 아키텍처 요약
```
사용자 → Open WebUI(프론트/백엔드) → Enhanced RAG API → LangGraph/엔진 → Vector Store → LLM
```
- WebUI는 UI/대화기록 관리, RAG는 검색/답변 생성에 집중

---

### 5. API 개요 (OpenAI 호환)
- POST `/v1/chat/completions`
- GET `/v1/models`
- GET `/health`
- Admin: GET `/admin/tasks`, POST `/admin/cache/clear`

응답 예시(OpenAI 포맷)
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1699999999,
  "model": "enhanced-rag",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 50, "completion_tokens": 200, "total_tokens": 250}
}
```

---

### 6. 데이터 저장소(WebUI)
- WebUI는 SQLite(`data/webui/webui.db`)에 대화/사용자/메모리/파일 기록
- SQLite CLI나 간단한 Python 코드로 확인/백업/초기화 가능

---

### 7. 자주 겪는 에러와 해결
1) Node 빌드 실패(undici)
- 에러: `Cannot find module '../cache/memory-cache-store'`
- 해결: `undici` 6.x 고정, `npm ci --legacy-peer-deps` 후 빌드

2) venv pip 경로 오류
- 에러: `pip: bad interpreter`
- 해결: 가상환경 재생성, `pip install -r requirements.txt`

3) langgraph import 오류
- 해결: `adaptive_rag/__init__.py`에서 그래프를 임포트하지 말고 필요한 위치에서 직접 import (본 저장소 적용)

---

### 8. 체크리스트
- [ ] `OPENAI_API_KEY` 설정
- [ ] Python 의존성 설치
- [ ] (선택) 프론트 빌드 성공
- [ ] (입문) 파이프라인 `Pipe`로 그래프 호출 확인
- [ ] (중급) 문서 인덱싱→검색 반영 확인
- [ ] (운영) Enhanced API 기동 및 `/v1/chat/completions` 확인
- [ ] 요약 시 초기 대화 포함 확인, 캐시/태스크 큐 동작 체감

---

### 9. 참고
- Open WebUI, LangChain, LangGraph, FAISS, FastAPI 공식 문서
- 본 저장소의 `enhanced_api_server.py`, `enhanced_rag/*`, `pipelines/*`, `scripts/*` 코드 주석(확장 포인트 명시)
