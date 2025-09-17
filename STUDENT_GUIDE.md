## 학생용 가이드: Open WebUI에 나만의 LangGraph RAG 붙이기

이 가이드는 웹을 처음 공부하는 수강생이 차근차근 따라 하며, 본인의 LangGraph 기반 RAG를 Open WebUI에 연결하고, 운영 수준(세션 메모리/Redis 큐)으로 확장하는 방법을 설명합니다.

### 1. 사전 준비물
- Python 3.11
- Node.js 20~22 (권장 22.x)
- Redis 서버 (macOS: `brew install redis`)
- OpenAI API Key (환경변수 `OPENAI_API_KEY` 설정)

### 2. 프로젝트 설치
1) 저장소 준비 및 가상환경 생성
```bash
cd /Users/<you>/projects
git clone <repo>
cd open-webui-rag-native
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2) Open WebUI 프론트엔드 의존성 설치 및 빌드
```bash
cd open-webui
npm ci --legacy-peer-deps
npm run build
```
빌드 에러(undici 모듈 관련)가 나면 `package.json`의 `undici`가 6.x로 고정되어 있는지 확인하세요. 본 저장소는 6.x로 고정되어 있습니다.

### 3. 통합 방식 개요
- 통합은 두 가지 트랙 중 하나로 진행합니다.
- 트랙 A(간단): Open WebUI “파이프라인 플러그인”에 여러분의 LangGraph를 꽂아 WebUI 내부에서 동작시킵니다.
- 트랙 B(운영형): 별도 고성능 API 서버(Enhanced RAG)를 띄우고 Open WebUI는 OpenAI 호환 API로 호출합니다. 세션 메모리, Redis 캐시, RQ 워커 등 운영 기능이 포함됩니다.

---

## 트랙 A. 파이프라인 플러그인으로 빠르게 붙이기(추천 입문)

핵심 파일: `pipelines/adaptive_rag_pipeline.py`
- 이 파일의 `Pipe` 클래스가 Open WebUI가 인식하는 진입점입니다.
- 여러분의 LangGraph 구현체를 만들어 여기서 불러오면 됩니다.

1) 나만의 LangGraph 클래스 만들기(예: `my_graph.py`)
- 인터페이스 예시: 생성자에서 벡터 스토어 받기, `create_graph()` 혹은 `run(question: str)` 제공.
- 기존 예시는 `adaptive_rag/graph.py`의 `AdaptiveRAGGraph`를 참고합니다.

2) 파이프라인에 연결하기
- `pipelines/adaptive_rag_pipeline.py` 상단에서 여러분의 그래프를 import하고, `_initialize_pipeline()`에서 인스턴스화하세요.
- 최소한 다음이 동작해야 합니다:
  - OpenAI 임베딩 생성 → FAISS 벡터스토어 초기화/로드
  - 여러분의 그래프에 벡터스토어 전달
  - `pipe()`에서 `self.rag_graph.run(user_message)` 호출해 답변 문자열 리턴

3) 실행
- Open WebUI 백엔드/프론트엔드를 평소처럼 실행하면, 파이프라인이 자동 로드되어 대화 중 사용됩니다.

문제 해결 팁
- “Model not found”가 뜨면 Open WebUI가 대상 모델/파이프를 못 찾는 경우입니다. 파이프라인이 초기화되었는지(`is_initialized`) 로그를 확인하세요.
- LangGraph import 오류가 나면, `__init__.py`에서 불필요한 import를 제거하고 직접 파일에서 import 하세요(본 저장소는 그렇게 수정되어 있음).

---

## 트랙 B. Enhanced RAG 서버로 운영 수준 확장

핵심 파일: `enhanced_api_server.py`
- FastAPI 기반 OpenAI 호환 API(`/v1/chat/completions`, `/v1/models`) 제공
- 구성 요소: 세션 메모리(`enhanced_rag/session_manager.py`), 캐시(`enhanced_rag/cache_manager.py`), 태스크 큐(`enhanced_rag/task_queue.py`)

1) 서버 시작
```bash
source venv/bin/activate
python scripts/start_enhanced_system.py
```
- 내부적으로 Redis 확인 → RQ 워커 → API 서버 순서로 기동합니다.
- 헬스체크: `http://localhost:8000/health`

2) Open WebUI와 연결
- Open WebUI에서 OpenAI 호환 엔드포인트를 `http://localhost:8000`로 등록하거나, 기본 통합을 사용합니다.
- 모델 ID는 `enhanced-rag` 입니다(`/v1/models`에서 확인 가능).

3) 여러분의 LangGraph로 교체하기(핵심 확장 포인트)
- `enhanced_api_server.py` → `initialize_rag_engine()`에서 현재는 간단한 `OptimizedRAGEngine`을 초기화합니다.
- 여러분의 그래프/엔진을 여기에 주입하세요.
  - 예: 벡터 스토어 초기화 부분은 그대로 두고, `OptimizedRAGEngine` 대신 여러분의 엔진 객체를 생성/할당
  - 세션 ID는 `chat_completions()`에서 파생되며, 세션 히스토리는 `SessionManager`가 관리합니다.

4) 운영 기능 개념
- 세션 메모리: 세션별 대화를 Redis에 저장, 요약/번역과 같은 맥락 작업에서 히스토리를 더 넓게 사용
- 캐시: 임베딩/검색/응답 캐시로 속도 향상
- 태스크 큐: 백그라운드 정리 작업(세션/캐시 관리 등)

문제 해결 팁
- “Model not found”: API가 죽거나 모델 목록이 비어 있을 때 발생 → API 헬스 확인, 로그 확인
- DEGRADED 상태: Redis 캐시 경고일 수 있음(기능에는 영향 없을 수 있음) → Redis 상태와 로그 확인
- uvicorn reload로 불필요한 재시작이 잦다면 `ENHANCED_API_RELOAD=0`

---

## 자주 겪는 에러와 해결

1) Node 빌드 실패(undici)
- 증상: `Cannot find module '../cache/memory-cache-store'`
- 해결: `open-webui/package.json`의 `undici`를 6.x로 고정, `npm ci --legacy-peer-deps` 후 `npm run build`

2) Python venv pip 경로 오류
- 증상: `pip: bad interpreter`
- 해결: 가상환경 재생성 후 `pip install -r requirements.txt`

3) langgraph import 오류
- 해결: `adaptive_rag/__init__.py`에서 그래프를 임포트하지 않고, 필요한 위치에서 직접 임포트

---

## 체크리스트
- [ ] OpenAI API 키 설정 완료 (`OPENAI_API_KEY`)
- [ ] Python 의존성 설치 완료
- [ ] (선택) 프론트엔드 빌드 성공
- [ ] 트랙 A: 파이프라인에서 그래프 호출 성공
- [ ] 트랙 B: Enhanced API 기동 및 `/v1/chat/completions` 응답 확인
- [ ] 요약 요청 시 초기 대화가 반영되는지 확인

---

## 다음 과제(선택)
- 벡터 스토어에 본인 문서를 추가하고 정확도 비교
- 세션 메모리를 활용한 대화형 요약 UX 만들기
- 웹 검색과 벡터 검색을 하이브리드로 결합해 라우팅 개선


