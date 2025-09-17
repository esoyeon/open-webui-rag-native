> 이 문서는 통합 가이드로 대체되었습니다. 최신 문서는 `GUIDE.md`를 참고하세요.

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

---

## 1단계(입문) — 파이프라인 플러그인으로 빠르게 붙이기

핵심 파일: `pipelines/adaptive_rag_pipeline.py`
- `Pipe` 클래스가 Open WebUI가 인식하는 진입점입니다.
- 여러분의 LangGraph 구현체를 만들어 여기서 불러옵니다.

절차
1) 그래프 클래스 만들기(예: `my_graph.py`)
- 인터페이스 예시: 생성자에서 벡터 스토어 받기, `create_graph()` 또는 `run(question: str)` 제공
- 참고: `adaptive_rag/graph.py`의 `AdaptiveRAGGraph`

2) 파이프라인 연결
- `pipelines/adaptive_rag_pipeline.py` 상단에서 import 후, `_initialize_pipeline()`에서 인스턴스화
- 최소 동작: OpenAI 임베딩 → FAISS 초기화/로드 → 그래프에 전달 → `pipe()`에서 `run()` 호출

3) 실행
- Open WebUI 백엔드/프론트엔드 실행 시 파이프가 자동 로드됩니다.

문제 해결
- “Model not found”: 파이프 초기화 여부(`is_initialized`) 로그 확인
- LangGraph import 오류: `adaptive_rag/__init__.py`에서 직접 import로 우회(본 저장소 적용)

---

## 2단계(중급) — 문서 임베딩/검색 붙이기(FAISS 로컬)

목표
- 로컬 문서를 FAISS에 인덱싱해서 검색 품질을 체감합니다.

절차
1) 문서 준비: `data/documents/*.pdf` 추가
2) 인덱싱 실행:
```bash
source venv/bin/activate
python scripts/index_documents.py
```
3) 파이프라인/엔진이 벡터 스토어를 로드하도록 확인
- 경로: `data/vector_store/`

실험
- 문서 내용 관련 질문 → 답변/근거 비교

---

## 3단계(운영형) — Enhanced RAG 서버로 확장

핵심 파일: `enhanced_api_server.py`
- FastAPI 기반 OpenAI 호환 API(`/v1/chat/completions`, `/v1/models`)
- 구성 요소: 세션 메모리, 캐시, 태스크 큐

### 3.1 로컬 기동·헬스 확인
```bash
source venv/bin/activate
python scripts/start_enhanced_system.py
# Health
curl -s http://localhost:8000/health | jq .
```
- Redis → RQ 워커 → API 서버 순으로 기동
- status가 degraded여도 엔드포인트 응답이 정상이면 실습가능

### 3.2 Open WebUI 연결 확인
```bash
curl -s http://localhost:8000/v1/models | jq .
# 모델 id: enhanced-rag
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"enhanced-rag","messages":[{"role":"user","content":"Ping?"}]}' | jq .
```

### 3.3 세션 메모리 이해·검증
- 대화: 날씨→번역→단위차이→요약 흐름을 보내고, 요약에 초기 대화가 포함되는지 확인
- `enhanced_rag/session_manager.py`의 `max_messages_per_session` 값을 조정해 영향 관찰

### 3.4 캐시 체감
- 동일 질문 2회 호출 시간 비교 → 캐시 효과 확인
- 캐시 비우기:
```bash
curl -s http://localhost:8000/admin/cache/clear | jq .
```

### 3.5 태스크 큐 실습
- RQ 워커 로그 확인(터미널)
- 태스크 상태 확인:
```bash
curl -s http://localhost:8000/admin/tasks | jq .
```

### 3.6 안정성/운용 팁
- 재시작 불필요한 reload 비활성화:
```bash
export ENHANCED_API_RELOAD=0
```
- `scripts/start_enhanced_system.py`는 PIPE 블로킹 방지로 안정성 향상 적용됨
- API 다운 시 Open WebUI의 “Model not found” 재현/원인 파악 훈련

선택 심화
- 엔진 교체: `enhanced_api_server.py` → `initialize_rag_engine()`에서 여러분의 엔진 주입
- 벡터 스토어 영속화: 문서 추가→저장→로드 실습
- 웹 검색 토글/비교(정확도/지연)
- 배포 맛보기(Docker는 수업 브랜치 기준 선택)

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
- [ ] 1단계: 파이프라인에서 그래프 호출 성공
- [ ] 2단계: 문서 인덱싱 후 검색 반영 확인
- [ ] 3단계: Enhanced API 기동 및 `/v1/chat/completions` 응답 확인
- [ ] 3.3: 요약 요청 시 초기 대화가 반영되는지 확인
- [ ] 3.4~3.5: 캐시/태스크 큐 동작 체감

---

## 다음 과제(선택)
- 벡터 스토어에 본인 문서를 추가하고 정확도 비교
- 세션 메모리를 활용한 대화형 요약 UX 만들기
- 웹 검색과 벡터 검색을 하이브리드로 결합해 라우팅 개선


