# Open WebUI + Adaptive RAG Integration (Docker-Free)

**도커 없이 Open WebUI와 Adaptive RAG 시스템을 연동하는 프로젝트**

이 프로젝트는 Open WebUI를 활용하여 자체 구현한 RAG(Retrieval-Augmented Generation) 시스템과 연동하는 방법을 학습할 수 있도록 구성되었습니다. 도커 없이 직접 환경을 구성하여 시스템의 동작 원리를 더 깊이 이해할 수 있습니다.

> **🔗 관련 프로젝트**: [Docker 버전](https://github.com/esoyeon/llm_chatbot) - Docker를 사용한 간편한 설정

## 🚀 빠른 시작 가이드 (Quick Start)

**전제 조건**: Python 3.9+ 와 Node.js 18+ 가 설치되어 있어야 합니다.

```bash
# 1. 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 환경 변수 설정 (.env 파일 생성)
# OPENAI_API_KEY=sk-your-api-key
# ENABLE_EVALUATION_ARENA_MODELS=False (Arena Model 제거)

# 4. 프론트엔드 빌드 (첫 실행시만, 5-10분 소요)
python scripts/build_frontend.py

# 5. 문서 인덱싱
python scripts/index_documents.py

# 6. 서버 실행 (2개 터미널 필요)
# 터미널1: python scripts/start_rag_server.py
# 터미널2: python scripts/start_webui.py

# 7. 브라우저에서 http://localhost:8080 접속
```

## 📋 주요 특징

- 🚫 **도커 없이 실행**: 복잡한 컨테이너 설정 없이 직접 환경 구성
- 🔗 **Open WebUI 연동**: 자체 RAG API를 Open WebUI에 연결
- 📚 **Adaptive RAG**: LangGraph 기반의 지능형 문서 검색
- 📖 **교육용 설계**: 학습자가 직접 환경을 구성하며 시스템 동작 원리 이해
- 🌐 **크로스 플랫폼**: Windows, macOS, Linux에서 모두 실행 가능
- ⚡ **빠른 시작**: Docker 설치 없이 바로 시작 가능

## 🛠 시스템 요구사항

- **운영체제**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+ (또는 동등한 Linux 배포판)
- **Python**: 3.9 이상
- **Node.js**: 18 이상 (프론트엔드 빌드용)
- **메모리**: 8GB 이상 권장
- **디스크**: 5GB 이상 여유공간

## 📦 환경 설정 가이드

### 1단계: Python 설치 확인

터미널(명령 프롬프트 또는 PowerShell)을 열고 Python 버전을 확인하세요:

```bash
python --version
```

Python 3.9 이상이 설치되어 있어야 합니다. 없다면 [python.org](https://python.org)에서 설치하세요.

### 2단계: 프로젝트 다운로드

```bash
git clone https://github.com/esoyeon/open-webui-rag-native.git
cd open-webui-rag-native
```

### 3단계: 가상환경 생성 및 활성화

**Windows (명령 프롬프트):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

가상환경이 활성화되면 터미널 앞에 `(venv)`가 표시됩니다.

### 4단계: 패키지 설치

프로젝트에 필요한 패키지들을 설치하세요:

```bash
pip install -r requirements.txt
```

**설치되는 주요 패키지들:**
- `langchain`, `langgraph`: RAG 시스템 구축
- `faiss-cpu`: 벡터 검색 엔진
- `fastapi`, `uvicorn`: API 서버 구축
- `pdfplumber`: PDF 문서 처리
- `openai`: OpenAI API 연동

### 5단계: 환경 변수 설정 (.env 파일)

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 다음 내용을 입력하세요(env_example.txt 파일 참고):

```env
# OpenAI API 설정 (필수)
OPENAI_API_KEY=your_openai_api_key_here

# 서버 설정 (기본값 사용 시 수정하지 않아도 됨)
RAG_SERVER_HOST=127.0.0.1
RAG_SERVER_PORT=8000

# Open WebUI 설정
WEBUI_HOST=127.0.0.1
WEBUI_PORT=8080

# 벡터 스토어 설정
VECTOR_STORE_PATH=data/vector_store
DOCUMENTS_PATH=data/documents

# Open WebUI 모델 설정 (RAG 시스템만 사용)
ENABLE_EVALUATION_ARENA_MODELS=False
ENABLE_OLLAMA_API=False
ENABLE_OPENAI_API=True

# Open WebUI 자체 RAG 기능 비활성화 (외부 RAG API 사용)
ENABLE_RAG=False
ENABLE_RAG_HYBRID_SEARCH=False
ENABLE_RAG_WEB_LOADER=False
```

⚠️ **중요**: `your_openai_api_key_here`를 실제 OpenAI API 키로 교체하세요.

### 6단계: 디렉토리 생성

필요한 디렉토리들을 만들어주세요:

**macOS/Linux:**
```bash
mkdir -p data/documents
mkdir -p data/vector_store
mkdir -p logs
```

**Windows:**
```cmd
mkdir data\documents
mkdir data\vector_store  
mkdir logs
```

### 7단계: 문서 추가

`data/documents/` 폴더에 학습시키고 싶은 PDF 문서들을 복사하세요.

예시:
- AI 관련 논문 PDF
- 회사 문서 PDF  
- 교육 자료 PDF

### 8단계: 문서 인덱싱

문서들을 벡터 데이터베이스에 저장하세요:

```bash
python scripts/index_documents.py
```

성공하면 `data/vector_store/` 폴더에 `faiss.index`와 `documents.pkl` 파일이 생성됩니다.

### 9단계: Open WebUI 프론트엔드 빌드 (중요!)

Open WebUI는 백엔드(Python/FastAPI)와 프론트엔드(Svelte/JavaScript)로 구성되어 있습니다.
웹 인터페이스를 사용하려면 프론트엔드를 빌드해야 합니다.

#### 🔍 이해하기: Python venv vs Node.js 환경

**두 환경은 독립적으로 작동합니다:**

| 구분 | Python 백엔드 | JavaScript 프론트엔드 |
|------|--------------|---------------------|
| **환경** | venv (가상환경) | Node.js (전역 설치) |
| **패키지 관리** | pip, requirements.txt | npm, package.json |
| **패키지 위치** | venv/lib/python3.x/ | node_modules/ |
| **실행 시점** | 서버 실행 중 계속 필요 | 빌드 시에만 필요 |
| **결과물** | Python 코드 실행 | HTML/CSS/JS 정적 파일 생성 |

**실제 사용 예시:**
```bash
# Python 환경 (venv 활성화된 상태)
(venv) > pip install fastapi  # Python 패키지 설치
(venv) > npm install svelte    # JavaScript 패키지 설치 (venv와 무관!)
```

💡 **핵심**: venv는 Python만을 위한 것이고, Node.js는 별도로 작동합니다.
터미널에 `(venv)`가 표시되어도 npm 명령어는 정상적으로 작동합니다.

#### 9-1. Node.js 설치 확인

**왜 필요한가?**
- Open WebUI의 프론트엔드는 JavaScript 기반(Svelte)으로 개발됨
- Node.js는 JavaScript 코드를 브라우저가 이해할 수 있는 형태로 변환(빌드)하는데 필요

```bash
node --version
npm --version
```

Node.js가 없다면 [nodejs.org](https://nodejs.org)에서 LTS 버전을 다운로드하여 설치하세요.

#### 9-2. 프론트엔드 빌드 실행

```bash
python scripts/build_frontend.py
```

**이 스크립트가 하는 일:**
1. Node.js 설치 확인
2. JavaScript 패키지 설치 (`npm install`)
3. 프론트엔드 소스코드를 정적 파일로 빌드 (`npm run build`)
4. 빌드 결과를 `open-webui/build/` 폴더에 저장

**소요 시간**: 첫 실행 시 5-10분 (패키지 다운로드 및 빌드)

⚠️ **주의사항**:
- 인터넷 연결이 필요합니다 (npm 패키지 다운로드)
- 디스크 공간 약 500MB 필요 (node_modules 폴더)
- Windows에서 관리자 권한이 필요할 수 있습니다

## 🚀 실행 방법

두 개의 터미널을 사용해서 서버들을 실행해야 합니다.

### 터미널 1: RAG API 서버 시작

```bash
# 가상환경이 활성화되어 있는지 확인
python scripts/start_rag_server.py
```

성공하면 다음과 같은 메시지가 나타납니다:
```
🚀 RAG API 서버를 시작합니다...
📍 주소: http://127.0.0.1:8000
📖 API 문서: http://127.0.0.1:8000/docs
```

### 터미널 2: Open WebUI 서버 시작

```bash  
# 새로운 터미널을 열고 가상환경 활성화
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

python scripts/start_webui.py
```

성공하면 다음과 같은 메시지가 나타납니다:
```
🚀 Open WebUI를 시작합니다...
📍 주소: http://127.0.0.1:8080
```

## 🌐 사용 방법

### 웹 인터페이스 접속

1. 웹 브라우저에서 **http://localhost:8080** 접속
2. 첫 방문 시 계정 생성 (이메일과 패스워드 입력)
3. 로그인 완료 후 채팅 인터페이스 확인

### RAG 시스템 테스트

채팅창에서 다음과 같은 질문들을 해보세요:

**일반적인 질문:**
- "안녕하세요!"
- "이 시스템은 무엇인가요?"

**문서 기반 질문 (PDF 내용에 따라):**
- "이 문서의 주요 내용을 요약해주세요"
- "AI 동향에 대해 알려주세요"
- "문서에서 언급된 기술들은 무엇인가요?"

### 시스템 동작 방식

**준비 단계 (최초 1회):**
1. **프론트엔드 빌드** → JavaScript 코드를 HTML/CSS/JS 파일로 변환
2. **문서 인덱싱** → PDF를 벡터 데이터베이스에 저장

**실행 단계 (매번):**
1. **질문 입력** → Open WebUI 인터페이스 (브라우저)
2. **요청 전달** → Open WebUI 백엔드 → RAG API 서버 (포트 8000)
3. **문서 검색** → FAISS 벡터 스토어에서 관련 문서 찾기
4. **답변 생성** → OpenAI GPT를 사용해 문서 기반 답변 생성
5. **응답 표시** → Open WebUI에서 최종 답변 표시

## 📁 프로젝트 구조

```
open-webui-rag-native/
├── 📘 Python 백엔드 영역
│   ├── adaptive_rag/          # RAG 핵심 로직 (Python)
│   ├── document_processing/   # 문서 처리 모듈 (Python)
│   ├── pipelines/            # RAG 파이프라인 (Python)
│   ├── scripts/              # 실행 스크립트들 (Python)
│   │   ├── index_documents.py   # 문서 인덱싱
│   │   ├── build_frontend.py    # 프론트엔드 빌드 실행
│   │   ├── start_rag_server.py  # RAG 서버 시작
│   │   └── start_webui.py      # WebUI 서버 시작
│   ├── venv/                 # Python 가상환경 (생성 후)
│   └── requirements.txt      # Python 패키지 목록
│
├── 📗 JavaScript 프론트엔드 영역
│   └── open-webui/           
│       ├── src/              # Svelte 소스코드 (JavaScript)
│       ├── package.json      # JavaScript 패키지 목록
│       ├── node_modules/     # JavaScript 패키지들 (설치 후)
│       ├── build/            # 빌드된 정적 파일 (빌드 후)
│       └── backend/          # Open WebUI 백엔드 (Python)
│
├── 💾 데이터 영역
│   └── data/
│       ├── documents/        # PDF 문서 저장
│       └── vector_store/     # 벡터 데이터베이스
│
└── ⚙️ 설정 파일
    ├── .env                  # 환경 변수 (직접 생성)
    └── .gitignore           # Git 제외 목록
```

## 🔧 구성 요소

### Adaptive RAG 시스템
- **Router**: 질문 유형 분석 및 라우팅
- **Retriever**: 관련 문서 검색
- **Grader**: 문서 관련성 평가
- **Rewriter**: 질문 재작성
- **Generator**: 최종 답변 생성

### Open WebUI 연동
- OpenAI 호환 API 엔드포인트 제공
- 실시간 채팅 인터페이스
- 사용자 관리 시스템

## 💾 데이터 관리

Open WebUI는 사용자 데이터를 SQLite 데이터베이스에 저장합니다. 자세한 내용은 [기술 가이드](TECHNICAL_GUIDE.md#데이터-저장소-open-webui)를 참조하세요.


## 🔍 API 엔드포인트

RAG 서버가 실행되면 다음 엔드포인트를 사용할 수 있습니다:

- `GET /health`: 서버 상태 확인
- `POST /v1/chat/completions`: OpenAI 호환 채팅 API
- `GET /docs`: API 문서 (Swagger UI)

## ⚠️ 문제 해결

### 환경 설정 문제

**Q: `python --version`이 작동하지 않아요**
- Windows에서 Python을 Microsoft Store에서 설치했다면 `python3` 명령어 사용
- PATH 환경변수에 Python이 등록되어 있는지 확인
- 명령 프롬프트를 관리자 권한으로 실행해보기

**Q: 가상환경 활성화가 안 돼요**
- Windows PowerShell에서 실행 정책 오류가 나면:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
- 그 후 다시 `venv\Scripts\Activate.ps1` 실행

**Q: 패키지 설치가 실패해요**
- 가상환경이 활성화되어 있는지 확인 (`(venv)` 표시)
- pip 업데이트: `python -m pip install --upgrade pip`
- 전체 재설치: `pip install -r requirements.txt --force-reinstall`
- 특정 패키지 오류 시 개별 설치: `pip install 패키지명`

### 프론트엔드 빌드 문제

**Q: Node.js/npm이 인식되지 않아요**
- Node.js 설치 후 터미널을 완전히 닫고 새로 열어보세요
- Windows: 환경변수 PATH에 Node.js가 추가되었는지 확인
- `where node` (Windows) 또는 `which node` (macOS/Linux)로 경로 확인

**Q: npm install이 실패해요**
- 관리자 권한으로 터미널 실행 (Windows)
- 캐시 정리: `npm cache clean --force`
- 프록시 환경이라면: `npm config set proxy http://proxy-server:port`

**Q: 빌드가 메모리 부족으로 실패해요**
- 다른 프로그램을 종료하여 메모리 확보
- Node.js 메모리 늘리기: `set NODE_OPTIONS=--max-old-space-size=4096` (Windows)

**Q: "Cannot find module" 오류가 나요**
- `node_modules` 폴더 삭제 후 재설치:
  ```bash
  cd open-webui
  rm -rf node_modules  # Windows: rmdir /s node_modules
  npm install --force
  ```

### 서버 실행 문제

**Q: RAG 서버가 시작되지 않아요**
- `.env` 파일의 `OPENAI_API_KEY`가 올바르게 설정되었는지 확인
- API 키 형식: `sk-...`로 시작하는 문자열
- 포트 8000이 다른 프로그램에서 사용 중인지 확인

**Q: Open WebUI 서버가 시작되지 않아요**
- RAG 서버(포트 8000)가 먼저 실행되어 있는지 확인
- 포트 8080이 사용 가능한지 확인
- Open WebUI 의존성이 설치되었는지 확인

**Q: 웹 페이지에 접속이 안 돼요**
- 서버 시작 메시지에서 올바른 주소 확인
- 방화벽이 포트를 차단하지 않는지 확인
- 다른 브라우저로 시도해보기

### 모델 및 UI 문제

**Q: Arena Model이나 불필요한 모델들이 보여요**
- `.env` 파일에 다음 설정 추가:
  ```env
  ENABLE_EVALUATION_ARENA_MODELS=False
  ENABLE_OLLAMA_API=False
  ```
- Open WebUI 서버를 재시작해주세요
- 브라우저 캐시를 삭제하고 페이지를 새로고침해주세요

**Q: 모델 선택에서 RAG 모델만 보이게 하고 싶어요**
- 위의 Arena Model 제거 설정 적용
- RAG API 서버가 올바르게 실행 중인지 확인
- Open WebUI 설정에서 다른 AI 제공자들을 비활성화

**Q: 로그에 "VECTOR_DB: chroma"가 나타나요**
- 이는 Open WebUI의 자체 RAG 기능이 활성화되어 있기 때문입니다
- `.env` 파일에 `ENABLE_RAG=False` 설정을 추가하세요
- Open WebUI 서버를 재시작하면 사라집니다

### 문서 및 RAG 문제

**Q: 문서 인덱싱이 실패해요**
- `data/documents/` 폴더에 PDF 파일이 있는지 확인
- PDF 파일이 손상되지 않았는지 확인
- OpenAI API 키와 인터넷 연결 상태 확인

**Q: 질문에 답변하지 못해요**
- 벡터 스토어가 생성되었는지 확인 (`data/vector_store/faiss.index` 파일 존재)
- 업로드한 문서와 관련된 질문인지 확인
- RAG 서버 로그에서 오류 메시지 확인

**Q: "문서에서 찾을 수 없습니다"라고 답해요**
- 질문이 업로드한 문서 내용과 관련이 있는지 확인
- 더 구체적이고 명확한 질문으로 다시 시도
- 문서에 해당 내용이 실제로 포함되어 있는지 확인

### 디버깅 팁

**서버 상태 확인:**
```bash
# RAG API 서버 동작 확인
curl http://localhost:8000/health

# API 문서 확인
# 브라우저에서 http://localhost:8000/docs 접속
```

**로그 확인:**
- RAG 서버: 터미널에서 오류 메시지 확인
- Open WebUI: 브라우저 개발자 도구(F12) 콘솔 확인


## 🔧 확장 가능성

이 기본 구조를 바탕으로 다음과 같은 기능들을 추가해볼 수 있습니다:

- **다른 문서 형식 지원**: Word, Excel, PowerPoint 등
- **다양한 벡터 DB**: Chroma, Pinecone, Weaviate 등
- **멀티모달 RAG**: 이미지, 음성 포함 문서 처리
- **사용자별 문서 관리**: 개인화된 지식베이스
- **실시간 문서 업데이트**: 문서 변경 시 자동 재인덱싱

## 🆚 Docker vs Docker-Free 비교

| 구분 | Docker-Free 버전 (현재) | Docker 버전 |
|------|------------------------|-------------|
| **설정 복잡도** | ⭐⭐⭐⭐ (복잡) | ⭐⭐ (간단) |
| **환경 일관성** | ⭐⭐⭐ (좋음) | ⭐⭐⭐⭐⭐ (완벽) |
| **학습 효과** | ⭐⭐⭐⭐⭐ (높음) | ⭐⭐ (낮음) |
| **운영 환경** | ⭐⭐⭐ (좋음) | ⭐⭐⭐⭐⭐ (완벽) |
| **디버깅** | ⭐⭐⭐⭐ (쉬움) | ⭐⭐ (어려움) |
| **커스터마이징** | ⭐⭐⭐⭐⭐ (쉬움) | ⭐⭐⭐ (보통) |

**🔧 Docker-Free 버전 (현재)**: 시스템 동작 원리를 학습하고 커스터마이징이 필요한 경우
**🐳 Docker 버전**: 빠른 시작과 안정적인 운영이 필요한 경우

## 📚 추가 학습 자료

- [LangChain 공식 문서](https://python.langchain.com/)
- [LangGraph 튜토리얼](https://langchain-ai.github.io/langgraph/)
- [Open WebUI 문서](https://docs.openwebui.com/)
- [FAISS 가이드](https://faiss.ai/)
- [FastAPI 튜토리얼](https://fastapi.tiangolo.com/tutorial/)

## 📄 라이선스

Open WebUI는 원본 라이선스를 따릅니다.