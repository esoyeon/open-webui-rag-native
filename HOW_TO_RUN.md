# 🚀 빠른 실행 가이드

> **5분 안에 Open WebUI + Adaptive RAG 시스템 실행하기**

## 📋 **실행 전 준비사항**

- [x] Python 3.11+ 설치
- [x] Docker 설치 및 실행  
- [x] OpenAI API 키 준비
- [x] Tavily API 키 준비 (웹 검색용)

---

## ⚡ **원클릭 실행** (권장)

```bash
# 1. 저장소 클론 후 이동
cd llm_chatbot

# 2. 가상환경 설정
uv venv && source .venv/bin/activate && uv pip install -e .

# 3. API 키 설정
cp env_example.txt .env
# .env 파일에서 OPENAI_API_KEY, TAVILY_API_KEY 설정

# 4. 전체 시스템 자동 실행
python scripts/start_webui_integration.py --mode api

# 5. 브라우저에서 접속: http://localhost:3000
# 6. 백엔드 설정: http://host.docker.internal:8000/v1
```

**⏱️ 예상 시간: 3-5분**

---

## 🎯 **브라우저에서 최종 설정**

**Open WebUI 접속 후:**
1. **Admin Panel** (우측 상단 ⚙️ 아이콘)
2. **Settings** → **Connections** → **OpenAI API**
3. 설정 입력:
   - **API Base URL**: `http://host.docker.internal:8000/v1`
   - **API Key**: `sk-dummy-key` (아무 값)
4. **Save** → **새로고침**

---

## 🧪 **테스트 질문 예시**

### **📚 문서 기반 질문** (FAISS 벡터 검색)
```
문서에서 독일과 한국의 AI 정책을 비교 분석해주세요
문서에 나온 2024년 AI 산업 동향의 핵심 내용은?
문서에서 삼성전자의 AI 전략에 대해 자세히 설명해주세요
```

### **🌐 실시간 정보 질문** (Tavily 웹 검색)
```
오늘 AI 관련 최신 뉴스를 알려주세요
현재 ChatGPT 최신 업데이트는 무엇인가요?
최근 한국 AI 기업들의 주요 소식은?
```

### **🎯 예상 응답 형식**
```
## 📋 핵심 내용
[주요 내용 요약]

## 🔍 상세 분석  
[구체적 분석 내용]

## 📊 비교/특징
[비교 분석 결과]

## 💡 시사점
[결론 및 전망]
```

---

## 🛠️ **빠른 문제 해결**

### **❌ "Connection Failed"**
```bash
# API 서버 상태 확인
curl http://localhost:8000/v1/models
# 응답: {"data":[{"id":"adaptive-rag",...}]}
```

### **❌ "모델이 선택되지 않았다"**
- 백엔드 URL 확인: `http://host.docker.internal:8000/v1`
- 브라우저 새로고침 (Ctrl+F5)

### **❌ "답변 생성 중 오류"**
```bash
# .env 파일의 API 키 확인
cat .env | grep OPENAI_API_KEY
```

### **💡 더 많은 해결책**
상세한 문제 해결은 **[TROUBLESHOOTING.md](./TROUBLESHOOTING.md)** 참조

---

## 📊 **시스템 상태 확인**

```bash
# 전체 시스템 진단 (권장)
python scripts/diagnose_system.py

# API 종합 테스트
python scripts/test_api.py

# 개별 확인
curl http://localhost:8000/v1/models    # API 서버
docker ps | grep open-webui             # WebUI 컨테이너  
ls data/vector_store/                   # 벡터 스토어 파일
```

---

## 📄 **PDF 문서 업데이트**

새로운 PDF를 추가했을 때:

```bash
# 1. PDF 파일을 data/ 폴더에 추가
cp new-document.pdf data/

# 2. 벡터 스토어 재구성
python scripts/index_documents.py --docs-dir data --force-rebuild

# 3. 서버 재시작
python scripts/start_webui_integration.py --mode api
```

**상세 가이드**: **[PDF_UPDATE_GUIDE.md](./PDF_UPDATE_GUIDE.md)**

---

## 🎉 **완성된 기능들**

✅ **고도화된 RAG**: 자가 수정, 환상 검증, 품질 보장  
✅ **하이브리드 검색**: 문서 검색 + 실시간 웹 검색  
✅ **구조화된 답변**: 4단계 구조로 상세하고 체계적인 응답  
✅ **한국어 최적화**: 모든 프롬프트와 응답 한국어 특화  
✅ **비교 분석**: 국가별, 기업별, 정책별 체계적 비교  
✅ **OpenAI 호환**: 기존 도구들과 완전 호환  

---

## 🚀 **다음 단계**

- **📋 [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md)**: 다른 시스템과 통합 방법
- **🏗️ [METHODOLOGY.md](./METHODOLOGY.md)**: 다른 프로젝트에 적용하는 방법론
- **🛠️ [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)**: 상세한 문제 해결 가이드

**🎯 5분만 투자하면 완전한 AI 어시스턴트를 경험할 수 있습니다!**