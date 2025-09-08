# 📄 PDF 파일 업데이트 가이드

## 🔄 **PDF 파일 업데이트 시 필요한 작업**

PDF 파일을 변경하거나 추가할 때마다 벡터 스토어를 재구성해야 합니다.

### **⚡ 자동 방법 (권장)**

```bash
# 1단계: 새로운 PDF를 data/ 폴더에 추가

# 2단계: 벡터 스토어 자동 재구성
python scripts/index_documents.py --docs-dir data --force-rebuild

# 3단계: 서버 재시작 (새로운 벡터 스토어 로드)
python scripts/start_webui_integration.py --mode api
```

### **📊 결과 확인**

```bash
# 벡터 스토어 상태 확인
curl -s "http://localhost:8000/" | jq .pipeline_status.total_documents

# 새로운 문서로 테스트
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "adaptive-rag", "messages": [{"role": "user", "content": "문서에서 핵심 내용을 요약해주세요"}]}'
```

---

## 🎯 **효과적인 질문 방법**

### **✅ 추천 질문 방식**

```
❌ "2024년 독일의 AI 정책을 요약해줘"
✅ "문서에서 독일의 AI 정책에 대한 내용을 요약해주세요"

❌ "최신 AI 동향은?"  
✅ "AI Brief 문서에서 다루는 최신 AI 동향을 설명해주세요"

❌ "삼성 AI 전략"
✅ "문서에 나온 삼성전자의 AI 전략을 정리해주세요"
```

### **🎯 핵심 팁**

1. **"문서에서"** 또는 **"문서에 나온"** 키워드 추가
2. **구체적인 질문**으로 작성
3. **PDF 내용과 관련된** 키워드 포함

---

## 🔧 **문제 해결**

### **문제 1: "답변 생성 중 오류 발생"**
**원인**: 웹 검색으로 라우팅되어 Mock 데이터 사용
**해결**: 질문에 "문서에서" 키워드 추가

### **문제 2: "관련 문서를 찾을 수 없습니다"**
**원인**: 벡터 스토어가 업데이트되지 않음
**해결**: 
```bash
python scripts/index_documents.py --docs-dir data --force-rebuild
```

### **문제 3: 서버가 응답하지 않음**
**해결**:
```bash
# 서버 재시작
pkill -f web_api_server
python web_api_server.py &
```

---

## 📁 **PDF 관리 모범 사례**

### **파일 구조**
```
data/
├── 📄 document1.pdf         # 메인 문서
├── 📄 document2.pdf         # 추가 문서  
└── 📁 vector_store/         # 자동 생성된 벡터 스토어
    ├── faiss.index
    └── documents.pkl
```

### **업데이트 프로세스**
1. **PDF 추가**: `data/` 폴더에 새 PDF 복사
2. **벡터화**: `python scripts/index_documents.py --force-rebuild`
3. **확인**: 문서 수 및 질문 테스트
4. **서버 재시작**: 변경사항 적용

### **성능 최적화**
- **문서 수**: 100-500개 권장 (응답 시간 최적화)
- **파일 크기**: PDF당 50MB 이하 권장
- **언어**: 한국어 문서 최적화됨

---

## 🚀 **자동화 스크립트**

### **완전 자동 업데이트**
```bash
#!/bin/bash
# update_docs.sh

echo "🔄 PDF 업데이트 시작..."

# 서버 중지
pkill -f web_api_server

# 벡터 스토어 재구성  
python scripts/index_documents.py --docs-dir data --force-rebuild

# 전체 시스템 재시작
python scripts/start_webui_integration.py --mode api

echo "✅ 업데이트 완료!"
echo "🌐 http://localhost:3000 에서 테스트 가능"
```

### **실행 권한 부여 및 사용**
```bash
chmod +x update_docs.sh
./update_docs.sh
```

---

## 📈 **업데이트 후 테스트 질문들**

```bash
# 1. 기본 연결 테스트
"안녕하세요! 문서에서 주요 내용을 한 문장으로 요약해주세요."

# 2. 구체적 정보 테스트  
"문서에서 다루는 핵심 주제들을 나열해주세요."

# 3. 상세 분석 테스트
"문서에 언급된 [특정 키워드]에 대해 자세히 설명해주세요."
```

---

## 💡 **요약**

**PDF 파일을 업데이트했을 때:**

1. **벡터 스토어 재구성**: `python scripts/index_documents.py --force-rebuild`
2. **서버 재시작**: `python scripts/start_webui_integration.py --mode api` 
3. **테스트 질문**: "문서에서..." 형식으로 질문
4. **확인**: 정상적인 한국어 답변 확인

**🎯 이제 새로운 PDF 내용으로 완벽하게 질문-답변이 가능합니다!**
