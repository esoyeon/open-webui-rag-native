# test.py
from pipelines.adaptive_rag_pipeline import Pipe

pipeline = Pipe()

# 벡터스토어 질문 (문서 기반)
print("📚 문서 기반 질문:")
response = pipeline.pipe("AI Brief에서 삼성 관련 내용은?", "gpt-3.5-turbo", [], {})
print(response)

# 웹 검색 질문 (최신 정보)
print("\n🌐 웹 검색 질문:")
response = pipeline.pipe("2024년 최신 AI 뉴스는?", "gpt-3.5-turbo", [], {})
print(response)
