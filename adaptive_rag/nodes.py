"""
Graph Nodes for Adaptive RAG
LangGraph에서 사용되는 각종 노드 함수들
"""

import logging
import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Tavily 기반 실제 웹 검색 도구"""

    def __init__(self):
        """Tavily API를 사용한 웹 검색 초기화"""
        try:
            self.tavily_tool = TavilySearchResults(
                max_results=3,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True,
                include_images=False,
                # API 키는 환경변수 TAVILY_API_KEY에서 자동으로 로드
            )
            logger.info("✅ Tavily 웹 검색 도구가 성공적으로 초기화되었습니다.")
        except Exception as e:
            logger.error(f"❌ Tavily 초기화 실패: {e}")
            # Fallback to mock for development
            self.tavily_tool = None
            logger.warning("Mock 웹 검색으로 대체됩니다.")

    def search(self, query: str, max_results: int = 3) -> List[dict]:
        """실제 웹 검색 수행"""
        logger.info(f"🔍 Tavily 웹 검색 쿼리: {query}")

        if self.tavily_tool is None:
            # Fallback mock results if Tavily is not available
            logger.warning("⚠️ Tavily 사용 불가, Mock 결과 반환")
            return [
                {
                    "content": f"Fallback search result for: {query}",
                    "url": "https://example.com/fallback",
                    "title": f"Fallback Result for {query}",
                }
            ]

        try:
            # Tavily 검색 수행
            results = self.tavily_tool.invoke({"query": query})

            # 결과 포맷팅
            formatted_results = []
            for result in results[:max_results]:
                formatted_result = {
                    "content": result.get("content", ""),
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                }
                formatted_results.append(formatted_result)

            logger.info(f"✅ Tavily에서 {len(formatted_results)}개 결과 반환")
            return formatted_results

        except Exception as e:
            logger.error(f"❌ Tavily 검색 실패: {e}")
            # Return fallback result on error
            return [
                {
                    "content": f"검색 오류 발생. 질문: {query}에 대한 답변을 찾을 수 없습니다.",
                    "url": "https://error.com",
                    "title": f"검색 오류: {query}",
                }
            ]


class RAGNodes:
    """Adaptive RAG를 위한 노드 함수들"""

    def __init__(
        self,
        vector_store=None,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0,
    ):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.web_search_tool = WebSearchTool()

        # RAG 프롬프트 설정
        rag_template = """당신은 전문적인 AI 연구 분석가입니다. 
검색된 문서 정보를 바탕으로 상세하고 구조화된 답변을 제공해주세요.

**답변 가이드라인:**
📋 **구조화된 답변**: 주제별로 명확히 구분하여 설명
🔍 **상세한 분석**: 핵심 내용, 배경, 영향, 의미 등을 포함
📊 **비교 분석**: 여러 국가/기업/정책이 언급된 경우 비교표나 차이점 명시
💡 **실용적 정보**: 구체적인 수치, 날짜, 정책명, 기관명 등 포함
🎯 **결론 및 시사점**: 핵심 요약과 향후 전망 제시

**답변 형식 예시:**
## 📋 핵심 내용
[주요 내용 설명]

## 🔍 상세 분석  
[구체적 분석 내용]

## 📊 비교/특징
[비교 분석 또는 주요 특징]

## 💡 시사점
[의미와 전망]

**사용자 질문이 한국어면 반드시 한국어로, 영어면 영어로 답변해주세요.**

---
**검색된 문서 정보:**
{context}

**질문:** {question}

**전문적이고 상세한 답변:**"""

        self.rag_prompt = ChatPromptTemplate.from_template(rag_template)
        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()

    def retrieve(self, state: dict) -> dict:
        """문서 검색 노드"""
        logger.info("==== [RETRIEVE] ====")
        question = state["question"]

        if not self.vector_store:
            logger.error("Vector store not initialized")
            return {"documents": []}

        try:
            # 문서 검색 수행
            documents = self.vector_store.similarity_search(question, k=10)
            logger.info(f"Retrieved {len(documents)} documents")
            return {"documents": documents}
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            return {"documents": []}

    def generate(self, state: dict) -> dict:
        """답변 생성 노드"""
        logger.info("==== [GENERATE] ====")
        question = state["question"]
        documents = state.get("documents", [])

        if not documents:
            # 문서가 없는 경우 일반적인 답변
            generation = (
                "죄송합니다. 관련 문서를 찾을 수 없어서 답변을 제공할 수 없습니다."
            )
        else:
            try:
                # 문서를 컨텍스트로 변환
                context = self._format_docs(documents)

                # RAG 답변 생성
                generation = self.rag_chain.invoke(
                    {"context": context, "question": question}
                )
            except Exception as e:
                logger.error(f"Error in answer generation: {e}")
                generation = "답변 생성 중 오류가 발생했습니다."

        return {"generation": generation}

    def web_search(self, state: dict) -> dict:
        """웹 검색 노드"""
        logger.info("==== [WEB SEARCH] ====")
        question = state["question"]

        try:
            # 웹 검색 수행
            web_results = self.web_search_tool.search(question, max_results=3)

            # 검색 결과를 Document 객체로 변환
            web_results_docs = [
                Document(
                    page_content=result["content"],
                    metadata={
                        "source": result["url"],
                        "title": result.get("title", ""),
                    },
                )
                for result in web_results
            ]

            logger.info(f"Found {len(web_results_docs)} web search results")
            return {"documents": web_results_docs}

        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return {"documents": []}

    def decide_to_generate(self, state: dict) -> str:
        """문서 관련성 평가 후 생성 여부 결정"""
        logger.info("==== [DECISION TO GENERATE] ====")
        filtered_documents = state.get("documents", [])
        retry_count = state.get("retry_count", 0)

        if not filtered_documents:
            # 재시도 횟수가 3번을 초과한 경우 강제로 답변 생성
            if retry_count >= 3:
                logger.info("==== [DECISION: RETRY LIMIT REACHED, FORCE GENERATE] ====")
                return "generate"
            else:
                # 모든 문서가 관련성 없는 경우 질문 재작성
                logger.info(
                    "==== [DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY] ===="
                )
                return "transform_query"
        else:
            # 관련성 있는 문서가 있는 경우 답변 생성
            logger.info("==== [DECISION: GENERATE] ====")
            return "generate"

    def _format_docs(self, docs: List[Document]) -> str:
        """문서들을 RAG 체인에 사용할 수 있도록 포맷팅"""
        if not docs:
            return ""

        formatted_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content
            formatted_parts.append(f"Document {i+1} (Source: {source}):\n{content}")

        return "\n\n".join(formatted_parts)
