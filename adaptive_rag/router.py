"""
Query Router for Adaptive RAG
사용자 쿼리를 가장 관련성 높은 데이터 소스로 라우팅
"""

import logging
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class RouteQuery(BaseModel):
    """사용자 쿼리를 가장 관련성 높은 데이터 소스로 라우팅하는 데이터 모델"""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


class QueryRouter:
    """쿼리 라우터 클래스"""

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.structured_llm_router = self.llm.with_structured_output(RouteQuery)

        # 시스템 메시지와 사용자 질문을 포함한 프롬프트 템플릿 생성
        system_message = """당신은 사용자 질문을 벡터스토어 검색 또는 웹 검색으로 라우팅하는 전문가입니다.

🔍 **벡터스토어**: 2024년 국내외 AI 산업 동향 연구 문서가 포함되어 있습니다.
- 국가별 AI 정책 (한국, 미국, 중국, 독일, 일본, 영국 등)
- AI 산업 동향 및 기술 발전 현황
- AI 관련 정책, 투자, 기업 전략
- 2024년 기준 AI 생태계 분석

🌐 **웹 검색**: 실시간 정보나 문서에 없는 최신 정보
- 오늘/어제 발생한 뉴스
- 실시간 주가, 환율, 날씨
- 문서 발행 이후의 최신 업데이트

**라우팅 가이드라인:**
1. **AI 정책, AI 산업, AI 기술, 국가별 AI 전략** → vectorstore
2. **2024년 AI 관련 정보** → vectorstore (문서가 2024년 기준)  
3. **"오늘", "어제", "최신 뉴스", "실시간"** → web_search
4. **기업명, 정책명, 기술명** 관련 → vectorstore 우선
5. **확실하지 않은 경우** → vectorstore 우선 선택"""

        self.route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", "{question}"),
            ]
        )

        # 프롬프트 템플릿과 구조화된 LLM 라우터를 결합
        self.question_router = self.route_prompt | self.structured_llm_router

    def route(self, question: str) -> str:
        """질문을 분석하여 적절한 데이터 소스로 라우팅"""
        try:
            result = self.question_router.invoke({"question": question})
            datasource = result.datasource

            logger.info(f"Routed question to: {datasource}")
            logger.debug(f"Question: {question}")

            return datasource

        except Exception as e:
            logger.error(f"Error in query routing: {e}")
            # 기본적으로 vectorstore로 라우팅
            return "vectorstore"

    def route_question_node(self, state: dict) -> str:
        """LangGraph 노드용 라우팅 함수"""
        logger.info("==== [ROUTE QUESTION] ====")
        question = state["question"]

        source = self.route(question)

        if source == "web_search":
            logger.info("==== [ROUTE QUESTION TO WEB SEARCH] ====")
            return "web_search"
        elif source == "vectorstore":
            logger.info("==== [ROUTE QUESTION TO VECTORSTORE] ====")
            return "vectorstore"
        else:
            # 기본값
            return "vectorstore"
