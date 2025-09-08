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
벡터스토어에는 업로드된 지식 베이스의 다양한 주제에 관련된 문서들이 포함되어 있습니다.
업로드된 문서나 지식 베이스에 대한 질문은 vectorstore를 사용하세요.
현재 이벤트, 실시간 정보, 또는 지식 베이스에 포함되지 않은 주제에 대한 질문은 web_search를 사용하세요.

가이드라인:
- 업로드된 문서에 있을 가능성이 높은 일반적인 지식에 대한 질문이면 vectorstore로 라우팅
- 현재/최근 정보(오늘, 올해, 최신 뉴스)를 요구하는 질문이면 web_search로 라우팅  
- 문서에 있을 수 있는 구체적인 사실에 대한 질문이면 vectorstore로 라우팅
- 확실하지 않은 경우 vectorstore를 우선 선택"""

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
