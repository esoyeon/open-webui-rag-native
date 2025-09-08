"""
Graph Nodes for Adaptive RAG
LangGraph에서 사용되는 각종 노드 함수들
"""

import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class WebSearchTool:
    """웹 검색 도구 (Tavily 대신 간단한 구현)"""

    def __init__(self):
        # 실제로는 Tavily나 다른 웹 검색 API를 사용해야 합니다
        logger.warning(
            "WebSearchTool: Using mock implementation. Please integrate real web search API."
        )

    def search(self, query: str, max_results: int = 3) -> List[dict]:
        """웹 검색 수행 (Mock 구현)"""
        # 실제 구현에서는 Tavily나 다른 검색 API를 호출
        logger.info(f"Web search query: {query}")

        # Mock 결과 반환
        mock_results = [
            {
                "content": f"Mock web search result 1 for query: {query}. This would contain actual web content in real implementation.",
                "url": "https://example.com/result1",
                "title": f"Search Result 1 for {query}",
            },
            {
                "content": f"Mock web search result 2 for query: {query}. Real implementation would fetch current web content.",
                "url": "https://example.com/result2",
                "title": f"Search Result 2 for {query}",
            },
        ]

        return mock_results[:max_results]


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
        rag_template = """당신은 질문-답변 업무를 수행하는 어시스턴트입니다. 
다음의 검색된 문맥 정보를 사용하여 질문에 답변해주세요. 
답을 모르는 경우, 모른다고 솔직하게 말해주세요. 
최대 3문장으로 간결하고 정확하게 답변해주세요.
**사용자 질문이 한국어면 반드시 한국어로, 영어면 영어로 답변해주세요.**

Context: {context}

Question: {question}

Answer (질문 언어와 동일한 언어로 답변):"""

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
            documents = self.vector_store.similarity_search(question, k=5)
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
