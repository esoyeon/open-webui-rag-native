"""
Adaptive RAG Graph Implementation
LangGraph를 사용한 Adaptive RAG 워크플로우
"""

import logging
from typing import List, Dict, Any
from typing_extensions import TypedDict, Annotated

from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

from .router import QueryRouter
from .grader import DocumentGrader, QualityGrader
from .rewriter import QueryRewriter
from .nodes import RAGNodes

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """
    그래프의 상태를 나타내는 데이터 모델

    Attributes:
        question: 사용자 질문
        generation: LLM이 생성한 답변
        documents: 검색된 문서들
        retry_count: 재시도 횟수
    """

    question: Annotated[str, "User question"]
    generation: Annotated[str, "LLM generated answer"]
    documents: Annotated[List[Dict], "List of retrieved documents"]
    retry_count: Annotated[int, "Number of retries"]


class AdaptiveRAGGraph:
    """Adaptive RAG 그래프 클래스"""

    def __init__(
        self,
        vector_store=None,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0,
    ):
        """
        Args:
            vector_store: FAISS 벡터 스토어 인스턴스
            model_name: 사용할 LLM 모델명
            temperature: LLM temperature 설정
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.temperature = temperature

        # 각 구성 요소 초기화
        self.query_router = QueryRouter(model_name, temperature)
        self.document_grader = DocumentGrader(model_name, temperature)
        self.quality_grader = QualityGrader(model_name)
        self.query_rewriter = QueryRewriter(model_name, temperature)
        self.rag_nodes = RAGNodes(vector_store, model_name, temperature)

        # 그래프 저장용
        self.app = None

    def create_graph(self):
        """LangGraph 워크플로우 생성 및 컴파일"""
        logger.info("Creating Adaptive RAG graph...")

        # 그래프 상태 초기화
        workflow = StateGraph(GraphState)

        # 노드 정의
        workflow.add_node("web_search", self.rag_nodes.web_search)
        workflow.add_node("retrieve", self.rag_nodes.retrieve)
        workflow.add_node("grade_documents", self.document_grader.grade_documents_node)
        workflow.add_node("generate", self.rag_nodes.generate)
        workflow.add_node("transform_query", self.query_rewriter.transform_query_node)

        # 조건부 엣지: 시작점에서 질문 라우팅
        workflow.add_conditional_edges(
            START,
            self.query_router.route_question_node,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
            },
        )

        # 웹 검색 후 바로 답변 생성
        workflow.add_edge("web_search", "generate")

        # 벡터 검색 후 문서 평가
        workflow.add_edge("retrieve", "grade_documents")

        # 문서 평가 후 생성 여부 결정
        workflow.add_conditional_edges(
            "grade_documents",
            self.rag_nodes.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )

        # 질문 재작성 후 다시 검색
        workflow.add_edge("transform_query", "retrieve")

        # 답변 생성 후 품질 평가
        workflow.add_conditional_edges(
            "generate",
            self.quality_grader.hallucination_check_node,
            {
                "hallucination": "generate",  # 환각 발견시 재생성
                "relevant": END,  # 관련성 있는 답변
                "not_relevant": "transform_query",  # 관련성 없으면 질문 재작성
            },
        )

        # 그래프 컴파일
        self.app = workflow.compile(checkpointer=MemorySaver())

        logger.info("Adaptive RAG graph created successfully")
        return self.app

    def run(self, question: str, thread_id: str = "default") -> Dict[str, Any]:
        """
        그래프 실행

        Args:
            question: 사용자 질문
            thread_id: 대화 스레드 ID

        Returns:
            최종 실행 결과
        """
        if not self.app:
            raise RuntimeError("Graph not created. Call create_graph() first.")

        logger.info(f"Running Adaptive RAG for question: {question}")

        # 입력 데이터 준비
        inputs = {"question": question, "retry_count": 0}
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50,  # 재귀 제한 증가
        }

        try:
            # 그래프 실행
            result = self.app.invoke(inputs, config)

            logger.info("Adaptive RAG execution completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error running Adaptive RAG: {e}")
            return {
                "question": question,
                "generation": "죄송합니다. 답변 생성 중 오류가 발생했습니다.",
                "documents": [],
            }

    def stream_run(self, question: str, thread_id: str = "default"):
        """
        그래프 스트리밍 실행 (단계별 결과 확인용)

        Args:
            question: 사용자 질문
            thread_id: 대화 스레드 ID

        Yields:
            각 노드 실행 결과
        """
        if not self.app:
            raise RuntimeError("Graph not created. Call create_graph() first.")

        inputs = {"question": question}
        config = {"configurable": {"thread_id": thread_id}}

        try:
            for step in self.app.stream(inputs, config):
                yield step
        except Exception as e:
            logger.error(f"Error in streaming Adaptive RAG: {e}")
            yield {"error": str(e)}

    def get_graph_image(self, output_path: str = "adaptive_rag_graph.png"):
        """그래프 시각화 이미지 생성"""
        if not self.app:
            raise RuntimeError("Graph not created. Call create_graph() first.")

        try:
            # Mermaid 형식으로 그래프 출력
            graph_representation = self.app.get_graph().draw_mermaid()
            logger.info(f"Graph representation:\n{graph_representation}")
            return graph_representation
        except Exception as e:
            logger.error(f"Error generating graph image: {e}")
            return None
