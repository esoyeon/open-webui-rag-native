"""
Adaptive RAG Pipeline for Open WebUI
Open WebUI에서 사용할 수 있는 Adaptive RAG 파이프라인
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from adaptive_rag import AdaptiveRAGGraph, FAISSVectorStore
from langchain_openai import OpenAIEmbeddings
from rag.pdf import PDFRetrievalChain

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pipe:
    """
    Open WebUI에서 인식하는 메인 파이프라인 클래스
    Adaptive RAG 기능을 Open WebUI와 통합
    """

    def __init__(self):
        self.type = "pipe"
        self.name = "Adaptive RAG Pipeline"
        self.id = "adaptive_rag_pipeline"

        # 환경변수 확인
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")

        # Adaptive RAG 구성요소 초기화
        self.vector_store = None
        self.rag_graph = None
        self.is_initialized = False

        # 초기화 시도
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """파이프라인 초기화"""
        try:
            if not self.openai_api_key:
                logger.warning(
                    "OpenAI API key not available. Pipeline will use mock responses."
                )
                return

            # OpenAI 임베딩 모델 초기화
            embeddings = OpenAIEmbeddings(
                openai_api_key=self.openai_api_key, model="text-embedding-ada-002"
            )

            # FAISS 벡터 스토어 초기화
            self.vector_store = FAISSVectorStore(
                embedding_function=embeddings, dimension=1536
            )

            # 기존 벡터 스토어가 있다면 로드
            vector_store_path = os.path.join(project_root, "data", "vector_store")
            if os.path.exists(vector_store_path):
                try:
                    self.vector_store.load(vector_store_path)
                    logger.info("Loaded existing vector store")
                except Exception as e:
                    logger.warning(f"Failed to load existing vector store: {e}")

            # Adaptive RAG 그래프 생성
            self.rag_graph = AdaptiveRAGGraph(
                vector_store=self.vector_store, model_name="gpt-3.5-turbo"
            )
            self.app = self.rag_graph.create_graph()

            self.is_initialized = True
            logger.info("Adaptive RAG Pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            self.is_initialized = False

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> str:
        """
        Open WebUI에서 호출하는 메인 파이프라인 메서드

        Args:
            user_message: 사용자의 현재 메시지
            model_id: 선택된 모델 ID
            messages: 전체 대화 히스토리
            body: 요청 본문

        Returns:
            생성된 답변
        """
        logger.info(f"Processing message: {user_message}")

        # 파이프라인이 초기화되지 않은 경우
        if not self.is_initialized:
            return self._handle_uninitialized_state(user_message)

        try:
            # Adaptive RAG 그래프 실행
            result = self.rag_graph.run(user_message)

            # 생성된 답변 반환
            generated_answer = result.get("generation", "답변을 생성할 수 없습니다.")

            # 메타데이터 로깅
            documents = result.get("documents", [])
            logger.info(f"Generated answer using {len(documents)} documents")

            return generated_answer

        except Exception as e:
            logger.error(f"Error in pipeline execution: {e}")
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"

    def _handle_uninitialized_state(self, user_message: str) -> str:
        """초기화되지 않은 상태에서의 처리"""
        logger.warning("Pipeline not initialized, providing fallback response")

        # 간단한 규칙 기반 응답
        if any(keyword in user_message.lower() for keyword in ["안녕", "hello", "hi"]):
            return "안녕하세요! Adaptive RAG 파이프라인입니다. 현재 초기화 중이니 잠시만 기다려주세요."

        return "죄송합니다. 시스템이 아직 완전히 초기화되지 않았습니다. OpenAI API 키를 확인하고 다시 시도해 주세요."

    def add_documents(self, documents: List[dict]) -> Dict[str, Any]:
        """
        문서를 벡터 스토어에 추가

        Args:
            documents: 추가할 문서들 (path, content, metadata 포함)

        Returns:
            처리 결과
        """
        if not self.is_initialized:
            return {"error": "Pipeline not initialized"}

        try:
            from langchain_core.documents import Document

            # 문서 객체 생성
            doc_objects = []
            for doc in documents:
                if isinstance(doc, str):
                    # 파일 경로인 경우
                    if os.path.exists(doc) and doc.endswith(".pdf"):
                        pdf_chain = PDFRetrievalChain([doc])
                        pdf_chain.create_chain()  # 체인 생성
                        pdf_docs = pdf_chain.load_documents([doc])  # 파라미터 전달
                        doc_objects.extend(pdf_docs)
                elif isinstance(doc, dict):
                    # 딕셔너리 형태인 경우
                    content = doc.get("content", "")
                    metadata = doc.get("metadata", {})
                    doc_objects.append(
                        Document(page_content=content, metadata=metadata)
                    )

            # 벡터 스토어에 추가
            if doc_objects:
                self.vector_store.add_documents(doc_objects)

                # 벡터 스토어 저장
                vector_store_path = os.path.join(project_root, "data", "vector_store")
                os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
                self.vector_store.save(vector_store_path)

                return {
                    "success": True,
                    "message": f"Successfully added {len(doc_objects)} documents",
                    "document_count": len(doc_objects),
                }
            else:
                return {"error": "No valid documents to add"}

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {"error": f"Failed to add documents: {str(e)}"}

    def get_status(self) -> Dict[str, Any]:
        """파이프라인 상태 정보 반환"""
        status = {
            "name": self.name,
            "id": self.id,
            "type": self.type,
            "initialized": self.is_initialized,
            "openai_api_key_available": bool(self.openai_api_key),
        }

        if self.vector_store:
            status.update(self.vector_store.get_stats())

        return status


# Open WebUI에서 파이프라인을 인식할 수 있도록 클래스를 모듈 레벨에서 사용 가능하게 만듦
__all__ = ["Pipe"]

# 테스트용 메인 함수
if __name__ == "__main__":
    # 파이프라인 테스트
    pipeline = Pipe()
    print("Pipeline Status:", pipeline.get_status())

    # 간단한 테스트
    test_message = "안녕하세요!"
    response = pipeline.pipe(test_message, "gpt-3.5-turbo", [], {})
    print(f"Test Response: {response}")
