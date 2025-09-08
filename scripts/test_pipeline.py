#!/usr/bin/env python3
"""
Pipeline Testing Script for Adaptive RAG
파이프라인 동작을 테스트하는 스크립트
"""
import os
import sys
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from pipelines.adaptive_rag_pipeline import Pipe
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_pipeline():
    """파이프라인 테스트 함수"""
    logger.info("Starting pipeline test...")

    try:
        # 파이프라인 초기화
        pipeline = Pipe()

        # 상태 확인
        status = pipeline.get_status()
        logger.info("Pipeline Status:")
        for key, value in status.items():
            logger.info(f"  {key}: {value}")

        # 테스트 질문들
        test_questions = [
            "안녕하세요!",
            "삼성전자가 개발한 생성형 AI의 이름은 무엇인가요?",
            "AI Brief에서 언급된 주요 내용을 요약해주세요.",
            "2024년 최신 AI 동향은 어떤가요?",
            "이 문서에서 다루는 주제는 무엇인가요?",
        ]

        logger.info("\nTesting pipeline with sample questions...")
        logger.info("=" * 60)

        for i, question in enumerate(test_questions, 1):
            logger.info(f"\n[Test {i}] Question: {question}")
            logger.info("-" * 40)

            try:
                response = pipeline.pipe(
                    user_message=question,
                    model_id="gpt-3.5-turbo",
                    messages=[],
                    body={},
                )

                logger.info(f"Response: {response}")

            except Exception as e:
                logger.error(f"Error processing question {i}: {e}")

        logger.info("\n" + "=" * 60)
        logger.info("Pipeline test completed!")

        return True

    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        return False


def test_document_addition():
    """문서 추가 기능 테스트"""
    logger.info("\nTesting document addition...")

    try:
        pipeline = Pipe()

        # 샘플 문서 경로
        sample_doc_path = project_root / "data" / "SPRI_AI_Brief_2023년12월호_F.pdf"

        if sample_doc_path.exists():
            logger.info(f"Adding document: {sample_doc_path}")
            result = pipeline.add_documents([str(sample_doc_path)])

            if result.get("success"):
                logger.info(f"Document added successfully: {result}")
            else:
                logger.error(f"Failed to add document: {result}")
        else:
            logger.warning(f"Sample document not found: {sample_doc_path}")

            # 샘플 텍스트 문서 추가
            sample_docs = [
                {
                    "content": "이것은 테스트용 문서입니다. Adaptive RAG 시스템에 대한 정보를 담고 있습니다.",
                    "metadata": {"source": "test_doc_1", "type": "sample"},
                },
                {
                    "content": "LangGraph를 사용한 Adaptive RAG는 쿼리 라우팅과 문서 평가 기능을 제공합니다.",
                    "metadata": {"source": "test_doc_2", "type": "sample"},
                },
            ]

            result = pipeline.add_documents(sample_docs)
            logger.info(f"Sample documents added: {result}")

    except Exception as e:
        logger.error(f"Document addition test failed: {e}")


def main():
    """메인 실행 함수"""
    logger.info("Adaptive RAG Pipeline Test Suite")
    logger.info("=" * 60)

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        logger.error("Please set your OpenAI API key in .env file or environment")
        return 1

    # 기본 파이프라인 테스트
    success = test_pipeline()

    if success:
        # 문서 추가 기능 테스트
        test_document_addition()

        logger.info("\nAll tests completed!")
        return 0
    else:
        logger.error("Pipeline test failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
