#!/usr/bin/env python3
"""
Document Indexing Script for Adaptive RAG
PDF 문서들을 FAISS 벡터 스토어에 인덱싱하는 스크립트
"""
import os
import sys
import logging
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from adaptive_rag import FAISSVectorStore
from document_processing.pdf import PDFRetrievalChain
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Index PDF documents for Adaptive RAG")
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="data/documents",
        help="Directory containing PDF documents to index",
    )
    parser.add_argument(
        "--vector-store-dir",
        type=str,
        default="data/vector_store",
        help="Directory to save the vector store",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild of vector store even if it exists",
    )

    args = parser.parse_args()

    # API 키 확인
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        logger.error("Please set your OpenAI API key in .env file or environment")
        return 1

    # 문서 디렉토리 확인
    docs_path = project_root / args.docs_dir
    if not docs_path.exists():
        logger.info(f"Creating documents directory: {docs_path}")
        docs_path.mkdir(parents=True, exist_ok=True)

    # PDF 파일 검색
    pdf_files = list(docs_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {docs_path}")
        logger.info("Please add PDF files to the documents directory and run again")

        # 샘플 PDF가 있다면 복사
        sample_pdf = project_root / "data" / "RE-189_2024년국내외인공지능산업동향연구.pdf"
        if sample_pdf.exists():
            import shutil

            target_pdf = docs_path / sample_pdf.name
            shutil.copy2(sample_pdf, target_pdf)
            logger.info(f"Copied sample PDF: {target_pdf}")
            pdf_files = [target_pdf]
        else:
            logger.warning("샘플 PDF도 찾을 수 없습니다.")
            logger.info("data/documents/ 폴더에 PDF 파일을 추가한 후 다시 실행해주세요.")
            return 1

    logger.info(f"Found {len(pdf_files)} PDF files to index")
    for pdf_file in pdf_files:
        logger.info(f"  - {pdf_file.name}")

    # 벡터 스토어 디렉토리 설정
    vector_store_path = project_root / args.vector_store_dir
    vector_store_path.mkdir(parents=True, exist_ok=True)

    # 기존 벡터 스토어가 있고 force-rebuild가 아닌 경우 확인
    if (vector_store_path / "faiss.index").exists() and not args.force_rebuild:
        response = input(
            "Vector store already exists. Do you want to rebuild it? (y/N): "
        )
        if response.lower() != "y":
            logger.info("Skipping indexing. Use --force-rebuild to override.")
            return 0

    try:
        # OpenAI 임베딩 모델 초기화
        logger.info("Initializing OpenAI embeddings...")
        embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key, model="text-embedding-ada-002"
        )

        # FAISS 벡터 스토어 초기화
        logger.info("Initializing FAISS vector store...")
        vector_store = FAISSVectorStore(embedding_function=embeddings, dimension=1536)

        # 각 PDF 파일 처리
        all_documents = []
        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file.name}...")

            try:
                # PDF 문서 로드
                pdf_chain = PDFRetrievalChain(str(pdf_file))
                documents = pdf_chain.load_documents([str(pdf_file)])

                logger.info(f"Loaded {len(documents)} chunks from {pdf_file.name}")
                all_documents.extend(documents)

            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                continue

        if not all_documents:
            logger.error("No documents were successfully processed")
            return 1

        # 벡터 스토어에 문서 추가
        logger.info(f"Adding {len(all_documents)} document chunks to vector store...")
        vector_store.add_documents(all_documents)

        # 벡터 스토어 저장
        logger.info(f"Saving vector store to {vector_store_path}...")
        vector_store.save(str(vector_store_path))

        # 통계 정보 출력
        stats = vector_store.get_stats()
        logger.info("Indexing completed successfully!")
        logger.info(f"Total documents: {stats['total_documents']}")
        logger.info(f"Vector dimension: {stats['dimension']}")
        logger.info(f"Index size: {stats['index_size']}")

        return 0

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
