"""
FAISS Vector Store Implementation for Adaptive RAG
"""

import os
import pickle
import logging
from typing import List, Optional, Tuple
import numpy as np
import faiss
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS 기반 벡터 스토어"""

    def __init__(self, embedding_function, dimension: int = 1536):
        self.embedding_function = embedding_function
        self.dimension = dimension
        self.index = None
        self.documents = []
        self.doc_embeddings = []
        self.is_built = False

    def add_documents(self, documents: List[Document]) -> None:
        """문서를 벡터 스토어에 추가"""
        logger.info(f"Adding {len(documents)} documents to vector store")

        # 문서들을 임베딩으로 변환
        texts = [doc.page_content for doc in documents]
        embeddings = []

        for text in texts:
            try:
                embedding = self.embedding_function.embed_query(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error embedding text: {e}")
                continue

        if not embeddings:
            logger.warning("No valid embeddings generated")
            return

        # FAISS 인덱스 생성 (처음인 경우)
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)

        # 임베딩을 numpy 배열로 변환하고 정규화
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)

        # 인덱스에 추가
        self.index.add(embeddings_array)

        # 문서와 임베딩 저장
        self.documents.extend(documents)
        self.doc_embeddings.extend(embeddings_array)

        self.is_built = True
        logger.info(f"Vector store now contains {len(self.documents)} documents")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """유사도 검색 수행"""
        if not self.is_built or self.index is None:
            logger.warning("Vector store is not built or empty")
            return []

        try:
            # 쿼리를 임베딩으로 변환
            query_embedding = self.embedding_function.embed_query(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)

            # 유사도 검색 수행
            distances, indices = self.index.search(
                query_vector, min(k, len(self.documents))
            )

            # 결과 문서 반환
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx != -1:  # 유효한 인덱스인지 확인
                    doc = self.documents[idx]
                    # 메타데이터에 검색 점수 추가
                    doc.metadata = doc.metadata or {}
                    doc.metadata["score"] = float(1 - distance)  # 거리를 유사도로 변환
                    doc.metadata["rank"] = i + 1
                    results.append(doc)

            logger.info(f"Found {len(results)} similar documents for query")
            return results

        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []

    def save(self, directory: str) -> None:
        """벡터 스토어를 디스크에 저장"""
        os.makedirs(directory, exist_ok=True)

        if self.index is not None:
            # FAISS 인덱스 저장
            faiss.write_index(self.index, os.path.join(directory, "faiss.index"))

            # 문서들과 메타데이터 저장
            with open(os.path.join(directory, "documents.pkl"), "wb") as f:
                pickle.dump(
                    {
                        "documents": self.documents,
                        "doc_embeddings": self.doc_embeddings,
                        "dimension": self.dimension,
                    },
                    f,
                )

            logger.info(f"Vector store saved to {directory}")

    def load(self, directory: str) -> None:
        """디스크에서 벡터 스토어 로드"""
        index_path = os.path.join(directory, "faiss.index")
        docs_path = os.path.join(directory, "documents.pkl")

        if os.path.exists(index_path) and os.path.exists(docs_path):
            # FAISS 인덱스 로드
            self.index = faiss.read_index(index_path)

            # 문서들과 메타데이터 로드
            with open(docs_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.doc_embeddings = data["doc_embeddings"]
                self.dimension = data["dimension"]

            self.is_built = True
            logger.info(
                f"Vector store loaded from {directory} with {len(self.documents)} documents"
            )
        else:
            logger.warning(f"Vector store files not found in {directory}")

    def get_stats(self) -> dict:
        """벡터 스토어 통계 반환"""
        return {
            "total_documents": len(self.documents),
            "dimension": self.dimension,
            "is_built": self.is_built,
            "index_size": self.index.ntotal if self.index else 0,
        }
