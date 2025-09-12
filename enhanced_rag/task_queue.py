"""
Background Task Queue System
RQ(Redis Queue)를 사용한 백그라운드 작업 처리

주요 기능:
- 비동기 RAG 처리
- 문서 인덱싱 작업
- 주기적 캐시 정리
- 세션 정리 작업
"""

import logging
import time
import asyncio
from typing import Any, Dict, Optional, Callable
from functools import wraps
from datetime import datetime, timedelta

from rq import Queue
from rq.job import Job, JobStatus
from rq.worker import Worker
import redis
from redis import Redis

from .cache_manager import get_cache_manager
from .session_manager import get_session_manager

logger = logging.getLogger(__name__)


class TaskQueue:
    """
    Background Task Queue Manager
    
    Features:
    - 비동기 작업 처리
    - 작업 상태 추적
    - 재시도 메커니즘
    - 우선순위 큐
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 1,  # 캐시와 다른 DB 사용
        default_timeout: int = 300,  # 5분
    ):
        try:
            # Redis 연결
            self.redis_conn = Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=False  # RQ는 bytes를 사용
            )
            
            # RQ Queue 생성
            self.default_queue = Queue('default', connection=self.redis_conn)
            self.high_priority_queue = Queue('high', connection=self.redis_conn)
            self.low_priority_queue = Queue('low', connection=self.redis_conn)
            
            # 작업 상태 추적용
            self.jobs = {}
            
            self.is_available = True
            logger.info("✅ Task Queue initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Task Queue initialization failed: {e}")
            self.redis_conn = None
            self.is_available = False
    
    def _get_queue(self, priority: str = 'default') -> Queue:
        """우선순위에 따른 큐 반환"""
        if priority == 'high':
            return self.high_priority_queue
        elif priority == 'low':
            return self.low_priority_queue
        else:
            return self.default_queue
    
    def enqueue_task(
        self,
        func: Callable,
        *args,
        priority: str = 'default',
        timeout: Optional[int] = None,
        retry_count: int = 3,
        **kwargs
    ) -> Optional[str]:
        """
        작업을 큐에 추가
        
        Args:
            func: 실행할 함수
            args: 함수 인자
            priority: 우선순위 ('high', 'default', 'low')
            timeout: 타임아웃 (초)
            retry_count: 재시도 횟수
            kwargs: 함수 키워드 인자
        
        Returns:
            Job ID 또는 None
        """
        if not self.is_available:
            logger.warning("Task queue not available, executing synchronously")
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Synchronous task execution failed: {e}")
                return None
        
        try:
            queue = self._get_queue(priority)
            
            # RQ 2.x API: use job_timeout and Retry object
            enqueue_kwargs = {
                "job_timeout": (timeout or 300),
            }

            job = queue.enqueue(
                func,
                *args,
                **enqueue_kwargs,
                **kwargs
            )
            
            self.jobs[job.id] = job
            logger.info(f"✅ Task enqueued: {job.id} ({func.__name__})")
            return job.id
            
        except Exception as e:
            logger.error(f"Failed to enqueue task: {e}")
            return None
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """작업 상태 조회"""
        if not self.is_available:
            return {"status": "unavailable"}
        
        try:
            job = Job.fetch(job_id, connection=self.redis_conn)
            
            return {
                "id": job_id,
                "status": job.get_status(),
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "ended_at": job.ended_at.isoformat() if job.ended_at else None,
                "result": job.result,
                "exc_info": job.exc_info
            }
            
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return {"status": "error", "error": str(e)}
    
    def cancel_job(self, job_id: str) -> bool:
        """작업 취소"""
        if not self.is_available:
            return False
        
        try:
            job = Job.fetch(job_id, connection=self.redis_conn)
            job.cancel()
            logger.info(f"✅ Job cancelled: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel job: {e}")
            return False
    
    def get_queue_info(self) -> Dict[str, Any]:
        """큐 정보 조회"""
        if not self.is_available:
            return {"available": False}
        
        try:
            return {
                "available": True,
                "queues": {
                    "high": {
                        "length": len(self.high_priority_queue),
                        "jobs": self.high_priority_queue.get_job_ids()
                    },
                    "default": {
                        "length": len(self.default_queue),
                        "jobs": self.default_queue.get_job_ids()
                    },
                    "low": {
                        "length": len(self.low_priority_queue),
                        "jobs": self.low_priority_queue.get_job_ids()
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue info: {e}")
            return {"available": False, "error": str(e)}


# Background task functions
def process_rag_async(question: str, session_id: str, search_type: str = None) -> Dict[str, Any]:
    """
    비동기 RAG 처리 태스크
    
    Args:
        question: 사용자 질문
        session_id: 세션 ID
        search_type: 검색 타입
    
    Returns:
        RAG 처리 결과
    """
    try:
        from .optimized_rag import OptimizedRAGEngine, SearchType
        
        # RAG 엔진 초기화 (워커에서 실행되므로 매번 초기화 필요)
        rag_engine = OptimizedRAGEngine()
        
        # 검색 타입 변환
        if search_type:
            search_type_enum = SearchType(search_type)
        else:
            search_type_enum = None
        
        # 동기 함수를 호출 (워커는 동기 환경)
        result = asyncio.run(
            rag_engine.process_question(question, session_id, search_type_enum)
        )
        
        return {
            "success": True,
            "answer": result.answer,
            "sources": [s.to_dict() for s in result.sources],
            "search_type": result.search_type.value,
            "response_time": result.response_time,
            "cached": result.cached
        }
        
    except Exception as e:
        logger.error(f"Async RAG processing failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "answer": "죄송합니다. 답변 생성 중 오류가 발생했습니다."
        }


def cleanup_expired_sessions() -> Dict[str, Any]:
    """만료된 세션 정리 태스크"""
    try:
        session_manager = get_session_manager()
        
        # 프라이빗 메소드를 직접 호출 (정리 작업)
        session_manager._cleanup_expired_sessions()
        
        stats = session_manager.get_session_stats()
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "session_stats": stats
        }
        
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")
        return {"success": False, "error": str(e)}


def cleanup_cache() -> Dict[str, Any]:
    """캐시 정리 태스크"""
    try:
        cache_manager = get_cache_manager()
        
        # 검색 캐시 무효화
        cleaned_count = cache_manager.invalidate_search_cache()
        
        # 캐시 상태 조회
        health = cache_manager.get_health()
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "cleaned_keys": cleaned_count,
            "cache_health": health
        }
        
    except Exception as e:
        logger.error(f"Cache cleanup failed: {e}")
        return {"success": False, "error": str(e)}


def index_document_async(document_path: str, document_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    문서 인덱싱 비동기 태스크
    
    Args:
        document_path: 문서 경로
        document_metadata: 문서 메타데이터
    
    Returns:
        인덱싱 결과
    """
    try:
        from document_processing.pdf import PDFRetrievalChain
        from adaptive_rag.vector_store import FAISSVectorStore
        from langchain_openai import OpenAIEmbeddings
        import os
        
        # PDF 처리
        if document_path.endswith('.pdf'):
            pdf_chain = PDFRetrievalChain([document_path])
            pdf_chain.create_chain()
            documents = pdf_chain.load_documents([document_path])
            
            # 메타데이터 추가
            if document_metadata:
                for doc in documents:
                    doc.metadata.update(document_metadata)
            
            # 벡터 스토어에 추가
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            vector_store = FAISSVectorStore(embedding_function=embeddings, dimension=1536)
            
            # 기존 벡터 스토어 로드
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            vector_store_path = os.path.join(project_root, "data", "vector_store")
            
            if os.path.exists(vector_store_path):
                vector_store.load(vector_store_path)
            
            # 새 문서 추가
            vector_store.add_documents(documents)
            
            # 저장
            os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
            vector_store.save(vector_store_path)
            
            return {
                "success": True,
                "document_path": document_path,
                "processed_chunks": len(documents),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "error": "Unsupported document format"
            }
            
    except Exception as e:
        logger.error(f"Document indexing failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "document_path": document_path
        }


# Singleton task queue
_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    """Global task queue 인스턴스 반환"""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
        logger.info("Task Queue initialized")
    return _task_queue


def async_task(priority: str = 'default', timeout: int = 300, retry: int = 3):
    """비동기 작업 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            task_queue = get_task_queue()
            return task_queue.enqueue_task(
                func, 
                *args, 
                priority=priority, 
                timeout=timeout, 
                retry_count=retry, 
                **kwargs
            )
        
        # 원래 함수도 접근 가능하도록 저장
        wrapper._original = func
        return wrapper
    
    return decorator


# Worker를 실행하는 헬퍼 함수
def start_worker(queue_names: list = None):
    """
    RQ 워커 시작
    
    Args:
        queue_names: 처리할 큐 이름 목록 (기본: ['high', 'default', 'low'])
    
    Usage:
        python -c "from enhanced_rag.task_queue import start_worker; start_worker()"
    """
    if queue_names is None:
        queue_names = ['high', 'default', 'low']
    
    try:
        # Redis 연결
        redis_conn = Redis(
            host='localhost',
            port=6379,
            db=1,
            decode_responses=False
        )
        
        # 큐 생성
        queues = [Queue(name, connection=redis_conn) for name in queue_names]
        
        # 워커 생성 및 시작
        worker = Worker(queues, connection=redis_conn)
        logger.info(f"🚀 Starting worker for queues: {queue_names}")
        
        worker.work()
        
    except KeyboardInterrupt:
        logger.info("🛑 Worker stopped by user")
    except Exception as e:
        logger.error(f"❌ Worker error: {e}")


if __name__ == "__main__":
    # 워커 실행
    start_worker()
