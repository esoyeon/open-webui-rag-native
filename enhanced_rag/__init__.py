"""
Enhanced RAG System
현업 패턴을 적용한 고성능 RAG 시스템

Features:
- Redis 기반 다단계 캐싱
- 효율적인 세션 메모리 관리
- 백그라운드 태스크 큐 (RQ)
- 간소화된 RAG 파이프라인
- Circuit breaker 패턴
- Connection pooling
"""

from .cache_manager import (
    CacheManager,
    RAGCacheManager,
    RedisConfig,
    get_cache_manager
)

from .session_manager import (
    SessionManager,
    ChatMessage,
    MessageRole,
    SessionInfo,
    get_session_manager
)

from .optimized_rag import (
    OptimizedRAGEngine,
    SearchType,
    SearchResult,
    RAGResponse
)

from .task_queue import (
    TaskQueue,
    get_task_queue,
    async_task,
    process_rag_async,
    cleanup_expired_sessions,
    cleanup_cache,
    index_document_async,
    start_worker
)

__version__ = "1.0.0"
__author__ = "Enhanced RAG Team"

__all__ = [
    # Cache Manager
    "CacheManager",
    "RAGCacheManager", 
    "RedisConfig",
    "get_cache_manager",
    
    # Session Manager
    "SessionManager",
    "ChatMessage",
    "MessageRole", 
    "SessionInfo",
    "get_session_manager",
    
    # Optimized RAG
    "OptimizedRAGEngine",
    "SearchType",
    "SearchResult", 
    "RAGResponse",
    
    # Task Queue
    "TaskQueue",
    "get_task_queue",
    "async_task",
    "process_rag_async",
    "cleanup_expired_sessions",
    "cleanup_cache",
    "index_document_async",
    "start_worker"
]
