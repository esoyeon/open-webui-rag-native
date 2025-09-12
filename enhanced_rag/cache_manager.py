"""
Enterprise-grade Redis Cache Manager
현업에서 사용하는 Redis 기반 캐싱 시스템
"""

import json
import hashlib
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import timedelta
import redis
from redis import ConnectionPool
import pickle
import numpy as np

logger = logging.getLogger(__name__)


class RedisConfig:
    """Redis 연결 설정"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 20,
        socket_keepalive: bool = True,
        socket_keepalive_options: Dict = None
    ):
        if socket_keepalive_options is None:
            socket_keepalive_options = {
                1: 1,  # TCP_KEEPIDLE
                2: 3,  # TCP_KEEPINTVL
                3: 5,  # TCP_KEEPCNT
            }
        
        self.pool = ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            socket_keepalive=socket_keepalive,
            socket_keepalive_options=socket_keepalive_options,
            decode_responses=True
        )


class CacheManager:
    """
    Enterprise-grade Redis 캐시 매니저
    
    Features:
    - Connection pooling
    - Automatic serialization
    - TTL management
    - Circuit breaker pattern
    - Metrics collection
    """
    
    def __init__(self, config: RedisConfig):
        self.redis_client = redis.Redis(connection_pool=config.pool)
        self.is_healthy = True
        self.error_count = 0
        self.max_errors = 5
        
    def _generate_key(self, prefix: str, *args) -> str:
        """키 생성 (해시 기반)"""
        key_data = f"{prefix}:{':'.join(map(str, args))}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def _serialize_value(self, value: Any) -> bytes:
        """값 직렬화"""
        if isinstance(value, (dict, list)):
            return json.dumps(value).encode()
        elif isinstance(value, np.ndarray):
            return pickle.dumps(value)
        elif isinstance(value, str):
            return value.encode()
        else:
            return pickle.dumps(value)
    
    def _deserialize_value(self, value: bytes, value_type: str = 'auto') -> Any:
        """값 역직렬화"""
        try:
            if value_type == 'json':
                return json.loads(value.decode())
            elif value_type == 'pickle':
                return pickle.loads(value)
            else:
                # Auto detect
                try:
                    return json.loads(value.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return pickle.loads(value)
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return None
    
    def _execute_with_circuit_breaker(self, operation):
        """Circuit breaker 패턴으로 Redis 작업 실행"""
        if not self.is_healthy:
            logger.warning("Circuit breaker is open, skipping Redis operation")
            return None
            
        try:
            result = operation()
            # 성공하면 에러 카운트 리셋
            if self.error_count > 0:
                self.error_count = 0
                logger.info("Redis connection restored")
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Redis error ({self.error_count}/{self.max_errors}): {e}")
            
            if self.error_count >= self.max_errors:
                self.is_healthy = False
                logger.error("Circuit breaker opened - Redis operations disabled")
            
            return None
    
    def get(self, key: str, default=None) -> Any:
        """캐시에서 값 조회"""
        def operation():
            value = self.redis_client.get(key)
            return self._deserialize_value(value) if value else default
            
        return self._execute_with_circuit_breaker(operation) or default
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[timedelta] = None
    ) -> bool:
        """캐시에 값 저장"""
        def operation():
            serialized_value = self._serialize_value(value)
            if ttl:
                return self.redis_client.setex(key, ttl, serialized_value)
            else:
                return self.redis_client.set(key, serialized_value)
                
        result = self._execute_with_circuit_breaker(operation)
        return result is not None and result
    
    def delete(self, key: str) -> bool:
        """캐시에서 값 삭제"""
        def operation():
            return self.redis_client.delete(key) > 0
            
        result = self._execute_with_circuit_breaker(operation)
        return result or False
    
    def exists(self, key: str) -> bool:
        """키 존재 여부 확인"""
        def operation():
            return self.redis_client.exists(key) > 0
            
        result = self._execute_with_circuit_breaker(operation)
        return result or False
    
    def get_health(self) -> Dict[str, Any]:
        """캐시 시스템 상태 조회"""
        def operation():
            info = self.redis_client.info('memory')
            return {
                'healthy': True,
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'connected_clients': info.get('connected_clients', 0)
            }
            
        result = self._execute_with_circuit_breaker(operation)
        if result:
            return result
        else:
            return {
                'healthy': False,
                'error_count': self.error_count,
                'is_healthy': self.is_healthy
            }


class RAGCacheManager(CacheManager):
    """
    RAG 특화 캐시 매니저
    검색 결과, 임베딩, 생성된 답변 캐싱
    """
    
    def __init__(self, config: RedisConfig):
        super().__init__(config)
        self.EMBEDDING_CACHE_TTL = timedelta(hours=24)
        self.SEARCH_CACHE_TTL = timedelta(hours=1)
        self.ANSWER_CACHE_TTL = timedelta(minutes=30)
    
    def cache_embedding(self, text: str, embedding: np.ndarray) -> bool:
        """임베딩 결과 캐싱"""
        key = self._generate_key("embedding", text)
        return self.set(key, embedding, self.EMBEDDING_CACHE_TTL)
    
    def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """캐시된 임베딩 조회"""
        key = self._generate_key("embedding", text)
        return self.get(key)
    
    def cache_search_results(
        self, 
        query: str, 
        results: List[Dict], 
        search_type: str = "vector"
    ) -> bool:
        """검색 결과 캐싱"""
        key = self._generate_key("search", search_type, query)
        return self.set(key, results, self.SEARCH_CACHE_TTL)
    
    def get_cached_search_results(
        self, 
        query: str, 
        search_type: str = "vector"
    ) -> Optional[List[Dict]]:
        """캐시된 검색 결과 조회"""
        key = self._generate_key("search", search_type, query)
        return self.get(key)
    
    def cache_answer(self, question: str, answer: str, context_hash: str) -> bool:
        """생성된 답변 캐싱"""
        key = self._generate_key("answer", question, context_hash)
        return self.set(key, answer, self.ANSWER_CACHE_TTL)
    
    def get_cached_answer(self, question: str, context_hash: str) -> Optional[str]:
        """캐시된 답변 조회"""
        key = self._generate_key("answer", question, context_hash)
        return self.get(key)
    
    def invalidate_search_cache(self) -> int:
        """검색 캐시 무효화"""
        def operation():
            pattern = f"*search*"
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
            
        result = self._execute_with_circuit_breaker(operation)
        return result or 0


# Singleton pattern for global cache manager
_cache_manager: Optional[RAGCacheManager] = None


def get_cache_manager() -> RAGCacheManager:
    """Global cache manager 인스턴스 반환"""
    global _cache_manager
    if _cache_manager is None:
        config = RedisConfig()
        _cache_manager = RAGCacheManager(config)
        logger.info("RAG Cache Manager initialized")
    return _cache_manager
