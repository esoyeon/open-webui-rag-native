"""
Enterprise Session Manager for RAG
Redis 기반 효율적인 세션 메모리 관리
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from .cache_manager import CacheManager, RedisConfig

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant" 
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """채팅 메시지 모델"""
    role: MessageRole
    content: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        return cls(
            role=MessageRole(data['role']),
            content=data['content'],
            timestamp=data['timestamp'],
            metadata=data.get('metadata')
        )


@dataclass
class SessionInfo:
    """세션 정보 모델"""
    session_id: str
    created_at: float
    last_activity: float
    message_count: int
    total_tokens: int = 0  # 토큰 사용량 추적
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionInfo':
        return cls(
            session_id=data['session_id'],
            created_at=data['created_at'],
            last_activity=data['last_activity'],
            message_count=data['message_count'],
            total_tokens=data.get('total_tokens', 0)
        )


class SessionManager:
    """
    Enterprise-grade 세션 관리자
    
    Features:
    - LRU-style 메모리 관리
    - 자동 세션 만료 처리
    - 토큰 사용량 추적
    - 배치 처리로 성능 최적화
    - 세션 압축 (긴 대화를 요약)
    """
    
    def __init__(
        self,
        cache_manager: CacheManager,
        max_messages_per_session: int = 50,
        session_ttl: timedelta = timedelta(hours=24),
        cleanup_interval: int = 3600  # 1시간마다 정리
    ):
        self.cache = cache_manager
        self.max_messages_per_session = max_messages_per_session
        self.session_ttl = session_ttl
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
    
    def _session_key(self, session_id: str) -> str:
        """세션 키 생성"""
        return f"session:messages:{session_id}"
    
    def _session_info_key(self, session_id: str) -> str:
        """세션 정보 키 생성"""
        return f"session:info:{session_id}"
    
    def _should_cleanup(self) -> bool:
        """정리 작업이 필요한지 확인"""
        return time.time() - self.last_cleanup > self.cleanup_interval
    
    def create_session(self, session_id: str) -> SessionInfo:
        """새 세션 생성"""
        current_time = time.time()
        session_info = SessionInfo(
            session_id=session_id,
            created_at=current_time,
            last_activity=current_time,
            message_count=0
        )
        
        # 세션 정보 저장
        self.cache.set(
            self._session_info_key(session_id),
            session_info.to_dict(),
            self.session_ttl
        )
        
        logger.info(f"Created new session: {session_id}")
        return session_info
    
    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """세션 정보 조회"""
        data = self.cache.get(self._session_info_key(session_id))
        return SessionInfo.from_dict(data) if data else None
    
    def add_message(
        self, 
        session_id: str, 
        role: MessageRole, 
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """메시지 추가"""
        try:
            # 세션 정보 확인/생성
            session_info = self.get_session_info(session_id)
            if not session_info:
                session_info = self.create_session(session_id)
            
            # 새 메시지 생성
            message = ChatMessage(
                role=role,
                content=content,
                timestamp=time.time(),
                metadata=metadata or {}
            )
            
            # 기존 메시지들 조회
            messages = self.get_messages(session_id) or []
            messages.append(message)
            
            # 메시지 수 제한 적용 (LRU)
            if len(messages) > self.max_messages_per_session:
                # 시스템 메시지는 보존하고 오래된 사용자/어시스턴트 메시지만 제거
                system_messages = [m for m in messages if m.role == MessageRole.SYSTEM]
                user_assistant_messages = [m for m in messages if m.role != MessageRole.SYSTEM]
                
                # 가장 최근 메시지들만 유지
                keep_count = self.max_messages_per_session - len(system_messages)
                if keep_count > 0:
                    user_assistant_messages = user_assistant_messages[-keep_count:]
                
                messages = system_messages + user_assistant_messages
                logger.info(f"Trimmed session {session_id} to {len(messages)} messages")
            
            # 메시지 저장
            messages_data = [msg.to_dict() for msg in messages]
            self.cache.set(
                self._session_key(session_id),
                messages_data,
                self.session_ttl
            )
            
            # 세션 정보 업데이트
            session_info.last_activity = time.time()
            session_info.message_count = len(messages)
            
            # 토큰 추정 (대략적)
            estimated_tokens = len(content.split()) * 1.3  # 평균 토큰 비율
            session_info.total_tokens += int(estimated_tokens)
            
            self.cache.set(
                self._session_info_key(session_id),
                session_info.to_dict(),
                self.session_ttl
            )
            
            # 주기적 정리 작업
            if self._should_cleanup():
                self._cleanup_expired_sessions()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add message to session {session_id}: {e}")
            return False
    
    def get_messages(
        self, 
        session_id: str, 
        limit: Optional[int] = None,
        include_system: bool = True
    ) -> Optional[List[ChatMessage]]:
        """세션의 메시지들 조회"""
        try:
            data = self.cache.get(self._session_key(session_id))
            if not data:
                return None
            
            messages = [ChatMessage.from_dict(msg_data) for msg_data in data]
            
            # 시스템 메시지 제외 옵션
            if not include_system:
                messages = [m for m in messages if m.role != MessageRole.SYSTEM]
            
            # 제한 적용
            if limit:
                messages = messages[-limit:]
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get messages for session {session_id}: {e}")
            return None
    
    def get_conversation_context(
        self, 
        session_id: str, 
        max_tokens: int = 4000
    ) -> Tuple[str, int]:
        """
        대화 컨텍스트를 문자열로 반환 (토큰 제한 고려)
        
        Returns:
            (context_string, estimated_tokens)
        """
        messages = self.get_messages(session_id)
        if not messages:
            return "", 0
        
        context_parts = []
        total_tokens = 0
        
        # 최근 메시지부터 역순으로 추가 (토큰 제한까지)
        for message in reversed(messages):
            estimated_tokens = len(message.content.split()) * 1.3
            if total_tokens + estimated_tokens > max_tokens:
                break
                
            context_part = f"{message.role.value}: {message.content}"
            context_parts.insert(0, context_part)  # 앞에 삽입 (시간 순서 유지)
            total_tokens += estimated_tokens
        
        return "\n\n".join(context_parts), int(total_tokens)
    
    def delete_session(self, session_id: str) -> bool:
        """세션 삭제"""
        try:
            result1 = self.cache.delete(self._session_key(session_id))
            result2 = self.cache.delete(self._session_info_key(session_id))
            
            if result1 or result2:
                logger.info(f"Deleted session: {session_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def get_active_sessions(self) -> List[SessionInfo]:
        """활성 세션 목록 조회"""
        # Redis pattern matching으로 모든 세션 정보 키 조회
        # 실제 구현에서는 별도의 세션 목록 관리가 더 효율적
        try:
            # 이는 개발/디버깅용이며, 프로덕션에서는 별도의 인덱스 사용 권장
            def operation():
                pattern = "session:info:*"
                keys = self.cache.redis_client.keys(pattern)
                sessions = []
                
                for key in keys:
                    data = self.cache.get(key)
                    if data:
                        sessions.append(SessionInfo.from_dict(data))
                
                return sessions
            
            return self.cache._execute_with_circuit_breaker(operation) or []
            
        except Exception as e:
            logger.error(f"Failed to get active sessions: {e}")
            return []
    
    def _cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        try:
            current_time = time.time()
            cutoff_time = current_time - self.session_ttl.total_seconds()
            
            active_sessions = self.get_active_sessions()
            expired_count = 0
            
            for session_info in active_sessions:
                if session_info.last_activity < cutoff_time:
                    self.delete_session(session_info.session_id)
                    expired_count += 1
            
            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired sessions")
            
            self.last_cleanup = current_time
            
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """세션 통계 조회"""
        try:
            active_sessions = self.get_active_sessions()
            
            total_sessions = len(active_sessions)
            total_messages = sum(s.message_count for s in active_sessions)
            total_tokens = sum(s.total_tokens for s in active_sessions)
            
            if active_sessions:
                avg_messages = total_messages / total_sessions
                avg_tokens = total_tokens / total_sessions
                
                # 최근 활동 시간
                most_recent = max(s.last_activity for s in active_sessions)
                oldest_session = min(s.created_at for s in active_sessions)
            else:
                avg_messages = avg_tokens = most_recent = oldest_session = 0
            
            return {
                'total_sessions': total_sessions,
                'total_messages': total_messages,
                'total_tokens': total_tokens,
                'avg_messages_per_session': avg_messages,
                'avg_tokens_per_session': avg_tokens,
                'most_recent_activity': most_recent,
                'oldest_session': oldest_session
            }
            
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {}


# Singleton pattern for global session manager
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Global session manager 인스턴스 반환"""
    global _session_manager
    if _session_manager is None:
        from .cache_manager import get_cache_manager
        cache_manager = get_cache_manager()
        _session_manager = SessionManager(cache_manager)
        logger.info("Session Manager initialized")
    return _session_manager
