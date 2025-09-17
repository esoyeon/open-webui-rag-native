"""
Enterprise Session Manager for RAG
Redis 기반 효율적인 세션 메모리 관리
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import hashlib
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
        # Enum은 JSON 직렬화가 불가하므로 문자열 값으로 저장
        return {
            "role": self.role.value if isinstance(self.role, MessageRole) else str(self.role),
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        role_val = data.get('role', 'user')
        try:
            role_enum = role_val if isinstance(role_val, MessageRole) else MessageRole(str(role_val))
        except Exception:
            role_enum = MessageRole.USER
        return cls(
            role=role_enum,
            content=data.get('content', ''),
            timestamp=float(data.get('timestamp', 0)),
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

        # 캐시 장애 시 임시 메모리 폴백 (프로세스 한정)
        self._ephemeral_store: Dict[str, List[ChatMessage]] = {}
        self._ephemeral_info: Dict[str, SessionInfo] = {}
        self._ephemeral_meta: Dict[str, Dict[str, Any]] = {}
    
    def _session_key(self, session_id: str) -> str:
        """세션 키 생성"""
        return f"session:messages:{session_id}"
    
    def _session_info_key(self, session_id: str) -> str:
        """세션 정보 키 생성"""
        return f"session:info:{session_id}"

    def _session_meta_key(self, session_id: str) -> str:
        """세션 메타(경량 상태) 키 생성"""
        return f"session:meta:{session_id}"
    
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
            ok = self.cache.set(
                self._session_key(session_id),
                messages_data,
                self.session_ttl
            )
            if not ok:
                # 캐시 장애 시 임시 저장
                self._ephemeral_store[session_id] = [ChatMessage.from_dict(m) for m in messages_data]
            
            # 세션 정보 업데이트
            session_info.last_activity = time.time()
            session_info.message_count = len(messages)
            
            # 토큰 추정 (대략적)
            estimated_tokens = len(content.split()) * 1.3  # 평균 토큰 비율
            session_info.total_tokens += int(estimated_tokens)
            
            ok2 = self.cache.set(
                self._session_info_key(session_id),
                session_info.to_dict(),
                self.session_ttl
            )
            if not ok2:
                self._ephemeral_info[session_id] = session_info
            
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
                # 캐시 미사용/장애 시 임시 저장소 폴백
                data = [m.to_dict() for m in self._ephemeral_store.get(session_id, [])]
                if not data:
                    return None
            
            messages = [ChatMessage.from_dict(msg_data) for msg_data in data]
            
            # 시스템 메시지 제외 옵션
            if not include_system:
                messages = [m for m in messages if m.role != MessageRole.SYSTEM]
            
            # 제한 적용 (최근 메시지 우선)
            if limit:
                messages = messages[-limit:]
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get messages for session {session_id}: {e}")
            return None
    
    def get_conversation_context(
        self, 
        session_id: str, 
        max_tokens: int = 4000,
        last_messages_limit: int = 12
    ) -> Tuple[str, int]:
        """
        대화 컨텍스트를 문자열로 반환 (토큰 제한 고려)
        
        Returns:
            (context_string, estimated_tokens)
        """
        messages = self.get_messages(session_id)
        if not messages:
            return "", 0
        
        # 너무 긴 대화는 최근 N개만 사용하여 효율적으로 컨텍스트 구성
        if last_messages_limit and len(messages) > last_messages_limit:
            messages = messages[-last_messages_limit:]
        
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
                # 임시 저장소도 정리
                self._ephemeral_store.pop(session_id, None)
                self._ephemeral_info.pop(session_id, None)
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

    # ------- 세션 메타데이터 (경량 상태 저장소) -------
    def get_session_meta(self, session_id: str) -> Dict[str, Any]:
        """세션 메타데이터 조회 (last_answer/last_sources/last_entities 등)"""
        try:
            data = self.cache.get(self._session_meta_key(session_id))
            if isinstance(data, dict):
                return data
            # 캐시 미사용 시 임시 저장소 폴백
            return dict(self._ephemeral_meta.get(session_id, {}))
        except Exception as e:
            logger.error(f"Failed to get session meta: {e}")
            return dict(self._ephemeral_meta.get(session_id, {}))

    def set_session_meta(self, session_id: str, meta: Dict[str, Any]) -> bool:
        """세션 메타데이터 병합 저장"""
        try:
            current = self.get_session_meta(session_id)
            current.update(meta or {})
            ok = self.cache.set(self._session_meta_key(session_id), current, self.session_ttl)
            if not ok:
                self._ephemeral_meta[session_id] = current
            return True
        except Exception as e:
            logger.error(f"Failed to set session meta: {e}")
            self._ephemeral_meta[session_id] = meta or {}
            return False

    def sync_messages(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        include_system: bool = True,
        merge: bool = True,
    ) -> bool:
        """외부에서 전달된 메시지 히스토리를 세션에 동기화
        - merge=True(기본): 기존 세션과 병합(중복 제거, 순서 보존)하여 초기 대화를 보존
        - merge=False: 전달된 messages로 교체
        """
        try:
            incoming_messages: List[ChatMessage] = []
            for msg in messages:
                role = str(msg.get('role', 'user')).lower()
                content = msg.get('content', '')
                if role == 'system' and not include_system:
                    continue
                try:
                    role_enum = MessageRole(role)
                except Exception:
                    role_enum = MessageRole.USER

                # 원본 메타/타임스탬프 수집 (가능하면 보존)
                created_at = msg.get('created_at') or msg.get('timestamp') or time.time()
                try:
                    created_at = float(created_at)
                except Exception:
                    created_at = time.time()
                meta: Dict[str, Any] = {}
                for k in ['id', 'message_id', 'client_id', 'server_seq', 'client_seq', 'created_at', 'anchor', 'topic_shift', 'action_item']:
                    if k in msg:
                        meta[k] = msg[k]

                incoming_messages.append(
                    ChatMessage(
                        role=role_enum,
                        content=content,
                        timestamp=created_at,
                        metadata=meta
                    )
                )

            if merge:
                # 기존 메시지와 병합 (역사 보존, 중복 제거)
                existing = self.get_messages(session_id) or []
                combined: List[ChatMessage] = list(existing)

                def _dup_key(m: ChatMessage) -> str:
                    # 다중키 중복 제거: (id/client_id) + created_at + sha256(content)
                    mid = None
                    if m.metadata:
                        mid = m.metadata.get('id') or m.metadata.get('message_id') or m.metadata.get('client_id')
                    created = m.metadata.get('created_at') if (m.metadata and 'created_at' in m.metadata) else m.timestamp
                    h = hashlib.sha256((m.content or '').encode()).hexdigest()[:16]
                    return f"{mid}|{created}|{h}"

                seen = {_dup_key(m) for m in combined}
                dedup_skipped = 0
                for m in incoming_messages:
                    key = _dup_key(m)
                    if key not in seen:
                        combined.append(m)
                        seen.add(key)
                    else:
                        dedup_skipped += 1
                chat_messages = combined
            else:
                chat_messages = incoming_messages

            # 정렬: server_seq > created_at > client_seq (오름차순)
            try:
                def _sort_key(m: ChatMessage):
                    meta = m.metadata or {}
                    server_seq = meta.get('server_seq')
                    client_seq = meta.get('client_seq')
                    try:
                        server_seq = int(server_seq) if server_seq is not None else 10**12
                    except Exception:
                        server_seq = 10**12
                    try:
                        client_seq = int(client_seq) if client_seq is not None else 10**12
                    except Exception:
                        client_seq = 10**12
                    created = None
                    if 'created_at' in meta:
                        try:
                            created = float(meta.get('created_at'))
                        except Exception:
                            created = None
                    created = created if created is not None else float(m.timestamp or time.time())
                    return (server_seq, created, client_seq)
                chat_messages.sort(key=_sort_key)
            except Exception:
                pass

            # 시스템 메시지 보존 + LRU 트림 적용
            if len(chat_messages) > self.max_messages_per_session:
                system_msgs = [m for m in chat_messages if m.role == MessageRole.SYSTEM]
                others = [m for m in chat_messages if m.role != MessageRole.SYSTEM]
                keep = self.max_messages_per_session - len(system_msgs)
                others = others[-keep:] if keep > 0 else []
                chat_messages = system_msgs + others
                logger.info(f"Trimmed session {session_id} to {len(chat_messages)} messages after merge={merge}")

            data = [m.to_dict() for m in chat_messages]
            ok = self.cache.set(self._session_key(session_id), data, self.session_ttl)

            # 기존 생성 시간 유지
            info_existing = self.get_session_info(session_id)
            created_at_val = info_existing.created_at if info_existing else time.time()
            info = SessionInfo(
                session_id=session_id,
                created_at=created_at_val,
                last_activity=time.time(),
                message_count=len(chat_messages)
            )
            ok2 = self.cache.set(self._session_info_key(session_id), info.to_dict(), self.session_ttl)
            if not ok or not ok2:
                self._ephemeral_store[session_id] = chat_messages
                self._ephemeral_info[session_id] = info
            # 메트릭 로깅
            try:
                if merge:
                    logger.info(f"metrics.merge_dedup_count={dedup_skipped}")
            except Exception:
                pass
            return True
        except Exception as e:
            logger.error(f"Failed to sync messages for session {session_id}: {e}")
            return False


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
