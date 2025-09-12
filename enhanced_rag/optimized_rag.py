"""
Optimized RAG Engine
현업 패턴을 적용한 고성능 RAG 시스템

Key Improvements:
- 캐싱으로 중복 계산 제거
- 병렬 검색 처리
- 간소화된 파이프라인
- Circuit breaker 패턴
- 토큰 최적화
"""

import asyncio
import re
import hashlib
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .cache_manager import RAGCacheManager, get_cache_manager
from .session_manager import SessionManager, MessageRole, get_session_manager

logger = logging.getLogger(__name__)


class SearchType(str, Enum):
    VECTOR = "vector"
    WEB = "web"
    HYBRID = "hybrid"


@dataclass
class SearchResult:
    """검색 결과 모델"""
    content: str
    source: str
    score: float
    title: str = ""
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'source': self.source, 
            'score': self.score,
            'title': self.title,
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_document(cls, doc: Document) -> 'SearchResult':
        """Langchain Document에서 SearchResult 생성"""
        metadata = doc.metadata or {}
        return cls(
            content=doc.page_content,
            source=metadata.get('source', 'unknown'),
            score=metadata.get('score', 0.0),
            title=metadata.get('title', ''),
            metadata=metadata
        )


@dataclass
class RAGResponse:
    """RAG 응답 모델"""
    answer: str
    sources: List[SearchResult]
    search_type: SearchType
    response_time: float
    cached: bool = False
    tokens_used: int = 0


class OptimizedRAGEngine:
    """
    최적화된 RAG 엔진
    
    Features:
    - Multi-level caching (embedding, search, answer)
    - Parallel search execution
    - Smart routing (simple keyword-based)
    - Circuit breaker for external services
    - Token optimization
    """
    
    def __init__(
        self,
        vector_store=None,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0,
        max_search_results: int = 8,
        max_context_tokens: int = 3000
    ):
        self.vector_store = vector_store
        self.model_name = model_name
        self.temperature = temperature
        self.max_search_results = max_search_results
        self.max_context_tokens = max_context_tokens
        
        # Core components
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.cache = get_cache_manager()
        self.session_manager = get_session_manager()
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Web search (optional)
        self.web_search_tool = None
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            self.web_search_tool = TavilySearchResults(
                max_results=3,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True,
                include_images=False
            )
            logger.info("✅ Web search tool initialized")
        except Exception as e:
            logger.warning(f"⚠️ Web search not available: {e}")
        
        # Optimized prompt
        self.rag_prompt = self._create_optimized_prompt()
        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()
        
        logger.info(f"OptimizedRAGEngine initialized with model: {model_name}")
    
    def _create_optimized_prompt(self) -> ChatPromptTemplate:
        """최적화된 프롬프트 생성"""
        template = """당신은 전문 AI 연구 분석가입니다. 주어진 정보를 바탕으로 정확하고 유용한 답변을 제공해주세요.

**답변 가이드라인:**
1. 📋 **핵심 내용**: 질문에 직접적으로 답변하는 핵심 정보
2. 🔍 **상세 설명**: 배경과 맥락이 필요한 경우 추가 설명
3. 📊 **데이터/수치**: 구체적인 수치나 데이터가 있다면 포함
4. 💡 **결론**: 요약과 시사점

**중요사항:**
- 문서에 없는 내용은 추측하지 말고 "문서에서 확인되지 않음"이라고 명시
- 질문 언어에 맞춰 답변 (한국어 질문→한국어 답변)
- 간결하면서도 정확한 답변 제공

**대화 맥락:**
{conversation_context}

**검색된 정보:**
{context}

**질문:** {question}

**답변:**"""
        
        return ChatPromptTemplate.from_template(template)
    
    def _simple_route_query(self, question: str) -> SearchType:
        """간단한 키워드/규칙 기반 쿼리 라우팅 (현업 지향)
        - 최신/실시간/가격/출시 관련 → WEB
        - 연도가 2025 이상이거나 2024가 아닌 명시적 연도 → WEB
        - 그 외 AI/정책/2024 관련 → VECTOR
        """
        q = question.lower()

        # 1) 연도 규칙: 2025년 이후, 혹은 2024가 아닌 명시적 연도면 웹
        year_match = re.search(r"(20\d{2})", q)
        if year_match:
            year = int(year_match.group(1))
            if year >= 2025 or year != 2024:
                return SearchType.WEB

        # 2) 최신/시간 관련 키워드 → 웹
        realtime_keywords = [
            '오늘', 'today', '어제', 'yesterday', '최신', 'latest',
            '실시간', 'realtime', '지금', 'now', '현재', 'current',
            '뉴스', 'news', '속보', 'breaking',
            '이번주', '이번 주', '지난주', '지난 주', '이번달', '이번 달', '지난달', '지난 달',
            '분기', '1분기', '2분기', '3분기', '4분기',
            '1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월',
            'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'
        ]
        if any(k in q for k in realtime_keywords):
            return SearchType.WEB

        # 3) 소비자 제품/가격/출시 관련 키워드 → 웹
        product_price_keywords = [
            '아이폰', 'iphone', '갤럭시', 'galaxy', '맥북', 'macbook', '애플', 'apple', '삼성', 'samsung',
            '가격', 'price', '출시', '발표', '발매', '런칭', 'official', 'event', '언제', '얼마'
        ]
        if any(k in q for k in product_price_keywords):
            return SearchType.WEB

        # 4) AI/정책 관련 (내부 문서 적합)
        vector_keywords = [
            'ai', '인공지능', '정책', 'policy', '산업', 'industry',
            '기술', 'technology', '전략', 'strategy', '투자', 'investment',
            '2024', '한국', 'korea', '미국', 'usa', '중국', 'china'
        ]
        if any(k in q for k in vector_keywords):
            return SearchType.VECTOR

        # 기본값: 하이브리드로 두 소스 모두 시도
        return SearchType.HYBRID
    
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """캐시된 임베딩 조회 또는 생성"""
        embedding = self.cache.get_cached_embedding(text)
        if embedding is not None:
            return embedding
        
        try:
            # 새로 생성하고 캐시
            new_embedding = np.array(self.embeddings.embed_query(text))
            self.cache.cache_embedding(text, new_embedding)
            return new_embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def _vector_search(self, question: str) -> List[SearchResult]:
        """벡터 검색 수행"""
        if not self.vector_store:
            logger.warning("Vector store not available")
            return []
        
        # 캐시 확인
        cached_results = self.cache.get_cached_search_results(question, "vector")
        if cached_results:
            logger.info("✅ Using cached vector search results")
            return [SearchResult(**result) for result in cached_results]
        
        try:
            # 벡터 검색 수행
            documents = self.vector_store.similarity_search(
                question, 
                k=self.max_search_results
            )
            
            results = [SearchResult.from_document(doc) for doc in documents]
            
            # 결과 캐싱
            results_dict = [result.to_dict() for result in results]
            self.cache.cache_search_results(question, results_dict, "vector")
            
            logger.info(f"✅ Vector search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _web_search(self, question: str) -> List[SearchResult]:
        """웹 검색 수행"""
        if not self.web_search_tool:
            logger.warning("Web search not available")
            return []
        
        # 캐시 확인
        cached_results = self.cache.get_cached_search_results(question, "web")
        if cached_results:
            logger.info("✅ Using cached web search results")
            return [SearchResult(**result) for result in cached_results]
        
        try:
            # 웹 검색 수행
            web_results = self.web_search_tool.invoke({"query": question})
            
            results = []
            for result in web_results[:self.max_search_results]:
                search_result = SearchResult(
                    content=result.get("content", ""),
                    source=result.get("url", ""),
                    score=0.8,  # 웹 검색은 관련성이 높다고 가정
                    title=result.get("title", ""),
                    metadata={"search_type": "web"}
                )
                results.append(search_result)
            
            # 결과 캐싱
            results_dict = [result.to_dict() for result in results]
            self.cache.cache_search_results(question, results_dict, "web")
            
            logger.info(f"✅ Web search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    def _parallel_search(self, question: str, search_types: List[SearchType]) -> List[SearchResult]:
        """병렬 검색 실행"""
        futures = []
        
        for search_type in search_types:
            if search_type == SearchType.VECTOR:
                future = self.executor.submit(self._vector_search, question)
                futures.append(future)
            elif search_type == SearchType.WEB:
                future = self.executor.submit(self._web_search, question)
                futures.append(future)
        
        # 결과 수집
        all_results = []
        for future in as_completed(futures, timeout=30):  # 30초 타임아웃
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Search operation failed: {e}")
        
        # 점수순 정렬 및 중복 제거
        unique_results = {}
        for result in all_results:
            key = hashlib.md5(result.content.encode()).hexdigest()[:16]
            if key not in unique_results or result.score > unique_results[key].score:
                unique_results[key] = result
        
        sorted_results = sorted(unique_results.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:self.max_search_results]
    
    def _format_context(self, results: List[SearchResult]) -> str:
        """검색 결과를 컨텍스트로 포맷팅"""
        if not results:
            return "관련 정보를 찾을 수 없습니다."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.title or result.source
            content = result.content[:500]  # 토큰 절약을 위해 제한
            context_parts.append(f"[문서 {i}] {source}\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _generate_context_hash(self, context: str) -> str:
        """컨텍스트 해시 생성 (답변 캐싱용)"""
        return hashlib.md5(context.encode()).hexdigest()[:16]
    
    def _is_contextual_operation(self, question: str) -> bool:
        """질문이 이전 대화 내용에 대한 작업(번역/요약/정리 등)인지 감지"""
        q = question.lower()
        patterns = [
            '번역', 'translation', 'translate',
            '요약', 'summary', 'summarize', '정리',
            '위 내용', '이 내용', '방금', 'previous', 'above', 'the content', '그 내용', '그걸'
        ]
        return any(p in q for p in patterns)

    async def process_question(
        self, 
        question: str, 
        session_id: str,
        force_search_type: Optional[SearchType] = None,
        force_operation: Optional[str] = None
    ) -> RAGResponse:
        """
        질문 처리 메인 함수
        
        Args:
            question: 사용자 질문
            session_id: 세션 ID
            force_search_type: 강제 검색 타입 (선택적)
        
        Returns:
            RAGResponse 객체
        """
        start_time = time.time()
        
        try:
            # 1. 세션에 사용자 질문 추가
            self.session_manager.add_message(
                session_id, MessageRole.USER, question
            )
            
            # 2. 컨텍스트성 작업(번역/요약/위 내용 등) 여부 판단
            # 가드1: 사용자가 명시적으로 operation을 지정하면 규칙보다 우선
            if force_operation:
                is_contextual = force_operation in [
                    'context', 'translate', 'summarize', 'rewrite'
                ]
            else:
                is_contextual = self._is_contextual_operation(question)
            
            # 3. 검색 타입 결정 (컨텍스트 작업이면 검색 생략)
            search_type = None
            if not is_contextual:
                search_type = force_search_type or self._simple_route_query(question)
                logger.info(f"🔍 Search type: {search_type}")
            
            # 3. 대화 컨텍스트 조회 (최근성 제한 포함)
            conversation_context, conv_tokens = self.session_manager.get_conversation_context(
                session_id, max_tokens=1000
            )
            
            # 4. 검색 수행 (컨텍스트 작업이면 검색 생략하고 대화 컨텍스트만 사용)
            if is_contextual:
                # 가드3: 최근 assistant 메시지가 없다면 검색 기반으로 폴백
                recent_msgs = self.session_manager.get_messages(session_id, limit=4) or []
                has_recent_assistant = any(m.role.value == 'assistant' for m in recent_msgs)

                if not has_recent_assistant and not force_operation:
                    logger.info("ℹ️ No recent assistant message; falling back to retrieval for contextual request")
                    is_contextual = False
                else:
                    search_results = []
            else:
                if search_type == SearchType.HYBRID:
                    search_results = self._parallel_search(question, [SearchType.VECTOR, SearchType.WEB])
                elif search_type == SearchType.VECTOR:
                    search_results = self._vector_search(question)
                    # 벡터 결과가 없으면 자동으로 웹 검색으로 폴백
                    if not search_results and self.web_search_tool:
                        logger.info("ℹ️ No vector results, falling back to web search")
                        search_results = self._web_search(question)
                        search_type = SearchType.WEB
                elif search_type == SearchType.WEB:
                    search_results = self._web_search(question)
                else:
                    search_results = []
            
            # 5. 컨텍스트 생성
            # 컨텍스트성 작업일 때는 검색 컨텍스트를 비워 LLM이 대화 맥락에만 집중하도록 함
            context = self._format_context(search_results) if not is_contextual else ""
            context_hash = self._generate_context_hash(context)
            
            # 6. 캐시된 답변 확인
            cached_answer = self.cache.get_cached_answer(question, context_hash)
            if cached_answer:
                logger.info("✅ Using cached answer")
                
                # 세션에 캐시된 답변 추가
                self.session_manager.add_message(
                    session_id, MessageRole.ASSISTANT, cached_answer
                )
                
                response_time = time.time() - start_time
                return RAGResponse(
                    answer=cached_answer,
                    sources=search_results,
                    search_type=search_type,
                    response_time=response_time,
                    cached=True
                )
            
            # 7. 새 답변 생성
            answer = await self._generate_answer_async(
                question, context, conversation_context
            )
            
            # 8. 답변 캐싱
            self.cache.cache_answer(question, answer, context_hash)
            
            # 9. 세션에 답변 추가
            self.session_manager.add_message(
                session_id, MessageRole.ASSISTANT, answer
            )
            
            response_time = time.time() - start_time
            
            logger.info(f"✅ RAG completed in {response_time:.2f}s")
            
            return RAGResponse(
                answer=answer,
                sources=search_results,
                search_type=search_type,
                response_time=response_time,
                cached=False,
                tokens_used=len(answer.split()) + conv_tokens  # 대략적인 토큰 수
            )
            
        except Exception as e:
            logger.error(f"RAG processing failed: {e}")
            error_answer = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
            
            response_time = time.time() - start_time
            return RAGResponse(
                answer=error_answer,
                sources=[],
                search_type=search_type,
                response_time=response_time,
                cached=False
            )
    
    async def _generate_answer_async(
        self, 
        question: str, 
        context: str, 
        conversation_context: str
    ) -> str:
        """비동기 답변 생성"""
        try:
            loop = asyncio.get_event_loop()
            
            # LLM 호출을 별도 스레드에서 실행
            answer = await loop.run_in_executor(
                self.executor,
                self._generate_answer_sync,
                question,
                context,
                conversation_context
            )
            
            return answer
            
        except Exception as e:
            logger.error(f"Async answer generation failed: {e}")
            return "답변 생성 중 오류가 발생했습니다."
    
    def _generate_answer_sync(
        self, 
        question: str, 
        context: str, 
        conversation_context: str
    ) -> str:
        """동기 답변 생성"""
        try:
            answer = self.rag_chain.invoke({
                "question": question,
                "context": context,
                "conversation_context": conversation_context
            })
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "답변 생성 중 오류가 발생했습니다."
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """엔진 통계 조회"""
        cache_health = self.cache.get_health()
        session_stats = self.session_manager.get_session_stats()
        
        return {
            'model_name': self.model_name,
            'cache_health': cache_health,
            'session_stats': session_stats,
            'vector_store_available': self.vector_store is not None,
            'web_search_available': self.web_search_tool is not None
        }
