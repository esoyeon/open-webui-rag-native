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
import json

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

**이전 답변(있다면):**
{previous_answer}

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

        # 3) AI/정책 관련 (내부 문서 적합)
        vector_keywords = [
            'ai', '인공지능', '정책', 'policy', '산업', 'industry',
            '기술', 'technology', '전략', 'strategy', '투자', 'investment',
            '2024', '한국', 'korea', '미국', 'usa', '중국', 'china'
        ]
        if any(k in q for k in vector_keywords):
            return SearchType.VECTOR

        # 기본값: 하이브리드로 두 소스 모두 시도
        return SearchType.HYBRID

    def _bias_query_with_session_meta(self, question: str, session_id: str) -> str:
        """세션 메타(last_entities)를 활용해 쿼리를 살짝 보강"""
        try:
            meta = self.session_manager.get_session_meta(session_id)
            entities = meta.get('last_entities', {}) if isinstance(meta, dict) else {}
            keywords = entities.get('keywords', []) if isinstance(entities, dict) else []
            if keywords:
                # 중복 방지
                to_add = [kw for kw in keywords if kw.lower() not in question.lower()]
                if to_add:
                    return question + " (관련 키워드: " + ", ".join(to_add[:5]) + ")"
        except Exception:
            pass
        return question

    def _expand_followup_query(self, question: str, session_id: str) -> str:
        """짧고 모호한 후속 질의를 세션 메타와 직전 답변으로 확장
        예: "한국가격은?" → "Galaxy S25 Ultra price in KRW, South Korea"
        """
        q = question.strip()
        # 길이가 짧거나 지시어 위주일 때만 확장 시도
        if len(q) > 20 and not any(k in q.lower() for k in ['krw', '원', '한국']):
            return question

        meta = self.session_manager.get_session_meta(session_id)
        entities = meta.get('last_entities', {}) if isinstance(meta, dict) else {}
        keywords = entities.get('keywords', []) if isinstance(entities, dict) else []
        product_hint = ''
        if keywords:
            # 제품/모델 관련 키워드만 추림
            prios = ['s25 ultra', 's25', 'ultra', 'galaxy', '갤럭시']
            ordered = [k for p in prios for k in keywords if p in k.lower()]
            product_hint = ordered[0] if ordered else keywords[0]
        if not product_hint:
            # 직전 답변에서 간단 추출 (영문 모델 포함 시)
            recent = self.session_manager.get_messages(session_id, limit=4) or []
            prev_ans = next((m.content for m in reversed(recent) if m.role.value == 'assistant'), '')
            for cand in ['Galaxy S25 Ultra', 'Galaxy S25', 'S25 Ultra', 'S25']:
                if cand.lower() in prev_ans.lower():
                    product_hint = cand
                    break

        base = product_hint or ''
        # 통화/지역 힌트
        suffix = ' price in KRW, South Korea'
        # 기존 질문도 포함해 의미 보존
        if base:
            return f"{base}{suffix} ({question})"
        else:
            return f"{question} (in KRW, South Korea)"

    def _llm_route_and_rewrite(self, question: str, session_id: str, default: SearchType) -> Tuple[SearchType, str]:
        """LLM 기반 라우팅/질의 재작성 (일반화된 방식)
        Returns: (search_type, expanded_query)
        """
        try:
            meta = self.session_manager.get_session_meta(session_id)
            entities = meta.get('last_entities', {}) if isinstance(meta, dict) else {}
            prev_msgs = self.session_manager.get_messages(session_id, limit=4) or []
            prev_answer = next((m.content for m in reversed(prev_msgs) if m.role.value == 'assistant'), "")

            prompt = ChatPromptTemplate.from_template(
                """
                You are a routing and query rewriting assistant.
                Given a user question, previous answer (optional), and lightweight entities (optional),
                decide the best search_type among: web | vector | hybrid.
                Then rewrite the user question into a fully self-contained query that disambiguates vague pronouns and context.

                Constraints:
                - Prefer vector for questions clearly answered by internal documents (policy/AI/2024/etc.)
                - Prefer web for real-time/consumer pricing/location/currency/availability and unknown years
                - Use hybrid if uncertain.
                - Output strict JSON only with keys: search_type, expanded_query

                User question: {question}
                Previous answer: {prev_answer}
                Entities (JSON): {entities}
                """
            )

            messages = prompt.format_messages(
                question=question,
                prev_answer=prev_answer,
                entities=json.dumps(entities, ensure_ascii=False)
            )
            raw = self.llm.invoke(messages).content
            data = json.loads(raw)
            st = str(data.get('search_type', default.value)).lower()
            if st not in ['web', 'vector', 'hybrid']:
                st = default.value
            expanded = data.get('expanded_query', question) or question
            return SearchType(st), expanded
        except Exception:
            # 폴백: 기존 바이어스 + 확장
            biased = self._bias_query_with_session_meta(question, session_id)
            if len(question.strip()) <= 10:
                biased = self._expand_followup_query(biased, session_id)
            return default, biased
    
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

    def _extract_entities_simple(self, text: str) -> Dict[str, Any]:
        """경량 엔티티 추출(규칙 기반): 제품/모델/가격/통화"""
        entities: Dict[str, Any] = {}
        try:
            # 가격/통화
            import re
            price_usd = re.findall(r"\$\s?([0-9][0-9,]*\.?[0-9]*)", text)
            price_krw = re.findall(r"([0-9][0-9,]*)\s?원", text)
            if price_usd:
                entities['price_usd'] = price_usd
            if price_krw:
                entities['price_krw'] = price_krw
            # 제품 키워드
            keywords = []
            for kw in ['galaxy', '갤럭시', 'iphone', '아이폰', 'ultra', '울트라', 'plus', '플러스', 's25', 's25 ultra', 's25 plus']:
                if kw.lower() in text.lower():
                    keywords.append(kw)
            if keywords:
                entities['keywords'] = list(set(keywords))
        except Exception:
            pass
        return entities
    
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
            
            # 3. 검색 타입 결정 (컨텍스트 작업이면 기본값 VECTOR로 설정하여 None 방지)
            if is_contextual:
                search_type = SearchType.VECTOR
                expanded_query = question
            else:
                # 1차 규칙 라우팅 후 LLM 기반 재확인/재작성으로 일반화
                initial = force_search_type or self._simple_route_query(question)
                search_type, expanded_query = self._llm_route_and_rewrite(question, session_id, initial)
            logger.info(f"🔍 Search type: {search_type}; expanded={expanded_query != question}")
            
            # 3. 대화 컨텍스트 조회
            # 요약/정리/번역 등 컨텍스트성 작업은 초기 대화까지 포함하도록 더 길게 가져와 누락 방지
            if is_contextual:
                # 동적 버짓팅: 모델 컨텍스트 여유(예: 2k 토큰) 내에서
                # 1) 최신 메시지, 2) 앵커(대화 시작/주제전환/액션아이템) 우선 포함
                # 간단 구현: 더 많은 last_messages_limit를 사용하고, 앵커는 별도로 앞쪽에 붙임
                anchor_msgs = []
                try:
                    full = self.session_manager.get_messages(session_id, include_system=True) or []
                    for m in full:
                        anchor = (m.metadata or {}).get('anchor') or (m.metadata or {}).get('topic_shift') or (m.metadata or {}).get('action_item')
                        if anchor:
                            anchor_msgs.append(m)
                except Exception:
                    pass

                conversation_context, conv_tokens = self.session_manager.get_conversation_context(
                    session_id, max_tokens=1800, last_messages_limit=60
                )
                # 앵커를 앞쪽에 prepend (중복을 피하기 위해 간단히 내용 기준)
                try:
                    anchor_texts = []
                    for am in anchor_msgs[:5]:
                        t = f"{am.role.value}: {am.content}"
                        if t not in conversation_context:
                            anchor_texts.append(t)
                    if anchor_texts:
                        conversation_context = "\n\n".join(anchor_texts) + "\n\n" + conversation_context
                except Exception:
                    pass
            else:
                conversation_context, conv_tokens = self.session_manager.get_conversation_context(
                    session_id, max_tokens=1000, last_messages_limit=12
                )
            
            # 4. 검색 수행 (컨텍스트 작업이면 검색 생략하고 대화 컨텍스트만 사용)
            if is_contextual:
                # 가드3: 최근 assistant 메시지가 없다면 '명시적 안내 + 컨텍스트 없이 처리'로 폴백해 무한 대기 방지
                recent_msgs = self.session_manager.get_messages(session_id, limit=4) or []
                has_recent_assistant = any(m.role.value == 'assistant' for m in recent_msgs)

                if not has_recent_assistant and not force_operation:
                    logger.info("ℹ️ No recent assistant message; proceeding without retrieval and with user guidance")
                    search_results = []
                    # 질문 앞에 안내를 덧붙여 LLM이 상황을 명확히 알도록 함
                    question = (
                        "이전 대화 맥락이 없습니다. 사용자가 제공한 현재 문장만을 대상으로 작업하세요.\n\n" + question
                    )
                else:
                    search_results = []
            else:
                # LLM 재작성 결과 우선 사용, 없으면 세션 엔티티 보강/짧은 후속 확장
                q_use = expanded_query or question
                if q_use == question:
                    q_use = self._bias_query_with_session_meta(q_use, session_id)
                    if len(question.strip()) <= 10:
                        q_use = self._expand_followup_query(q_use, session_id)
                if search_type == SearchType.HYBRID:
                    search_results = self._parallel_search(q_use, [SearchType.VECTOR, SearchType.WEB])
                elif search_type == SearchType.VECTOR:
                    search_results = self._vector_search(q_use)
                    # 벡터 결과가 없으면 자동으로 웹 검색으로 폴백
                    if not search_results and self.web_search_tool:
                        logger.info("ℹ️ No vector results, falling back to web search")
                        search_results = self._web_search(q_use)
                        search_type = SearchType.WEB
                elif search_type == SearchType.WEB:
                    search_results = self._web_search(q_use)
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
            
            # 이전 assistant 답변(있다면) 추출해 후속 질문 품질 개선
            recent_msgs_for_prev = self.session_manager.get_messages(session_id, limit=4) or []
            prev_assistant = next((m.content for m in reversed(recent_msgs_for_prev) if m.role.value == 'assistant'), "")

            # 7. 새 답변 생성 (타임아웃 가드로 무한 대기 방지)
            try:
                answer = await asyncio.wait_for(
                    self._generate_answer_async(
                        question, context, conversation_context, prev_assistant
                    ), timeout=30
                )
            except asyncio.TimeoutError:
                logger.error("LLM generation timeout; returning fallback message")
                answer = "요청이 예상보다 오래 걸립니다. 잠시 후 다시 시도해 주세요."
            
            # 8. 답변 캐싱 및 세션 메타 업데이트
            self.cache.cache_answer(question, answer, context_hash)
            try:
                # 간단 엔티티 추출 및 소스 보존
                entities = self._extract_entities_simple(answer)
                sources_compact = [
                    {
                        'title': s.title,
                        'source': s.source,
                        'score': s.score
                    } for s in (search_results[:3] if search_results else [])
                ]
                self.session_manager.set_session_meta(session_id, {
                    'last_answer': answer,
                    'last_entities': entities,
                    'last_sources': sources_compact
                })
            except Exception as _:
                pass
            
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
        conversation_context: str,
        previous_answer: str
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
                conversation_context,
                previous_answer
            )
            
            return answer
            
        except Exception as e:
            logger.error(f"Async answer generation failed: {e}")
            return "답변 생성 중 오류가 발생했습니다."
    
    def _generate_answer_sync(
        self, 
        question: str, 
        context: str, 
        conversation_context: str,
        previous_answer: str
    ) -> str:
        """동기 답변 생성"""
        try:
            answer = self.rag_chain.invoke({
                "question": question,
                "context": context,
                "conversation_context": conversation_context,
                "previous_answer": previous_answer
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
