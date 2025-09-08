"""
Query Rewriter for Adaptive RAG
더 나은 검색을 위한 질문 재작성
"""

import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class QueryRewriter:
    """질문 재작성기"""

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

        system_message = """당신은 사용자의 질문을 벡터 스토어 검색에 최적화된 더 나은 버전으로 개선하는 질문 재작성 전문가입니다.
입력된 질문의 핵심 의미와 의도를 분석하여 개선해주세요.

재작성 가이드라인:
1. 질문을 더 구체적이고 자세하게 만들어주세요
2. 문서 검색에 도움이 될 키워드를 추가해주세요
3. 복잡한 질문은 더 명확하고 집중된 형태로 분해해주세요
4. 원래 의도를 보존하면서도 검색하기 쉽게 만들어주세요
5. 문서에서 나타날 수 있는 동의어나 관련 용어를 사용해주세요
6. **원래 질문이 한국어면 반드시 한국어로, 영어면 영어로 답변하세요**

예시:
원본: "이거 어떻게 써?"
개선: "이 소프트웨어나 도구를 어떻게 사용하고 구현할 수 있나요?"

원본: "가격 정보?"
개선: "가격 세부사항, 비용, 요금 정보는 어떻게 되나요?"
"""

        self.rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    "다음은 원래 질문입니다: \n\n {question} \n\n 더 나은 질문으로 개선해주세요.",
                ),
            ]
        )

        self.question_rewriter = self.rewrite_prompt | self.llm | StrOutputParser()

    def rewrite(self, question: str) -> str:
        """질문을 재작성하여 더 나은 검색 쿼리로 변환"""
        try:
            rewritten_question = self.question_rewriter.invoke({"question": question})
            logger.info(f"Original question: {question}")
            logger.info(f"Rewritten question: {rewritten_question}")
            return rewritten_question.strip()
        except Exception as e:
            logger.error(f"Error rewriting question: {e}")
            return question  # 오류 발생 시 원래 질문 반환

    def transform_query_node(self, state: dict) -> dict:
        """LangGraph 노드용 질문 재작성 함수"""
        logger.info("==== [TRANSFORM QUERY] ====")
        question = state["question"]
        retry_count = state.get("retry_count", 0)

        # 질문 재작성
        better_question = self.rewrite(question)

        # 재시도 횟수 증가
        return {"question": better_question, "retry_count": retry_count + 1}
