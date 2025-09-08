"""
Document and Answer Graders for Adaptive RAG
검색된 문서의 관련성과 생성된 답변의 품질을 평가
"""

import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class GradeDocuments(BaseModel):
    """문서 평가를 위한 데이터 모델"""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """할루시네이션 체크를 위한 데이터 모델"""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """답변 품질 평가를 위한 데이터 모델"""

    binary_score: str = Field(
        description="Indicate 'yes' or 'no' whether the answer solves the question"
    )


class DocumentGrader:
    """문서 관련성 평가기"""

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.structured_llm_grader = self.llm.with_structured_output(GradeDocuments)

        system_message = """당신은 검색된 문서와 사용자 질문의 관련성을 평가하는 평가자입니다. 
문서가 사용자 질문과 관련된 키워드나 의미적 내용을 포함하고 있다면 관련 있음으로 평가하세요. 
엄격한 테스트일 필요는 없습니다. 목표는 잘못된 검색 결과를 필터링하는 것입니다. 
문서가 질문과 관련이 있는지 여부를 나타내기 위해 'yes' 또는 'no'의 이진 점수를 부여하세요."""

        self.grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    "검색된 문서: \n\n {document} \n\n 사용자 질문: {question}",
                ),
            ]
        )

        self.retrieval_grader = self.grade_prompt | self.structured_llm_grader

    def grade(self, question: str, document: str) -> str:
        """단일 문서 평가"""
        try:
            result = self.retrieval_grader.invoke(
                {"question": question, "document": document}
            )
            return result.binary_score
        except Exception as e:
            logger.error(f"Error grading document: {e}")
            return "no"

    def grade_documents_node(self, state: dict) -> dict:
        """LangGraph 노드용 문서 평가 함수"""
        logger.info("==== [CHECK DOCUMENT RELEVANCE TO QUESTION] ====")
        question = state["question"]
        documents = state["documents"]

        # 각 문서에 대한 관련성 점수 계산
        filtered_docs = []
        for d in documents:
            score = self.grade(question, d.page_content)
            if score == "yes":
                logger.info("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
                continue

        return {"documents": filtered_docs}


class HallucinationGrader:
    """환각(Hallucination) 평가기"""

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.structured_llm_grader = self.llm.with_structured_output(
            GradeHallucinations
        )

        system_message = """당신은 LLM 생성 답변이 검색된 사실들에 근거하고 있는지 평가하는 평가자입니다.
'yes' 또는 'no'의 이진 점수를 부여하세요. 'yes'는 답변이 제공된 사실들에 근거하고 지원받고 있음을 의미합니다."""

        self.hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    "사실들: \n\n {documents} \n\n LLM 생성 답변: {generation}",
                ),
            ]
        )

        self.hallucination_grader = (
            self.hallucination_prompt | self.structured_llm_grader
        )

    def grade(self, documents: List[Document], generation: str) -> str:
        """환각 여부 평가"""
        try:
            result = self.hallucination_grader.invoke(
                {"documents": documents, "generation": generation}
            )
            return result.binary_score
        except Exception as e:
            logger.error(f"Error checking hallucination: {e}")
            return "no"


class AnswerGrader:
    """답변 품질 평가기"""

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.structured_llm_grader = self.llm.with_structured_output(GradeAnswer)

        system_message = """당신은 답변이 질문을 해결하거나 다루고 있는지 평가하는 평가자입니다.
'yes' 또는 'no'의 이진 점수를 부여하세요. 'yes'는 답변이 질문을 해결했음을 의미합니다."""

        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    "사용자 질문: \n\n {question} \n\n LLM 생성 답변: {generation}",
                ),
            ]
        )

        self.answer_grader = self.answer_prompt | self.structured_llm_grader

    def grade(self, question: str, generation: str) -> str:
        """답변 품질 평가"""
        try:
            result = self.answer_grader.invoke(
                {"question": question, "generation": generation}
            )
            return result.binary_score
        except Exception as e:
            logger.error(f"Error grading answer: {e}")
            return "no"


class QualityGrader:
    """통합 품질 평가기"""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.hallucination_grader = HallucinationGrader(model_name)
        self.answer_grader = AnswerGrader(model_name)

    def hallucination_check_node(self, state: dict) -> str:
        """LangGraph 노드용 환각 및 답변 품질 체크"""
        logger.info("==== [CHECK HALLUCINATIONS] ====")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        retry_count = state.get("retry_count", 0)

        # 재시도 횟수가 많은 경우 더 관대한 평가
        if retry_count >= 3:
            logger.info(
                "==== [DECISION: RETRY LIMIT REACHED, ACCEPTING CURRENT GENERATION] ===="
            )
            return "relevant"

        # 환각 평가
        hallucination_score = self.hallucination_grader.grade(documents, generation)

        if hallucination_score == "yes":
            logger.info("==== [DECISION: GENERATION IS GROUNDED IN DOCUMENTS] ====")

            # 답변의 관련성(Relevance) 평가
            logger.info("==== [GRADE GENERATED ANSWER vs QUESTION] ====")
            answer_score = self.answer_grader.grade(question, generation)

            if answer_score == "yes":
                logger.info("==== [DECISION: GENERATED ANSWER ADDRESSES QUESTION] ====")
                return "relevant"
            else:
                logger.info(
                    "==== [DECISION: GENERATED ANSWER DOES NOT ADDRESS QUESTION] ===="
                )
                return "not_relevant"
        else:
            logger.info(
                "==== [DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY] ===="
            )
            return "hallucination"
