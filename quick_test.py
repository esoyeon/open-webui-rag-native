#!/usr/bin/env python3
"""
빠른 파이프라인 테스트
"""
import os
import sys

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from pipelines.adaptive_rag_pipeline import Pipe


def main():
    print("🚀 Adaptive RAG Pipeline 빠른 테스트")
    print("=" * 50)

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        print("📝 .env 파일에 OPENAI_API_KEY를 설정해주세요.")
        return

    try:
        # 파이프라인 초기화
        print("🔧 파이프라인 초기화 중...")
        pipeline = Pipe()

        # 상태 확인
        status = pipeline.get_status()
        print(f"✅ 파이프라인 상태: {'성공' if status.get('initialized') else '실패'}")
        print(f"📊 문서 수: {status.get('total_documents', 0)}개")

        # 간단한 질문들
        questions = [
            "안녕하세요!",
            "삼성전자 생성형 AI에 대해 알려주세요",
        ]

        print("\n💬 질문 테스트:")
        for i, question in enumerate(questions, 1):
            print(f"\n[Q{i}] {question}")
            print("-" * 30)

            response = pipeline.pipe(
                user_message=question, model_id="gpt-3.5-turbo", messages=[], body={}
            )

            print(f"[A{i}] {response}")

        print("\n✅ 테스트 완료!")

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        print("💡 문제 해결 방법:")
        print("1. .env 파일에 OPENAI_API_KEY 설정")
        print("2. 가상환경 활성화: source .venv/bin/activate")
        print("3. 의존성 설치: uv pip install -e .")


if __name__ == "__main__":
    main()
