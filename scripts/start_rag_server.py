#!/usr/bin/env python3
"""
RAG Server Startup Script for Windows
윈도우 환경에서 RAG API 서버를 시작하는 스크립트
"""
import os
import sys
import logging
import uvicorn
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def check_environment():
    """환경 설정 확인"""
    logger.info("환경 설정을 확인하는 중...")
    
    # API 키 확인
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key == "your_openai_api_key_here":
        logger.error("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        logger.error("   .env 파일에서 OPENAI_API_KEY를 설정해주세요.")
        return False
    
    logger.info("✅ OPENAI_API_KEY 확인 완료")
    
    # 벡터 스토어 확인
    vector_store_path = project_root / "data" / "vector_store"
    if not vector_store_path.exists() or not (vector_store_path / "faiss.index").exists():
        logger.warning("⚠️  벡터 스토어가 없습니다.")
        logger.warning("   먼저 index_documents.bat를 실행하여 문서를 인덱싱해주세요.")
        
        # 벡터 스토어가 없어도 서버는 시작할 수 있도록 허용
        return True
    
    logger.info("✅ 벡터 스토어 확인 완료")
    return True

def start_server():
    """RAG 서버 시작"""
    # 환경 설정 확인
    if not check_environment():
        return 1
    
    # 서버 설정
    host = os.getenv("RAG_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("RAG_SERVER_PORT", "8000"))
    
    logger.info("🚀 RAG API 서버를 시작합니다...")
    logger.info(f"📍 주소: http://{host}:{port}")
    logger.info(f"📖 API 문서: http://{host}:{port}/docs")
    logger.info("🛑 종료하려면 Ctrl+C를 눌러주세요")
    
    try:
        # web_api_server.py의 FastAPI 앱을 import
        from web_api_server import app
        
        # 서버 실행
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=False,  # 프로덕션에서는 reload 비활성화
            log_level="info"
        )
        
    except ImportError:
        logger.error("❌ web_api_server.py를 찾을 수 없습니다.")
        logger.error("   프로젝트 루트 디렉토리에서 실행해주세요.")
        return 1
    except Exception as e:
        logger.error(f"❌ 서버 시작 중 오류 발생: {e}")
        return 1
    
    return 0

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🚀 RAG API Server for Open WebUI Integration")
    print("=" * 60)
    
    return start_server()

if __name__ == "__main__":
    sys.exit(main())
