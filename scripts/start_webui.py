#!/usr/bin/env python3
"""
Open WebUI Startup Script for Windows
윈도우 환경에서 Open WebUI를 시작하는 스크립트
"""
import os
import sys
import logging
import subprocess
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

def check_rag_server():
    """RAG 서버 연결 확인"""
    import requests
    
    rag_host = os.getenv("RAG_SERVER_HOST", "127.0.0.1")
    rag_port = int(os.getenv("RAG_SERVER_PORT", "8000"))
    rag_url = f"http://{rag_host}:{rag_port}"
    
    try:
        response = requests.get(f"{rag_url}/health", timeout=5)
        if response.status_code == 200:
            logger.info(f"✅ RAG 서버 연결 확인: {rag_url}")
            return True
    except requests.exceptions.RequestException:
        pass
    
    logger.warning(f"⚠️  RAG 서버에 연결할 수 없습니다: {rag_url}")
    logger.warning("   먼저 start_rag_server.bat를 실행해주세요.")
    return False

def setup_webui_environment():
    """Open WebUI 환경 설정"""
    open_webui_backend = project_root / "open-webui" / "backend"
    
    if not open_webui_backend.exists():
        logger.error("❌ Open WebUI 백엔드를 찾을 수 없습니다.")
        logger.error("   open-webui 폴더가 프로젝트 루트에 있는지 확인해주세요.")
        return None
    
    # 환경 변수 설정
    env = os.environ.copy()
    
    # Python 경로 설정
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    
    # RAG API 서버 URL 설정
    rag_host = os.getenv("RAG_SERVER_HOST", "127.0.0.1")
    rag_port = int(os.getenv("RAG_SERVER_PORT", "8000"))
    env["OPENAI_API_BASE_URL"] = f"http://{rag_host}:{rag_port}/v1"
    env["OPENAI_API_KEY"] = "adaptive-rag-local-key"
    
    # Open WebUI 데이터 디렉토리 설정
    data_dir = project_root / "data" / "webui"
    data_dir.mkdir(exist_ok=True)
    env["DATA_DIR"] = str(data_dir)
    
    # 불필요한 모델과 기능 비활성화 (RAG 시스템만 사용)
    env["ENABLE_EVALUATION_ARENA_MODELS"] = "False"  # Arena Model 비활성화
    env["ENABLE_OLLAMA_API"] = "False"              # Ollama API 비활성화
    env["ENABLE_OPENAI_API"] = "True"               # OpenAI 호환 API만 활성화
    env["ENABLE_MODEL_FILTER"] = "True"             # 모델 필터링 활성화
    
    # Open WebUI 자체 RAG 기능 비활성화 (외부 RAG API 사용)
    env["ENABLE_RAG"] = "False"                     # Open WebUI RAG 비활성화
    env["ENABLE_RAG_HYBRID_SEARCH"] = "False"       # 하이브리드 검색 비활성화
    env["ENABLE_RAG_WEB_LOADER"] = "False"          # 웹 로더 비활성화
    
    logger.info("✅ Open WebUI 환경 설정 완료")
    logger.info(f"📁 데이터 디렉토리: {data_dir}")
    logger.info(f"🔗 RAG API URL: {env['OPENAI_API_BASE_URL']}")
    
    return env, open_webui_backend

def install_webui_dependencies(backend_path):
    """Open WebUI 의존성 설치"""
    requirements_path = backend_path / "requirements.txt"
    
    if not requirements_path.exists():
        logger.error("❌ requirements.txt를 찾을 수 없습니다.")
        return False
    
    logger.info("📦 Open WebUI 의존성을 확인하는 중...")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            logger.info("✅ Open WebUI 의존성 설치 완료")
        else:
            logger.warning("⚠️  일부 의존성 설치에 문제가 있을 수 있습니다.")
            if result.stderr:
                logger.warning(f"경고: {result.stderr[:200]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 의존성 설치 중 오류: {e}")
        return False

def check_frontend_build():
    """프론트엔드 빌드 확인"""
    build_dir = project_root / "open-webui" / "build"
    
    if not build_dir.exists():
        logger.warning("⚠️  프론트엔드가 빌드되지 않았습니다.")
        logger.warning("   먼저 다음 명령을 실행해주세요:")
        logger.warning("   python scripts/build_frontend.py")
        return False
    
    logger.info(f"✅ 프론트엔드 빌드 확인: {build_dir}")
    return True

def start_webui_server():
    """Open WebUI 서버 시작"""
    # 환경 설정
    env_result = setup_webui_environment()
    if not env_result:
        return 1
    
    env, backend_path = env_result
    
    # 프론트엔드 빌드 확인
    if not check_frontend_build():
        logger.error("❌ 프론트엔드가 빌드되지 않았습니다.")
        logger.error("   README의 9단계를 참조하여 프론트엔드를 빌드해주세요.")
        return 1
    
    # 의존성 설치
    if not install_webui_dependencies(backend_path):
        logger.error("❌ 의존성 설치 실패")
        return 1
    
    # RAG 서버 확인 (필수는 아님)
    check_rag_server()
    
    # 서버 설정
    host = os.getenv("WEBUI_HOST", "127.0.0.1")
    port = int(os.getenv("WEBUI_PORT", "8080"))
    
    logger.info("🚀 Open WebUI를 시작합니다...")
    logger.info(f"📍 주소: http://{host}:{port}")
    logger.info("🛑 종료하려면 Ctrl+C를 눌러주세요")
    
    try:
        # uvicorn으로 서버 실행
        cmd = [
            sys.executable, "-m", "uvicorn",
            "open_webui.main:app",
            "--host", host,
            "--port", str(port),
            "--reload"
        ]
        
        logger.info(f"실행 명령: {' '.join(cmd)}")
        logger.info(f"작업 디렉토리: {backend_path}")
        
        # 서버 실행
        process = subprocess.run(
            cmd,
            cwd=str(backend_path),
            env=env
        )
        
        return process.returncode
        
    except KeyboardInterrupt:
        logger.info("🛑 사용자가 서버를 종료했습니다.")
        return 0
    except Exception as e:
        logger.error(f"❌ 서버 시작 중 오류 발생: {e}")
        return 1

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🌐 Open WebUI Server for RAG Integration")
    print("=" * 60)
    
    return start_webui_server()

if __name__ == "__main__":
    sys.exit(main())
