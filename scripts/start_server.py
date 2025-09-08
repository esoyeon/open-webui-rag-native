#!/usr/bin/env python3
"""
Web Server Startup Script for LLM Chatbot
Open WebUI 백엔드 서버를 시작하는 스크립트
"""
import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent.parent
open_webui_backend = project_root / "open-webui" / "backend"

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_environment():
    """환경 설정 확인"""
    logger.info("Checking environment...")

    # Python 버전 확인
    python_version = sys.version
    logger.info(f"Python version: {python_version}")

    # 환경변수 확인
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.warning(
            "Some features may not work properly. Please check your .env file."
        )

    # 디렉토리 확인
    if not open_webui_backend.exists():
        logger.error(f"Open WebUI backend directory not found: {open_webui_backend}")
        return False

    logger.info("Environment check completed")
    return True


def install_dependencies():
    """의존성 설치"""
    logger.info("Installing/checking dependencies...")

    try:
        # Open WebUI 백엔드 의존성 설치
        requirements_path = open_webui_backend / "requirements.txt"
        if requirements_path.exists():
            logger.info("Installing Open WebUI backend dependencies...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.warning(
                    f"Some dependencies may have failed to install: {result.stderr}"
                )
            else:
                logger.info("Open WebUI dependencies installed successfully")

        return True

    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        return False


def start_backend_server(host="127.0.0.1", port=8080, dev_mode=False):
    """백엔드 서버 시작"""
    logger.info(f"Starting Open WebUI backend server on {host}:{port}")

    # 환경 변수 설정
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        if dev_mode:
            # 개발 모드 - 자동 리로드 활성화
            cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                "open_webui.main:app",
                "--host",
                host,
                "--port",
                str(port),
                "--reload",
                "--reload-dir",
                str(open_webui_backend / "open_webui"),
            ]
        else:
            # 프로덕션 모드
            cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                "open_webui.main:app",
                "--host",
                host,
                "--port",
                str(port),
            ]

        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(f"Working directory: {open_webui_backend}")

        # 서버 실행
        process = subprocess.Popen(cmd, cwd=str(open_webui_backend), env=env)

        logger.info(f"Server started with PID: {process.pid}")
        logger.info(f"Web interface available at: http://{host}:{port}")
        logger.info("Press Ctrl+C to stop the server")

        # 서버 프로세스 대기
        try:
            process.wait()
        except KeyboardInterrupt:
            logger.info("Stopping server...")
            process.terminate()
            process.wait()
            logger.info("Server stopped")

    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return False

    return True


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Start LLM Chatbot Web Server")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port number (default: 8080)"
    )
    parser.add_argument(
        "--dev", action="store_true", help="Enable development mode with auto-reload"
    )
    parser.add_argument(
        "--skip-deps", action="store_true", help="Skip dependency installation"
    )

    args = parser.parse_args()

    logger.info("LLM Chatbot with Adaptive RAG - Starting Server...")
    logger.info("=" * 60)

    # 환경 확인
    if not check_environment():
        logger.error("Environment check failed")
        return 1

    # 의존성 설치 (선택적)
    if not args.skip_deps:
        if not install_dependencies():
            logger.error("Dependency installation failed")
            return 1

    # 서버 시작
    success = start_backend_server(host=args.host, port=args.port, dev_mode=args.dev)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
