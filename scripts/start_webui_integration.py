#!/usr/bin/env python3
"""
🚀 원클릭 실행 스크립트 - Open WebUI + Adaptive RAG 완전 자동화

이 스크립트는 다음을 자동으로 실행합니다:
1. 환경 검사 (Python, Docker, API 키)
2. 기존 컨테이너 정리
3. Adaptive RAG API 서버 시작 (포트 8000)
4. Open WebUI Docker 컨테이너 실행 (포트 3000)
5. 자동 연결 설정 (OpenAI 호환 API)

실행 모드:
- --mode api: OpenAI 호환 API 서버로 연결 (권장)
- --mode pipelines: Pipelines Plugin Framework로 연결

사용법:
    python scripts/start_webui_integration.py --mode api

완료 후 브라우저에서 http://localhost:3000 접속
"""
import os
import sys
import time
import subprocess
import argparse
from pathlib import Path


def check_docker():
    """Docker 실행 상태 확인"""
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker가 설치되어 있습니다.")
            return True
    except FileNotFoundError:
        pass

    print("❌ Docker가 설치되지 않았거나 실행되지 않습니다.")
    print("   Docker Desktop을 설치하고 실행해주세요.")
    return False


def check_environment():
    """환경 변수 및 가상환경 확인"""
    if not os.environ.get("VIRTUAL_ENV"):
        print("❌ 가상환경이 활성화되지 않았습니다.")
        print("   다음 명령어를 실행해주세요: source .venv/bin/activate")
        return False

    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   export OPENAI_API_KEY='your-api-key' 로 설정해주세요.")

    print("✅ 환경 설정이 완료되었습니다.")
    return True


def stop_existing_containers():
    """기존 컨테이너 정리"""
    containers = ["open-webui", "open-webui-pipelines"]
    for container in containers:
        try:
            subprocess.run(
                ["docker", "rm", "-f", container],
                capture_output=True,
                text=True,
                check=False,
            )
            print(f"🧹 기존 컨테이너 '{container}' 정리 완료")
        except:
            pass


def get_host_ip():
    """호스트 IP 주소 확인"""
    try:
        # macOS/Linux에서 IP 확인
        result = subprocess.run(["ifconfig"], capture_output=True, text=True)
        lines = result.stdout.split("\n")
        for line in lines:
            if "inet 192.168." in line or "inet 10." in line:
                ip = line.split()[1]
                return ip
    except:
        pass

    return "host.docker.internal"


def start_api_server(mode="api"):
    """API 서버 시작"""
    if mode == "api":
        script = "web_api_server.py"
        print("🚀 OpenAI 호환 API 서버를 시작합니다...")
    else:
        script = "pipelines_server.py"
        print("🚀 Pipelines 서버를 시작합니다...")

    try:
        # 백그라운드에서 서버 시작
        process = subprocess.Popen(
            [sys.executable, script], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        time.sleep(3)  # 서버 시작 대기

        if process.poll() is None:
            print(f"✅ {script} 서버가 성공적으로 시작되었습니다.")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ 서버 시작 실패:")
            print(f"   STDOUT: {stdout.decode()}")
            print(f"   STDERR: {stderr.decode()}")
            return None

    except Exception as e:
        print(f"❌ 서버 시작 중 오류: {e}")
        return None


def start_open_webui(mode="api"):
    """Open WebUI Docker 컨테이너 시작"""
    host_ip = get_host_ip()

    if mode == "api":
        api_url = f"http://{host_ip}:8000/v1"
        container_name = "open-webui"
        port = 3000
    else:
        api_url = f"http://{host_ip}:9099"
        container_name = "open-webui-pipelines"
        port = 3001

    print(f"🐳 Open WebUI Docker 컨테이너를 시작합니다...")
    print(f"   API URL: {api_url}")
    print(f"   컨테이너명: {container_name}")

    docker_cmd = [
        "docker",
        "run",
        "-d",
        "-p",
        f"{port}:8080",
        "-e",
        f"OPENAI_API_BASE_URL={api_url}",
        "-e",
        "OPENAI_API_KEY=adaptive-rag-local-key",
        "-v",
        "open-webui:/app/backend/data",
        "--add-host=host.docker.internal:host-gateway",
        "--name",
        container_name,
        "--restart",
        "always",
        "ghcr.io/open-webui/open-webui:main",
    ]

    try:
        result = subprocess.run(docker_cmd, capture_output=True, text=True, check=True)
        print(f"✅ Open WebUI 컨테이너가 시작되었습니다.")
        print(f"🌐 웹 접속: http://localhost:{port}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Docker 컨테이너 시작 실패:")
        print(f"   명령어: {' '.join(docker_cmd)}")
        print(f"   오류: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Open WebUI + Adaptive RAG 통합 실행")
    parser.add_argument(
        "--mode",
        choices=["api", "pipelines"],
        default="api",
        help="실행 모드: api (OpenAI 호환) 또는 pipelines (Pipelines Framework)",
    )
    parser.add_argument("--skip-checks", action="store_true", help="환경 검사 생략")

    args = parser.parse_args()

    print("🎯 Open WebUI + Adaptive RAG 통합 시작")
    print(f"📋 모드: {args.mode.upper()}")
    print("=" * 50)

    # 환경 검사
    if not args.skip_checks:
        if not check_docker():
            return 1
        if not check_environment():
            return 1

    # 기존 컨테이너 정리
    stop_existing_containers()

    # API 서버 시작
    api_process = start_api_server(args.mode)
    if not api_process:
        return 1

    # Open WebUI 시작
    if not start_open_webui(args.mode):
        api_process.terminate()
        return 1

    # 성공 메시지
    port = 3000 if args.mode == "api" else 3001
    api_port = 8000 if args.mode == "api" else 9099

    print("\n🎉 통합 완료!")
    print("=" * 50)
    print(f"🌐 Open WebUI: http://localhost:{port}")
    print(f"📊 API 서버: http://localhost:{api_port}")
    if args.mode == "api":
        print(f"📖 API 문서: http://localhost:{api_port}/docs")
    print("=" * 50)
    print("💡 사용 방법:")
    print("1. 웹 브라우저에서 Open WebUI 접속")
    print("2. 계정 생성 또는 로그인")
    print("3. 채팅에서 질문하여 Adaptive RAG 테스트")
    print("4. Ctrl+C로 중지")

    try:
        # 서버 프로세스 대기
        api_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 서버를 종료합니다...")
        api_process.terminate()
        stop_existing_containers()
        print("✅ 정리 완료")

    return 0


if __name__ == "__main__":
    exit(main())
