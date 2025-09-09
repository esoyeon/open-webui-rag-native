#!/usr/bin/env python3
"""
Open WebUI Frontend Build Script
Open WebUI 프론트엔드를 빌드하는 스크립트
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
open_webui_root = project_root / "open-webui"


# ---------- 유틸: Windows에서 npm(.cmd) 확실히 찾기 ----------
def _resolve_cmd(name: str) -> str:
    """
    크로스 플랫폼에서 실행 파일을 안전하게 찾기.
    Windows에서는 npm(.cmd/.bat/.exe) 확장자 문제를 해결.
    """
    # 먼저 PATH에서 탐색
    found = shutil.which(name)
    if found:
        return found

    if os.name == "nt":  # Windows
        # 확장자 포함 탐색
        for suf in (".cmd", ".bat", ".exe"):
            p = shutil.which(name + suf)
            if p:
                return p

        # 기본 설치 경로 보정 (일반적인 설치 위치)
        candidates = [
            r"C:\Program Files\nodejs\npm.cmd",
            r"C:\Program Files\nodejs\npm.exe",
            r"C:\Program Files (x86)\nodejs\npm.cmd",  # 32-bit 시스템
            r"C:\Program Files (x86)\nodejs\npm.exe",
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
    elif os.name == "posix":  # macOS/Linux
        # Unix 계열에서 추가 경로 확인
        candidates = [
            f"/usr/local/bin/{name}",
            f"/opt/homebrew/bin/{name}",  # Apple Silicon Mac
            f"/usr/bin/{name}",
        ]
        for c in candidates:
            if os.path.exists(c) and os.access(c, os.X_OK):
                return c

    # 마지막 fallback (없는 경우 그대로 반환해서 이후 에러 메시지 유도)
    return name


def _run(cmd, **kwargs) -> subprocess.CompletedProcess:
    """
    공통 실행 래퍼: 에러 내용을 보기 좋게 정리.
    """
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)
    except subprocess.CalledProcessError as e:
        out = (e.stdout or "").strip()
        err = (e.stderr or "").strip()
        msg = f"command: {cmd}\nreturncode: {e.returncode}\nstdout:\n{out}\nstderr:\n{err}"
        raise RuntimeError(msg) from e


def check_nodejs() -> bool:
    """Node.js / npm 설치 및 버전 확인"""
    print("📋 Node.js 설치 확인...")

    # node 확인 (크로스 플랫폼)
    node_path = _resolve_cmd("node")
    if not node_path or not os.path.exists(node_path):
        print("❌ node를 찾을 수 없습니다. https://nodejs.org 에서 LTS 버전을 설치하세요.")
        if os.name == "nt":
            print("   - Windows: Node.js 설치 후 PATH에 추가되었는지 확인하세요.")
        elif os.name == "posix":
            print("   - macOS: brew install node 또는 공식 설치 프로그램 사용")
            print("   - Linux: 패키지 매니저로 설치 (apt, yum, pacman 등)")
        return False

    # node 버전
    try:
        node_version = _run([node_path, "--version"]).stdout.strip()
        print(f"✅ Node.js 버전: {node_version}")
        
        # 최소 버전 체크 (Node.js 16+ 권장)
        version_parts = node_version.lstrip('v').split('.')
        major_version = int(version_parts[0])
        if major_version < 16:
            print(f"⚠️  Node.js {node_version}은 권장 버전(16+)보다 낮습니다.")
            print("   최신 LTS 버전 설치를 권장합니다.")
    except Exception as e:
        print(f"❌ Node.js 버전 확인 실패:\n{e}")
        return False

    # npm 확인
    npm_path = _resolve_cmd("npm")
    if not npm_path or not os.path.exists(npm_path):
        print("❌ npm 실행 파일을 찾을 수 없습니다.")
        print("   - Node.js와 함께 설치되어야 합니다.")
        print("   - PATH 또는 설치 상태를 확인하세요.")
        return False

    # npm 버전
    try:
        npm_version = _run([npm_path, "--version"]).stdout.strip()
        print(f"✅ npm 버전: {npm_version}")
    except Exception as e:
        print("❌ npm 버전 확인 실패.")
        print(f"   상세:\n{e}")
        return False

    return True


def install_dependencies() -> bool:
    """프론트엔드 의존성 설치"""
    print("\n📦 프론트엔드 패키지 설치 중...")
    print("   (이 과정은 몇 분 소요될 수 있습니다)")

    npm_path = _resolve_cmd("npm")

    try:
        # npm install --force (의존성 충돌 해결)
        # 경고가 있더라도 continue 할 수 있도록 returncode 체크는 우리가 직접 함
        result = subprocess.run(
            [npm_path, "install", "--force"],
            cwd=open_webui_root,
            capture_output=True,
            text=True,
        )

        # 출력 정리
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()

        if result.returncode != 0:
            print("⚠️  일부 의존성 설치 에러/경고가 발생했습니다. (계속 진행을 시도합니다)")
            if err:
                print(f"   stderr (앞부분):\n{err[:1000]}")
            if out:
                print(f"   stdout (앞부분):\n{out[:1000]}")
        else:
            # 정상 설치
            print("✅ 프론트엔드 패키지 설치 완료")

        # node_modules 디렉토리 확인
        node_modules = open_webui_root / "node_modules"
        if node_modules.exists():
            print(f"✅ node_modules 디렉토리 생성됨: {node_modules}")
        else:
            print("⚠️  node_modules 디렉토리가 생성되지 않았습니다.")

        return True

    except FileNotFoundError:
        print("❌ npm 실행 파일을 찾지 못했습니다. (PATH 혹은 설치 상태 확인 필요)")
        return False
    except Exception as e:
        print(f"❌ 패키지 설치 중 예외 발생:\n{e}")
        return False


def build_frontend() -> bool:
    """프론트엔드 빌드"""
    print("\n🔨 프론트엔드 빌드 중...")
    print("   (이 과정은 몇 분 소요될 수 있습니다)")

    npm_path = _resolve_cmd("npm")

    try:
        result = subprocess.run(
            [npm_path, "run", "build"],
            cwd=open_webui_root,
            capture_output=True,
            text=True,
        )

        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()

        if result.returncode != 0:
            print("❌ 빌드 실패")
            if err:
                print(f"   stderr:\n{err[:4000]}")
            if out:
                print(f"   stdout:\n{out[:2000]}")
            return False

        # 빌드 결과 확인
        build_dir = open_webui_root / "build"
        if build_dir.exists():
            print(f"✅ 프론트엔드 빌드 완료: {build_dir}")
            
            # 빌드된 파일들 확인
            try:
                files = list(build_dir.iterdir())
                print(f"   생성된 파일/폴더: {len(files)}개")
                
                # 주요 빌드 파일들 확인
                important_files = ["index.html", "assets"]
                for file_name in important_files:
                    file_path = build_dir / file_name
                    if file_path.exists():
                        print(f"   ✅ {file_name} 생성됨")
                    else:
                        print(f"   ⚠️  {file_name} 누락")
                        
            except Exception as e:
                print(f"   ⚠️  빌드 결과 확인 중 오류: {e}")
            
            return True
        else:
            print("❌ 빌드 디렉토리가 생성되지 않았습니다.")
            return False

    except FileNotFoundError:
        print("❌ npm 실행 파일을 찾지 못했습니다. (PATH 혹은 설치 상태 확인 필요)")
        return False
    except Exception as e:
        print(f"❌ 빌드 중 예외 발생:\n{e}")
        return False


def main() -> int:
    """메인 실행 함수"""
    print("=" * 60)
    print("🎨 Open WebUI Frontend Builder")
    print("=" * 60)

    # 1. Node.js 확인
    if not check_nodejs():
        print("\n💡 Node.js/npm 설치 또는 PATH를 점검한 뒤 다시 실행해주세요.")
        return 1

    # 2. package.json 확인
    package_json = open_webui_root / "package.json"
    if not package_json.exists():
        print(f"❌ package.json을 찾을 수 없습니다: {package_json}")
        print("   Open WebUI 프로젝트가 올바르게 클론되었는지 확인해주세요.")
        return 1

    # 3. 기존 빌드 확인
    build_dir = open_webui_root / "build"
    if build_dir.exists():
        try:
            response = input("\n기존 빌드가 있습니다. 다시 빌드하시겠습니까? (y/N): ")
        except EOFError:
            response = "n"
        if response.lower() != 'y':
            print("✅ 기존 빌드를 사용합니다.")
            return 0
        else:
            print("🧹 기존 빌드를 삭제합니다...")
            shutil.rmtree(build_dir, ignore_errors=True)

    # 4. 의존성 설치
    if not install_dependencies():
        print("\n❌ 의존성 설치 실패")
        return 1

    # 5. 프론트엔드 빌드
    if not build_frontend():
        print("\n❌ 프론트엔드 빌드 실패")
        return 1

    print("\n🎉 프론트엔드 빌드가 성공적으로 완료되었습니다!")
    print("=" * 60)
    print("📋 다음 단계:")
    print("1. RAG 서버 시작: python scripts/start_rag_server.py")
    print("2. Open WebUI 시작: python scripts/start_webui.py")
    print("3. 브라우저에서 http://localhost:8080 접속")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
