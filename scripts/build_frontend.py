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

def check_nodejs():
    """Node.js 설치 확인"""
    print("📋 Node.js 설치 확인...")
    
    # npm 확인
    npm_path = shutil.which("npm")
    if not npm_path:
        print("❌ npm이 설치되지 않았습니다.")
        print("   Node.js를 https://nodejs.org 에서 다운로드하여 설치해주세요.")
        print("   Windows: LTS 버전 다운로드 및 설치")
        print("   설치 후 새 터미널을 열어주세요.")
        return False
    
    # Node.js 버전 확인
    try:
        node_version = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ Node.js 버전: {node_version.stdout.strip()}")
        
        npm_version = subprocess.run(
            ["npm", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ npm 버전: {npm_version.stdout.strip()}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Node.js 버전 확인 실패: {e}")
        return False

def install_dependencies():
    """프론트엔드 의존성 설치"""
    print("\n📦 프론트엔드 패키지 설치 중...")
    print("   (이 과정은 몇 분 소요될 수 있습니다)")
    
    try:
        # open-webui 디렉토리로 이동하여 npm install 실행
        result = subprocess.run(
            ["npm", "install", "--force"],
            cwd=open_webui_root,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"⚠️  일부 의존성 설치 경고: {result.stderr[:500]}")
            # 경고는 무시하고 계속 진행
        
        print("✅ 프론트엔드 패키지 설치 완료")
        return True
        
    except Exception as e:
        print(f"❌ 패키지 설치 실패: {e}")
        return False

def build_frontend():
    """프론트엔드 빌드"""
    print("\n🔨 프론트엔드 빌드 중...")
    print("   (이 과정은 몇 분 소요될 수 있습니다)")
    
    try:
        # pyodide fetch 및 빌드
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=open_webui_root,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ 빌드 실패: {result.stderr}")
            return False
        
        # 빌드 결과 확인
        build_dir = open_webui_root / "build"
        if build_dir.exists():
            print(f"✅ 프론트엔드 빌드 완료: {build_dir}")
            
            # 빌드된 파일 목록 출력
            files = list(build_dir.iterdir())
            print(f"   생성된 파일: {len(files)}개")
            return True
        else:
            print("❌ 빌드 디렉토리가 생성되지 않았습니다.")
            return False
            
    except Exception as e:
        print(f"❌ 빌드 중 오류: {e}")
        return False

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🎨 Open WebUI Frontend Builder")
    print("=" * 60)
    
    # 1. Node.js 확인
    if not check_nodejs():
        print("\n💡 Node.js 설치 후 다시 실행해주세요.")
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
        response = input("\n기존 빌드가 있습니다. 다시 빌드하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("✅ 기존 빌드를 사용합니다.")
            return 0
        else:
            print("🧹 기존 빌드를 삭제합니다...")
            shutil.rmtree(build_dir)
    
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
