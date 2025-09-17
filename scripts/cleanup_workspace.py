#!/usr/bin/env python3
"""
Workspace Cleanup Utility
로그/캐시/빌드 산출물 정리 스크립트 (안전 모드)

사용:
  python scripts/cleanup_workspace.py --dry-run   # 실제 삭제 없이 목록만 출력
  python scripts/cleanup_workspace.py             # 삭제 실행
"""
import os
import sys
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGETS = [
    # Logs
    "*.log",
    "enhanced_rag_errors.log",
    # Caches & temp
    "**/cache/*",
    "**/.cache/*",
    # Vector/DB artifacts (optional: comment out to keep)
    # "data/vector_store/*",
    # "data/enhanced_rag.db",
    # WebUI build artifacts
    "open-webui/build/*",
]


def expand_patterns(base: Path, patterns):
    files = []
    for pat in patterns:
        files.extend(base.glob(pat))
    return files


def remove_path(p: Path):
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    else:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def main():
    dry_run = "--dry-run" in sys.argv
    items = expand_patterns(PROJECT_ROOT, TARGETS)
    items = [p for p in items if p.exists()]

    if not items:
        print("Nothing to clean.")
        return 0

    print(f"Found {len(items)} items to clean:")
    for p in items:
        print(f" - {p}")

    if dry_run:
        print("\nDry run complete. No files removed.")
        return 0

    # Execute removals
    for p in items:
        remove_path(p)

    print("\nCleanup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


