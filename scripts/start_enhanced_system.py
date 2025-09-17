#!/usr/bin/env python3
"""
Enhanced RAG System Startup Script
시스템 구성 요소들을 순서대로 시작하고 상태를 확인합니다.

실행 순서:
1. Redis 서버 확인/시작
2. RQ Worker 시작 (백그라운드)
3. Enhanced API 서버 시작
4. 시스템 상태 확인

사용법:
    python scripts/start_enhanced_system.py [--check-only] [--port 8000]
"""

import os
import sys
import subprocess
import time
import argparse
import logging
import signal
from typing import Dict, Any, List
import psutil
import requests

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemManager:
    """Enhanced RAG 시스템 관리자"""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.project_root = project_root
        self.processes = {}
        self.redis_pid = None
        
        # 가상환경 확인
        self.venv_python = self._find_venv_python()
        
    def _find_venv_python(self) -> str:
        """venv Python 경로 찾기"""
        venv_paths = [
            os.path.join(self.project_root, "venv", "bin", "python"),
            os.path.join(self.project_root, "venv", "Scripts", "python.exe"),  # Windows
            sys.executable  # fallback
        ]
        
        for path in venv_paths:
            if os.path.exists(path):
                logger.info(f"✅ Found venv Python: {path}")
                return path
        
        logger.warning("⚠️ venv not found, using system Python")
        return sys.executable
    
    def check_redis(self) -> bool:
        """Redis 서버 상태 확인"""
        try:
            import redis
            client = redis.Redis(host='localhost', port=6379, db=0)
            client.ping()
            logger.info("✅ Redis server is running")
            return True
        except Exception as e:
            logger.error(f"❌ Redis not available: {e}")
            return False
    
    def start_redis(self) -> bool:
        """Redis 서버 시작"""
        if self.check_redis():
            return True
            
        logger.info("🚀 Starting Redis server...")
        
        try:
            # Redis 설정 파일 생성 (옵션)
            redis_conf = os.path.join(self.project_root, "redis.conf")
            if not os.path.exists(redis_conf):
                self._create_redis_config(redis_conf)
            
            # Redis 서버 시작
            redis_cmd = ["redis-server", redis_conf]
            process = subprocess.Popen(
                redis_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.processes['redis'] = process
            
            # 시작 대기
            for i in range(10):
                time.sleep(1)
                if self.check_redis():
                    logger.info("✅ Redis server started successfully")
                    return True
            
            logger.error("❌ Redis server failed to start")
            return False
            
        except FileNotFoundError:
            logger.error("❌ redis-server command not found. Please install Redis.")
            logger.info("   macOS: brew install redis")
            logger.info("   Ubuntu: sudo apt install redis-server")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to start Redis: {e}")
            return False
    
    def _create_redis_config(self, config_path: str):
        """간단한 Redis 설정 파일 생성"""
        config_content = """
# Enhanced RAG Redis Configuration
port 6379
bind 127.0.0.1
daemonize no
dir ./
logfile ""
loglevel notice
save 900 1
save 300 10 
save 60 10000
"""
        with open(config_path, 'w') as f:
            f.write(config_content.strip())
        logger.info(f"📝 Created Redis config: {config_path}")
    
    def start_rq_worker(self) -> bool:
        """RQ Worker 시작"""
        if not self.check_redis():
            logger.error("❌ Redis not available for RQ worker")
            return False
        
        logger.info("🚀 Starting RQ worker...")
        
        try:
            worker_cmd = [
                self.venv_python, 
                "-c", 
                "from enhanced_rag.task_queue import start_worker; start_worker(['high', 'default', 'low'])"
            ]
            
            process = subprocess.Popen(
                worker_cmd,
                cwd=self.project_root,
                stdout=None,
                stderr=None
            )
            
            self.processes['rq_worker'] = process
            
            # 잠시 대기 (워커 시작 확인)
            time.sleep(2)
            
            if process.poll() is None:
                logger.info("✅ RQ worker started successfully")
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"❌ RQ worker failed to start: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start RQ worker: {e}")
            return False
    
    def start_api_server(self) -> bool:
        """Enhanced API 서버 시작"""
        logger.info(f"🚀 Starting Enhanced API server on port {self.port}...")
        
        try:
            server_cmd = [
                self.venv_python, 
                "enhanced_api_server.py"
            ]
            
            # 환경 변수 설정
            env = os.environ.copy()
            env['PYTHONPATH'] = self.project_root
            
            process = subprocess.Popen(
                server_cmd,
                cwd=self.project_root,
                env=env,
                stdout=None,
                stderr=None
            )
            
            self.processes['api_server'] = process
            
            # 서버 시작 대기 및 확인
            for i in range(30):
                time.sleep(1)
                try:
                    response = requests.get(f"http://localhost:{self.port}/health", timeout=5)
                    if response.status_code == 200:
                        logger.info(f"✅ API server started successfully on port {self.port}")
                        return True
                except requests.exceptions.RequestException:
                    pass
            
            logger.error("❌ API server failed to start or respond")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to start API server: {e}")
            return False
    
    def check_system_health(self) -> Dict[str, Any]:
        """전체 시스템 상태 확인"""
        logger.info("🔍 Checking system health...")
        
        health = {
            'redis': False,
            'rq_worker': False,
            'api_server': False,
            'overall': False
        }
        
        # Redis 확인
        health['redis'] = self.check_redis()
        
        # RQ worker 확인
        rq_process = self.processes.get('rq_worker')
        health['rq_worker'] = rq_process is not None and rq_process.poll() is None
        
        # API 서버 확인
        try:
            response = requests.get(f"http://localhost:{self.port}/health", timeout=10)
            if response.status_code == 200:
                health['api_server'] = True
                health['api_details'] = response.json()
        except:
            health['api_server'] = False
        
        # 전체 상태
        health['overall'] = all([
            health['redis'],
            health['rq_worker'],
            health['api_server']
        ])
        
        return health
    
    def print_system_status(self, health: Dict[str, Any]):
        """시스템 상태 출력"""
        print("\n" + "="*60)
        print("🚀 ENHANCED RAG SYSTEM STATUS")
        print("="*60)
        
        status_icon = "✅" if health['overall'] else "❌"
        print(f"Overall Status: {status_icon} {'HEALTHY' if health['overall'] else 'DEGRADED'}")
        print()
        
        # 개별 구성 요소 상태
        components = [
            ("Redis Server", health['redis']),
            ("RQ Worker", health['rq_worker']),
            ("API Server", health['api_server'])
        ]
        
        for name, status in components:
            icon = "✅" if status else "❌"
            print(f"  {name:15} {icon}")
        
        print()
        
        if health['overall']:
            print("🌐 API Endpoints:")
            print(f"   Health Check: http://localhost:{self.port}/health")
            print(f"   API Docs:     http://localhost:{self.port}/docs")
            print(f"   Chat API:     http://localhost:{self.port}/v1/chat/completions")
            print()
            
            if 'api_details' in health:
                api_health = health['api_details']
                performance = api_health.get('performance', {})
                print("📊 Performance Metrics:")
                print(f"   Uptime:        {performance.get('uptime_seconds', 0):.0f} seconds")
                print(f"   Active Sessions: {performance.get('active_sessions', 0)}")
                print(f"   Memory Usage:  {performance.get('memory_usage', 'unknown')}")
        
        print("="*60)
    
    def shutdown(self):
        """시스템 종료"""
        logger.info("🛑 Shutting down Enhanced RAG system...")
        
        # 프로세스들 종료
        for name, process in self.processes.items():
            if process and process.poll() is None:
                logger.info(f"Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        logger.info("✅ System shutdown complete")
    
    def start_all(self) -> bool:
        """전체 시스템 시작"""
        logger.info("🚀 Starting Enhanced RAG System...")
        
        # 1. Redis 시작
        if not self.start_redis():
            return False
        
        # 2. RQ Worker 시작
        if not self.start_rq_worker():
            return False
        
        # 3. API 서버 시작
        if not self.start_api_server():
            return False
        
        # 4. 상태 확인
        health = self.check_system_health()
        self.print_system_status(health)
        
        return health['overall']


def main():
    parser = argparse.ArgumentParser(description="Enhanced RAG System Manager")
    parser.add_argument("--check-only", action="store_true", help="Only check system status")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--stop", action="store_true", help="Stop running system")
    
    args = parser.parse_args()
    
    manager = SystemManager(port=args.port)
    
    if args.check_only:
        # 상태 확인만
        health = manager.check_system_health()
        manager.print_system_status(health)
        sys.exit(0 if health['overall'] else 1)
    
    if args.stop:
        # 시스템 종료
        manager.shutdown()
        sys.exit(0)
    
    # 시그널 핸들러 등록 (Ctrl+C 처리)
    def signal_handler(signum, frame):
        print("\n🛑 Received shutdown signal")
        manager.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 시스템 시작
        if manager.start_all():
            print("\n🎉 Enhanced RAG System is ready!")
            print("   Press Ctrl+C to stop the system")
            
            # 메인 루프 (종료 신호 대기)
            while True:
                time.sleep(10)
                # 주기적 상태 확인
                health = manager.check_system_health()
                if not health['overall']:
                    logger.warning("⚠️ System health degraded")
                    manager.print_system_status(health)
        else:
            logger.error("❌ Failed to start Enhanced RAG System")
            manager.shutdown()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        manager.shutdown()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        manager.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
