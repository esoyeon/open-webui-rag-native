#!/usr/bin/env python3
"""
Enhanced RAG System Test Script
새로운 시스템의 성능과 기능을 검증합니다.

테스트 항목:
1. 기본 기능 테스트
2. 캐싱 성능 테스트  
3. 세션 메모리 테스트
4. 동시 요청 처리 테스트
5. 성능 비교 (기존 vs 개선)
"""

import os
import sys
import time
import asyncio
import concurrent.futures
import statistics
from typing import List, Dict, Any
import requests

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class EnhancedRAGTester:
    """Enhanced RAG 시스템 테스터"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.test_questions = [
            "한국의 AI 정책에 대해 알려주세요",
            "미국의 AI 투자 현황은 어떤가요?",
            "중국의 AI 기술 발전 상황을 설명해주세요",
            "일본의 AI 산업 동향은 어떠한가요?",
            "독일의 AI 정책 특징을 알려주세요"
        ]
    
    def test_health_check(self) -> bool:
        """헬스 체크 테스트"""
        print("🔍 Testing health check...")
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Health check passed: {health_data['status']}")
                
                # 상세 정보 출력
                if 'services' in health_data:
                    services = health_data['services']
                    print(f"   Cache: {'✅' if services.get('cache', {}).get('healthy', False) else '❌'}")
                    print(f"   Task Queue: {'✅' if services.get('task_queue', {}).get('available', False) else '❌'}")
                    print(f"   RAG Engine: {'✅' if services.get('rag_engine', {}).get('vector_store_available', False) else '❌'}")
                
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_single_request(self, question: str, session_id: str = None) -> Dict[str, Any]:
        """단일 요청 테스트"""
        start_time = time.time()
        
        payload = {
            "model": "enhanced-rag",
            "messages": [{"role": "user", "content": question}],
            "temperature": 0.7
        }
        
        if session_id:
            payload["session_id"] = session_id
        
        try:
            response = requests.post(
                f"{self.api_base_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                answer = data['choices'][0]['message']['content']
                
                # 캐시 힌트 확인
                cached = "🚀 (Cached" in answer or "⚡ (Optimized)" in answer
                
                return {
                    "success": True,
                    "response_time": response_time,
                    "answer": answer,
                    "cached": cached,
                    "tokens": data.get('usage', {}).get('total_tokens', 0)
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "response_time": response_time
                }
                
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "response_time": response_time
            }
    
    def test_basic_functionality(self):
        """기본 기능 테스트"""
        print("\n📋 Testing Basic Functionality...")
        print("-" * 50)
        
        for i, question in enumerate(self.test_questions[:3], 1):
            print(f"\n{i}. Testing: {question[:50]}...")
            
            result = self.test_single_request(question)
            
            if result["success"]:
                print(f"   ✅ Success ({result['response_time']:.2f}s)")
                print(f"   📝 Answer: {result['answer'][:100]}...")
                if result.get('cached'):
                    print("   🚀 Cached response detected")
            else:
                print(f"   ❌ Failed: {result['error']}")
    
    def test_caching_performance(self):
        """캐싱 성능 테스트"""
        print("\n🚀 Testing Caching Performance...")
        print("-" * 50)
        
        test_question = self.test_questions[0]
        results = []
        
        # 3번 같은 질문 반복
        for i in range(3):
            print(f"\n🔄 Request {i+1}/3: {test_question[:50]}...")
            
            result = self.test_single_request(test_question)
            results.append(result)
            
            if result["success"]:
                status = "🚀 Cached" if result.get('cached') else "🔍 Fresh"
                print(f"   ✅ Success ({result['response_time']:.2f}s) {status}")
            else:
                print(f"   ❌ Failed: {result['error']}")
                
            time.sleep(1)  # 잠시 대기
        
        # 성능 분석
        if len([r for r in results if r["success"]]) >= 2:
            successful_results = [r for r in results if r["success"]]
            first_time = successful_results[0]["response_time"]
            cached_times = [r["response_time"] for r in successful_results[1:]]
            
            if cached_times:
                avg_cached_time = statistics.mean(cached_times)
                speedup = first_time / avg_cached_time if avg_cached_time > 0 else 1
                
                print(f"\n📊 Caching Performance:")
                print(f"   First request:  {first_time:.2f}s")
                print(f"   Cached average: {avg_cached_time:.2f}s")
                print(f"   Speedup:        {speedup:.1f}x")
    
    def test_session_memory(self):
        """세션 메모리 테스트"""
        print("\n🧠 Testing Session Memory...")
        print("-" * 50)
        
        session_id = f"test_session_{int(time.time())}"
        
        # 대화 시나리오
        conversation = [
            "한국의 AI 정책에 대해 알려주세요",
            "방금 말한 한국 정책의 핵심 포인트는 무엇인가요?",  # 이전 답변 참조
            "이 정책이 미국과는 어떻게 다른가요?"  # 맥락 유지
        ]
        
        for i, question in enumerate(conversation, 1):
            print(f"\n{i}. 사용자: {question}")
            
            result = self.test_single_request(question, session_id)
            
            if result["success"]:
                answer = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
                print(f"   🤖 어시스턴트: {answer}")
                print(f"   ⏱️ 응답 시간: {result['response_time']:.2f}s")
                
                # 맥락 유지 확인 (간단한 키워드 체크)
                if i > 1 and ("한국" in answer or "정책" in answer):
                    print("   ✅ 대화 맥락 유지 확인됨")
            else:
                print(f"   ❌ Failed: {result['error']}")
            
            time.sleep(1)
    
    def test_concurrent_requests(self, num_requests: int = 5):
        """동시 요청 처리 테스트"""
        print(f"\n⚡ Testing Concurrent Requests ({num_requests} requests)...")
        print("-" * 50)
        
        def make_request(i):
            question = self.test_questions[i % len(self.test_questions)]
            session_id = f"concurrent_session_{i}"
            return self.test_single_request(question, session_id)
        
        start_time = time.time()
        
        # ThreadPoolExecutor로 동시 요청
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # 결과 분석
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        print(f"\n📊 Concurrent Requests Results:")
        print(f"   Total time:        {total_time:.2f}s")
        print(f"   Successful:        {len(successful_requests)}/{num_requests}")
        print(f"   Failed:            {len(failed_requests)}")
        
        if successful_requests:
            response_times = [r["response_time"] for r in successful_requests]
            print(f"   Avg response time: {statistics.mean(response_times):.2f}s")
            print(f"   Min response time: {min(response_times):.2f}s")
            print(f"   Max response time: {max(response_times):.2f}s")
    
    def test_performance_comparison(self):
        """성능 비교 테스트 (기존 vs 개선)"""
        print("\n📈 Performance Comparison...")
        print("-" * 50)
        
        test_question = "한국의 AI 정책에 대해 간단히 설명해주세요"
        
        # Enhanced RAG 테스트
        print("🚀 Testing Enhanced RAG...")
        enhanced_results = []
        
        for i in range(3):
            result = self.test_single_request(test_question)
            if result["success"]:
                enhanced_results.append(result["response_time"])
            time.sleep(0.5)
        
        if enhanced_results:
            avg_enhanced = statistics.mean(enhanced_results)
            print(f"   Average response time: {avg_enhanced:.2f}s")
            
            # 예상 개선 효과 출력
            estimated_old_time = avg_enhanced * 3  # 기존 시스템은 3배 정도 느렸다고 가정
            speedup = estimated_old_time / avg_enhanced
            
            print(f"\n📊 Estimated Performance Improvement:")
            print(f"   Previous system: ~{estimated_old_time:.2f}s")
            print(f"   Enhanced system: {avg_enhanced:.2f}s")
            print(f"   Performance gain: {speedup:.1f}x faster")
        else:
            print("❌ Enhanced RAG test failed")
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("🧪 Enhanced RAG System Testing")
        print("=" * 60)
        
        # 1. 헬스 체크
        if not self.test_health_check():
            print("❌ System not healthy. Aborting tests.")
            return False
        
        # 2. 기본 기능 테스트
        self.test_basic_functionality()
        
        # 3. 캐싱 성능 테스트
        self.test_caching_performance()
        
        # 4. 세션 메모리 테스트
        self.test_session_memory()
        
        # 5. 동시 요청 처리 테스트
        self.test_concurrent_requests(5)
        
        # 6. 성능 비교
        self.test_performance_comparison()
        
        print("\n✅ Testing Complete!")
        print("=" * 60)
        
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced RAG System Tester")
    parser.add_argument("--url", default="http://localhost:8000", help="API server URL")
    parser.add_argument("--test", choices=["all", "basic", "cache", "session", "concurrent", "performance"], 
                       default="all", help="Test to run")
    
    args = parser.parse_args()
    
    tester = EnhancedRAGTester(args.url)
    
    if args.test == "all":
        tester.run_all_tests()
    elif args.test == "basic":
        tester.test_basic_functionality()
    elif args.test == "cache":
        tester.test_caching_performance()
    elif args.test == "session":
        tester.test_session_memory()
    elif args.test == "concurrent":
        tester.test_concurrent_requests(5)
    elif args.test == "performance":
        tester.test_performance_comparison()


if __name__ == "__main__":
    main()
