"""
Performance testing suite for Pharmaceutical Research Assistant.
Tests API response times, throughput, and system limits for demo purposes.
"""
import asyncio
import time
import statistics
import psutil
import requests
import json
import concurrent.futures
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    operation_name: str
    response_times: List[float] = field(default_factory=list)
    success_count: int = 0
    error_count: int = 0
    total_requests: int = 0
    start_time: float = 0
    end_time: float = 0
    
    @property
    def duration(self) -> float:
        """Total test duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.success_count / self.total_requests) * 100
    
    @property
    def avg_response_time(self) -> float:
        """Average response time in milliseconds."""
        return statistics.mean(self.response_times) if self.response_times else 0
    
    @property
    def p95_response_time(self) -> float:
        """95th percentile response time in milliseconds."""
        if not self.response_times:
            return 0
        sorted_times = sorted(self.response_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[index] if index < len(sorted_times) else sorted_times[-1]
    
    @property
    def throughput_rps(self) -> float:
        """Requests per second."""
        if self.duration == 0:
            return 0
        return self.success_count / self.duration


class PharmaceuticalPerformanceTester:
    """Performance testing suite for pharmaceutical research system."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1"
        self.session = requests.Session()
        
        # Sample data for testing
        self.sample_compounds = [
            {
                "name": "Aspirin",
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"
            },
            {
                "name": "Ibuprofen", 
                "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
            },
            {
                "name": "Caffeine",
                "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
            },
            {
                "name": "Acetaminophen",
                "smiles": "CC(=O)NC1=CC=C(C=C1)O"
            },
            {
                "name": "Morphine",
                "smiles": "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"
            }
        ]
        
        self.literature_queries = [
            "anti-inflammatory drugs",
            "drug discovery",
            "pharmacokinetics",
            "clinical trials",
            "biomarkers"
        ]
    
    def check_system_health(self) -> bool:
        """Check if the system is running and healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                return health_data.get('status') == 'healthy'
            return False
        except Exception:
            return False
    
    def measure_response_time(self, func, *args, **kwargs) -> Tuple[float, Any, bool]:
        """Measure response time of a function call."""
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            return response_time, result, True
        except Exception as e:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            return response_time, str(e), False
    
    def test_chemical_analysis_performance(self, num_requests: int = 50) -> PerformanceMetrics:
        """Test chemical analysis endpoint performance."""
        print(f"\n🧪 Testing Chemical Analysis Performance ({num_requests} requests)")
        
        metrics = PerformanceMetrics("Chemical Analysis")
        metrics.start_time = time.time()
        metrics.total_requests = num_requests
        
        for i in range(num_requests):
            compound = self.sample_compounds[i % len(self.sample_compounds)]
            
            def make_request():
                return self.session.post(
                    f"{self.api_base}/chemical/analyze",
                    json=compound,
                    timeout=30
                )
            
            response_time, result, success = self.measure_response_time(make_request)
            metrics.response_times.append(response_time)
            
            if success and hasattr(result, 'status_code') and result.status_code == 200:
                metrics.success_count += 1
            else:
                metrics.error_count += 1
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_requests} requests")
        
        metrics.end_time = time.time()
        return metrics
    
    def test_literature_search_performance(self, num_requests: int = 20) -> PerformanceMetrics:
        """Test literature search endpoint performance."""
        print(f"\n📚 Testing Literature Search Performance ({num_requests} requests)")
        
        metrics = PerformanceMetrics("Literature Search")
        metrics.start_time = time.time()
        metrics.total_requests = num_requests
        
        for i in range(num_requests):
            query = self.literature_queries[i % len(self.literature_queries)]
            
            def make_request():
                return self.session.get(
                    f"{self.api_base}/literature/search",
                    params={"query": query, "max_results": 5},
                    timeout=60
                )
            
            response_time, result, success = self.measure_response_time(make_request)
            metrics.response_times.append(response_time)
            
            if success and hasattr(result, 'status_code') and result.status_code == 200:
                metrics.success_count += 1
            else:
                metrics.error_count += 1
            
            # Progress indicator
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{num_requests} requests")
        
        metrics.end_time = time.time()
        return metrics
    
    def test_concurrent_requests(self, num_concurrent: int = 10, requests_per_thread: int = 5) -> PerformanceMetrics:
        """Test concurrent request handling."""
        print(f"\n⚡ Testing Concurrent Performance ({num_concurrent} threads, {requests_per_thread} requests each)")
        
        metrics = PerformanceMetrics("Concurrent Requests")
        metrics.start_time = time.time()
        metrics.total_requests = num_concurrent * requests_per_thread
        
        def worker_thread(thread_id: int) -> List[Tuple[float, bool]]:
            """Worker thread function."""
            results = []
            for i in range(requests_per_thread):
                compound = self.sample_compounds[(thread_id + i) % len(self.sample_compounds)]
                
                def make_request():
                    return self.session.post(
                        f"{self.api_base}/chemical/analyze",
                        json=compound,
                        timeout=30
                    )
                
                response_time, result, success = self.measure_response_time(make_request)
                results.append((response_time, success))
            
            return results
        
        # Execute concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            future_to_thread = {
                executor.submit(worker_thread, i): i 
                for i in range(num_concurrent)
            }
            
            for future in concurrent.futures.as_completed(future_to_thread):
                thread_results = future.result()
                for response_time, success in thread_results:
                    metrics.response_times.append(response_time)
                    if success:
                        metrics.success_count += 1
                    else:
                        metrics.error_count += 1
        
        metrics.end_time = time.time()
        return metrics
    
    def test_similarity_search_performance(self, num_requests: int = 30) -> PerformanceMetrics:
        """Test chemical similarity search performance."""
        print(f"\n🔍 Testing Similarity Search Performance ({num_requests} requests)")
        
        metrics = PerformanceMetrics("Similarity Search")
        metrics.start_time = time.time()
        metrics.total_requests = num_requests
        
        query_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        candidate_smiles = [compound["smiles"] for compound in self.sample_compounds]
        
        for i in range(num_requests):
            def make_request():
                return self.session.post(
                    f"{self.api_base}/chemical/similarity",
                    params={
                        "query_smiles": query_smiles,
                        "candidate_smiles": candidate_smiles,
                        "threshold": 0.3,
                        "max_results": 10
                    },
                    timeout=30
                )
            
            response_time, result, success = self.measure_response_time(make_request)
            metrics.response_times.append(response_time)
            
            if success and hasattr(result, 'status_code') and result.status_code == 200:
                metrics.success_count += 1
            else:
                metrics.error_count += 1
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_requests} requests")
        
        metrics.end_time = time.time()
        return metrics
    
    def monitor_system_resources(self, duration: int = 60) -> Dict[str, Any]:
        """Monitor system resource usage during testing."""
        print(f"\n📊 Monitoring System Resources for {duration} seconds")
        
        cpu_usage = []
        memory_usage = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            cpu_usage.append(psutil.cpu_percent(interval=1))
            memory_info = psutil.virtual_memory()
            memory_usage.append(memory_info.percent)
        
        return {
            "avg_cpu_usage": statistics.mean(cpu_usage),
            "max_cpu_usage": max(cpu_usage),
            "avg_memory_usage": statistics.mean(memory_usage),
            "max_memory_usage": max(memory_usage),
            "cpu_samples": cpu_usage,
            "memory_samples": memory_usage
        }
    
    def generate_performance_report(self, all_metrics: List[PerformanceMetrics], 
                                   resource_data: Dict[str, Any]) -> None:
        """Generate comprehensive performance report."""
        print("\n" + "="*80)
        print("📊 PHARMACEUTICAL RESEARCH ASSISTANT - PERFORMANCE REPORT")
        print("="*80)
        
        # Summary table
        print("\n📈 PERFORMANCE SUMMARY")
        print("-" * 80)
        print(f"{'Operation':<20} {'Requests':<10} {'Success%':<10} {'Avg (ms)':<12} {'P95 (ms)':<12} {'RPS':<8}")
        print("-" * 80)
        
        for metrics in all_metrics:
            print(f"{metrics.operation_name:<20} "
                  f"{metrics.total_requests:<10} "
                  f"{metrics.success_rate:<10.1f} "
                  f"{metrics.avg_response_time:<12.1f} "
                  f"{metrics.p95_response_time:<12.1f} "
                  f"{metrics.throughput_rps:<8.2f}")
        
        # System resources
        print(f"\n🖥️  SYSTEM RESOURCES")
        print("-" * 40)
        print(f"Average CPU Usage: {resource_data['avg_cpu_usage']:.1f}%")
        print(f"Maximum CPU Usage: {resource_data['max_cpu_usage']:.1f}%")
        print(f"Average Memory Usage: {resource_data['avg_memory_usage']:.1f}%")
        print(f"Maximum Memory Usage: {resource_data['max_memory_usage']:.1f}%")
        
        # Performance insights
        print(f"\n💡 PERFORMANCE INSIGHTS")
        print("-" * 40)
        
        # Find fastest and slowest operations
        fastest = min(all_metrics, key=lambda m: m.avg_response_time)
        slowest = max(all_metrics, key=lambda m: m.avg_response_time)
        
        print(f"✅ Fastest Operation: {fastest.operation_name} ({fastest.avg_response_time:.1f}ms avg)")
        print(f"⏳ Slowest Operation: {slowest.operation_name} ({slowest.avg_response_time:.1f}ms avg)")
        
        # Overall success rate
        total_requests = sum(m.total_requests for m in all_metrics)
        total_successes = sum(m.success_count for m in all_metrics)
        overall_success_rate = (total_successes / total_requests) * 100 if total_requests > 0 else 0
        
        print(f"🎯 Overall Success Rate: {overall_success_rate:.1f}%")
        print(f"📊 Total Requests Processed: {total_requests}")
        
        # Performance recommendations
        print(f"\n🎯 RECOMMENDATIONS")
        print("-" * 40)
        
        if slowest.avg_response_time > 1000:
            print(f"⚠️  {slowest.operation_name} is slow (>{slowest.avg_response_time:.0f}ms) - consider caching")
        
        if resource_data['max_cpu_usage'] > 80:
            print("⚠️  High CPU usage detected - consider scaling")
        
        if resource_data['max_memory_usage'] > 80:
            print("⚠️  High memory usage detected - check for memory leaks")
        
        if overall_success_rate < 95:
            print("⚠️  Success rate below 95% - investigate error handling")
        
        if all(m.success_rate > 95 for m in all_metrics):
            print("✅ Excellent reliability - all operations >95% success rate")
        
        if fastest.avg_response_time < 200:
            print("✅ Great responsiveness - fastest operation <200ms")
    
    def create_performance_visualizations(self, all_metrics: List[PerformanceMetrics], 
                                        resource_data: Dict[str, Any]) -> None:
        """Create performance visualization charts."""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Pharmaceutical Research Assistant - Performance Dashboard', fontsize=16)
        
        # Response time comparison
        operations = [m.operation_name for m in all_metrics]
        avg_times = [m.avg_response_time for m in all_metrics]
        p95_times = [m.p95_response_time for m in all_metrics]
        
        x = range(len(operations))
        width = 0.35
        
        axes[0, 0].bar([i - width/2 for i in x], avg_times, width, label='Average', alpha=0.8)
        axes[0, 0].bar([i + width/2 for i in x], p95_times, width, label='95th Percentile', alpha=0.8)
        axes[0, 0].set_xlabel('Operations')
        axes[0, 0].set_ylabel('Response Time (ms)')
        axes[0, 0].set_title('Response Time Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(operations, rotation=45, ha='right')
        axes[0, 0].legend()
        
        # Throughput comparison
        throughputs = [m.throughput_rps for m in all_metrics]
        axes[0, 1].bar(operations, throughputs, alpha=0.8, color='green')
        axes[0, 1].set_xlabel('Operations')
        axes[0, 1].set_ylabel('Requests per Second')
        axes[0, 1].set_title('Throughput Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Success rate comparison
        success_rates = [m.success_rate for m in all_metrics]
        colors = ['green' if rate >= 95 else 'orange' if rate >= 90 else 'red' for rate in success_rates]
        axes[1, 0].bar(operations, success_rates, alpha=0.8, color=colors)
        axes[1, 0].set_xlabel('Operations')
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].set_title('Success Rate by Operation')
        axes[1, 0].set_ylim(80, 102)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # System resource usage over time
        time_points = list(range(len(resource_data['cpu_samples'])))
        axes[1, 1].plot(time_points, resource_data['cpu_samples'], label='CPU Usage', alpha=0.8)
        axes[1, 1].plot(time_points, resource_data['memory_samples'], label='Memory Usage', alpha=0.8)
        axes[1, 1].set_xlabel('Time (seconds)')
        axes[1, 1].set_ylabel('Usage (%)')
        axes[1, 1].set_title('System Resource Usage')
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig('performance_report.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Performance charts saved as 'performance_report.png'")
    
    def run_full_performance_suite(self) -> None:
        """Run the complete performance testing suite."""
        print("🧬 PHARMACEUTICAL RESEARCH ASSISTANT - PERFORMANCE TESTING SUITE")
        print("="*80)
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check system health
        if not self.check_system_health():
            print("❌ System health check failed. Please ensure the API is running.")
            print("   Start with: uvicorn src.main:app --reload")
            return
        
        print("✅ System health check passed")
        
        # Start resource monitoring in background
        import threading
        resource_data = {}
        
        def monitor_resources():
            resource_data.update(self.monitor_system_resources(duration=120))
        
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()
        
        # Run performance tests
        all_metrics = []
        
        try:
            # Test individual endpoints
            all_metrics.append(self.test_chemical_analysis_performance(50))
            all_metrics.append(self.test_similarity_search_performance(30))
            all_metrics.append(self.test_literature_search_performance(10))  # Fewer due to external API
            all_metrics.append(self.test_concurrent_requests(10, 5))
            
            # Wait for resource monitoring to complete
            monitor_thread.join()
            
            # Generate comprehensive report
            self.generate_performance_report(all_metrics, resource_data)
            
            # Create visualizations
            try:
                self.create_performance_visualizations(all_metrics, resource_data)
            except Exception as e:
                print(f"⚠️  Could not generate visualizations: {e}")
                print("   Install matplotlib and seaborn for charts")
            
        except KeyboardInterrupt:
            print("\n⏹️  Performance testing interrupted by user")
        except Exception as e:
            print(f"\n❌ Performance testing failed: {e}")
        
        print(f"\n🏁 Performance testing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main function to run performance tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pharmaceutical Research Assistant Performance Testing")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--quick", action="store_true", help="Run quick test suite")
    parser.add_argument("--endpoint", choices=["chemical", "literature", "similarity", "concurrent"], 
                       help="Test specific endpoint only")
    
    args = parser.parse_args()
    
    tester = PharmaceuticalPerformanceTester(args.url)
    
    if args.endpoint:
        # Test specific endpoint
        if args.endpoint == "chemical":
            metrics = tester.test_chemical_analysis_performance(20 if args.quick else 50)
        elif args.endpoint == "literature":
            metrics = tester.test_literature_search_performance(5 if args.quick else 10)
        elif args.endpoint == "similarity":
            metrics = tester.test_similarity_search_performance(10 if args.quick else 30)
        elif args.endpoint == "concurrent":
            metrics = tester.test_concurrent_requests(5 if args.quick else 10, 3 if args.quick else 5)
        
        print(f"\n📊 {metrics.operation_name} Results:")
        print(f"   Average Response Time: {metrics.avg_response_time:.1f}ms")
        print(f"   95th Percentile: {metrics.p95_response_time:.1f}ms")
        print(f"   Success Rate: {metrics.success_rate:.1f}%")
        print(f"   Throughput: {metrics.throughput_rps:.2f} RPS")
    else:
        # Run full suite
        tester.run_full_performance_suite()


if __name__ == "__main__":
    main()