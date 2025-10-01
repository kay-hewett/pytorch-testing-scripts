#!/usr/bin/env python3
"""
GPU vs CPU Benchmark with CUDAGraphs and Host Synchronization
============================================================

This script benchmarks the performance difference between GPU and CPU operations,
including the impact of host synchronizations and CUDAGraph optimization.

Features:
- GPU vs CPU performance comparison
- Host synchronization overhead analysis
- CUDAGraph optimization benchmarking
- Various workload sizes and patterns
"""

import argparse
import time
import json
from typing import Tuple, Callable
from dataclasses import dataclass
import statistics

import torch
import torch.nn.functional as F


@dataclass
class BenchmarkResult:
    name: str
    device: str
    use_cudagraph: bool
    use_sync: bool
    avg_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    speedup: float = 1.0


class BenchmarkSuite:
    def __init__(self, warmup_iters: int = 10, benchmark_iters: int = 100):
        self.warmup_iters = warmup_iters
        self.benchmark_iters = benchmark_iters
        self.results: list[BenchmarkResult] = []
        
    def time_cpu(self, fn: Callable, args: Tuple, iters: int) -> list[float]:
        """Time CPU operations"""
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            fn(*args)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to milliseconds
        return times
    
    def time_gpu(self, fn: Callable, args: Tuple, iters: int, use_sync: bool = True, device_type: str = "cuda") -> list[float]:
        """Time GPU operations with optional synchronization"""
        times = []
        
        if use_sync:
            # Use GPU events for accurate timing (CUDA only) or time.perf_counter for MPS
            if device_type == "cuda":
                for _ in range(iters):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    start_event.record()
                    fn(*args)
                    end_event.record()
                    
                    torch.cuda.synchronize()
                    times.append(start_event.elapsed_time(end_event))
            else:  # MPS or other devices
                for _ in range(iters):
                    if device_type == "mps":
                        torch.mps.synchronize()
                    
                    start = time.perf_counter()
                    fn(*args)
                    
                    if device_type == "mps":
                        torch.mps.synchronize()
                    
                    end = time.perf_counter()
                    times.append((end - start) * 1000)
        else:
            # Measure without explicit synchronization (includes host overhead)
            for _ in range(iters):
                start = time.perf_counter()
                fn(*args)
                # No synchronization here - measures async dispatch time
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to milliseconds
                
        return times
    
    def benchmark_function(self, name: str, fn: Callable, args_cpu: Tuple, args_gpu: Tuple,
                           use_cudagraph: bool = False) -> list[BenchmarkResult]:
        """Benchmark a function on both CPU and GPU with various configurations"""
        results = []
        
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {name}")
        print(f"{'=' * 60}")
        
        # CPU Benchmark
        print("  Running CPU benchmark...")
        
        # Warmup
        for _ in range(self.warmup_iters):
            fn(*args_cpu)
        
        # Benchmark
        cpu_times = self.time_cpu(fn, args_cpu, self.benchmark_iters)
        cpu_result = BenchmarkResult(
            name=name,
            device="cpu", 
            use_cudagraph=False,
            use_sync=True,  # CPU doesn't need async handling
            avg_time_ms=statistics.mean(cpu_times),
            std_time_ms=statistics.stdev(cpu_times) if len(cpu_times) > 1 else 0,
            min_time_ms=min(cpu_times),
            max_time_ms=max(cpu_times)
        )
        results.append(cpu_result)
        
        # Determine GPU device type and availability
        gpu_available = torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
        device_name = "cuda" if torch.cuda.is_available() else "mps" if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else None
        
        if gpu_available and device_name:
            # GPU Benchmark - with synchronization
            print(f"  Running {device_name.upper()} benchmark (with sync)...")
            
            # Warmup
            for _ in range(self.warmup_iters):
                fn(*args_gpu)
            if device_name == "cuda":
                torch.cuda.synchronize()
            elif device_name == "mps":
                torch.mps.synchronize()
            
            # Benchmark
            gpu_sync_times = self.time_gpu(fn, args_gpu, self.benchmark_iters, use_sync=True, device_type=device_name)
            gpu_sync_result = BenchmarkResult(
                name=name,
                device=device_name,
                use_cudagraph=False,
                use_sync=True,
                avg_time_ms=statistics.mean(gpu_sync_times),
                std_time_ms=statistics.stdev(gpu_sync_times) if len(gpu_sync_times) > 1 else 0,
                min_time_ms=min(gpu_sync_times),
                max_time_ms=max(gpu_sync_times),
                speedup=cpu_result.avg_time_ms / statistics.mean(gpu_sync_times)
            )
            results.append(gpu_sync_result)
            
            # GPU Benchmark - without explicit synchronization
            print(f"  Running {device_name.upper()} benchmark (async dispatch)...")
            
            # Warmup
            for _ in range(self.warmup_iters):
                fn(*args_gpu)
            
            # Benchmark
            gpu_async_times = self.time_gpu(fn, args_gpu, self.benchmark_iters, use_sync=False, device_type=device_name)
            gpu_async_result = BenchmarkResult(
                name=name,
                device=device_name,
                use_cudagraph=False,
                use_sync=False,
                avg_time_ms=statistics.mean(gpu_async_times),
                std_time_ms=statistics.stdev(gpu_async_times) if len(gpu_async_times) > 1 else 0,
                min_time_ms=min(gpu_async_times),
                max_time_ms=max(gpu_async_times),
                speedup=cpu_result.avg_time_ms / statistics.mean(gpu_async_times)
            )
            results.append(gpu_async_result)
            
            # CUDAGraph Benchmark (only for CUDA devices)
            if use_cudagraph and device_name == "cuda" and hasattr(torch.cuda, 'CUDAGraph'):
                print("  Running GPU benchmark (with CUDAGraph)...")
                
                try:
                    # Create static inputs for graph capture
                    static_args = tuple(torch.zeros_like(arg) if isinstance(arg, torch.Tensor) else arg for arg in args_gpu)
                    
                    # Copy input data to static tensors
                    for static_arg, orig_arg in zip(static_args, args_gpu):
                        if isinstance(static_arg, torch.Tensor):
                            static_arg.copy_(orig_arg)
                    
                    # Warmup
                    torch.cuda.synchronize()
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        fn(*static_args)
                    stream.synchronize()
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    
                    # Capture graph
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        fn(*static_args)
                    
                    # Benchmark graph replay
                    torch.cuda.synchronize()
                    
                    cudagraph_times = []
                    for _ in range(self.benchmark_iters):
                        # Update inputs if needed
                        for static_arg, orig_arg in zip(static_args, args_gpu):
                            if isinstance(static_arg, torch.Tensor):
                                static_arg.copy_(orig_arg)
                        
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        
                        start_event.record()
                        graph.replay()
                        end_event.record()
                        
                        torch.cuda.synchronize()
                        cudagraph_times.append(start_event.elapsed_time(end_event))
                    
                    cudagraph_result = BenchmarkResult(
                        name=name,
                        device="cuda",
                        use_cudagraph=True,
                        use_sync=True,
                        avg_time_ms=statistics.mean(cudagraph_times),
                        std_time_ms=statistics.stdev(cudagraph_times) if len(cudagraph_times) > 1 else 0,
                        min_time_ms=min(cudagraph_times),
                        max_time_ms=max(cudagraph_times),
                        speedup=cpu_result.avg_time_ms / statistics.mean(cudagraph_times)
                    )
                    results.append(cudagraph_result)
                    
                except Exception as e:
                    print(f"    CUDAGraph benchmark failed: {e}")
            elif use_cudagraph and device_name == "mps":
                print("  CUDAGraph not supported on MPS - skipping...")
            
            # Check if this is a torch.compile function (has compile-related attributes)
            if hasattr(fn, '_torchdynamo_orig_callable') or 'Compiled' in name:
                print(f"  Note: {name} uses torch.compile(mode='reduce-overhead') which automatically")
                print("        employs CUDAGraph-like optimizations for reduced launch overhead")
        
        self.results.extend(results)
        return results


def create_workloads():
    """Create various computational workloads for benchmarking"""
    
    # Small matrix operations
    def small_matmul(a, b):
        return torch.matmul(a, b)
    
    # Medium sized operations
    def medium_operations(x):
        return torch.relu(torch.matmul(x, x.T) + 1.0)
    
    # Element-wise operations
    def elementwise_ops(x):
        return torch.sin(x) + torch.cos(x) * torch.exp(-x)
    
    # Convolution operation
    def conv_op(x, weight):
        return F.conv2d(x, weight, padding=1)
    
    # Reduction operations  
    def reduction_ops(x):
        return torch.sum(x, dim=-1) + torch.mean(x, dim=-2)
    
    # Compiled versions for torch.compile benchmarking
    compiled_small_matmul = torch.compile(small_matmul, mode="reduce-overhead")
    compiled_medium_operations = torch.compile(medium_operations, mode="reduce-overhead") 
    compiled_elementwise_ops = torch.compile(elementwise_ops, mode="reduce-overhead")
    compiled_conv_op = torch.compile(conv_op, mode="reduce-overhead")
    compiled_reduction_ops = torch.compile(reduction_ops, mode="reduce-overhead")
    
    workloads = []
    
    # Determine GPU device
    device = "cuda" if torch.cuda.is_available() else "mps" if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else "cpu"
    
    # Small workloads
    small_size = (128, 128)
    workloads.append({
        'name': 'Small Matrix Multiply',
        'fn': small_matmul,
        'args_cpu': (torch.randn(*small_size), torch.randn(*small_size)),
        'args_gpu': (torch.randn(*small_size, device=device), torch.randn(*small_size, device=device)),
        'use_cudagraph': True
    })
    
    # Add torch.compile version for small matrices
    workloads.append({
        'name': 'Small Matrix Multiply (Compiled)',
        'fn': compiled_small_matmul,
        'args_cpu': (torch.randn(*small_size), torch.randn(*small_size)),
        'args_gpu': (torch.randn(*small_size, device=device), torch.randn(*small_size, device=device)),
        'use_cudagraph': False  # torch.compile handles CUDAGraph automatically
    })
    
    # Medium workloads
    medium_size = (512, 512)
    workloads.append({
        'name': 'Medium Matrix Operations', 
        'fn': medium_operations,
        'args_cpu': (torch.randn(*medium_size),),
        'args_gpu': (torch.randn(*medium_size, device=device),),
        'use_cudagraph': True
    })
    
    # Add torch.compile version for medium matrices
    workloads.append({
        'name': 'Medium Matrix Operations (Compiled)',
        'fn': compiled_medium_operations,
        'args_cpu': (torch.randn(*medium_size),),
        'args_gpu': (torch.randn(*medium_size, device=device),),
        'use_cudagraph': False  # torch.compile handles CUDAGraph automatically
    })
    
    # Large workloads
    large_size = (2048, 2048)
    workloads.append({
        'name': 'Large Matrix Multiply',
        'fn': small_matmul,
        'args_cpu': (torch.randn(*large_size), torch.randn(*large_size)),
        'args_gpu': (torch.randn(*large_size, device=device), torch.randn(*large_size, device=device)),
        'use_cudagraph': True
    })
    
    # Element-wise operations
    ewise_size = (1024, 1024)
    workloads.append({
        'name': 'Element-wise Operations',
        'fn': elementwise_ops,
        'args_cpu': (torch.randn(*ewise_size),),
        'args_gpu': (torch.randn(*ewise_size, device=device),),
        'use_cudagraph': True
    })
    
    # Add torch.compile version for element-wise ops
    workloads.append({
        'name': 'Element-wise Operations (Compiled)',
        'fn': compiled_elementwise_ops,
        'args_cpu': (torch.randn(*ewise_size),),
        'args_gpu': (torch.randn(*ewise_size, device=device),),
        'use_cudagraph': False  # torch.compile handles CUDAGraph automatically
    })
    
    # Convolution
    conv_input = (32, 64, 128, 128)  # batch, channels, height, width
    conv_weight = (64, 64, 3, 3)     # out_channels, in_channels, kernel_h, kernel_w
    workloads.append({
        'name': 'Convolution 2D',
        'fn': conv_op,
        'args_cpu': (torch.randn(*conv_input), torch.randn(*conv_weight)),
        'args_gpu': (torch.randn(*conv_input, device=device), torch.randn(*conv_weight, device=device)),
        'use_cudagraph': True
    })
    
    # Add torch.compile version for convolution
    workloads.append({
        'name': 'Convolution 2D (Compiled)',
        'fn': compiled_conv_op,
        'args_cpu': (torch.randn(*conv_input), torch.randn(*conv_weight)),
        'args_gpu': (torch.randn(*conv_input, device=device), torch.randn(*conv_weight, device=device)),
        'use_cudagraph': False  # torch.compile handles CUDAGraph automatically
    })
    
    # Reduction operations
    reduce_size = (1024, 1024)
    workloads.append({
        'name': 'Reduction Operations',
        'fn': reduction_ops,
        'args_cpu': (torch.randn(*reduce_size),),
        'args_gpu': (torch.randn(*reduce_size, device=device),),
        'use_cudagraph': True
    })
    
    return workloads


def print_results(results: list[BenchmarkResult]):
    """Print benchmark results in a formatted table"""
    
    print(f"\n{'=' * 120}")
    print(f"{'BENCHMARK RESULTS':^120}")
    print(f"{'=' * 120}")
    
    # Group results by benchmark name
    grouped_results = {}
    for result in results:
        if result.name not in grouped_results:
            grouped_results[result.name] = []
        grouped_results[result.name].append(result)
    
    for name, group_results in grouped_results.items():
        print(f"\n{name}")
        print("-" * len(name))
        
        header = f"{'Device':<10} {'CUDAGraph':<10} {'Sync':<6} {'Avg (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Speedup':<10}"
        print(header)
        print("-" * len(header))
        
        for result in group_results:
            cudagraph_str = "Yes" if result.use_cudagraph else "No"
            sync_str = "Yes" if result.use_sync else "No"
            
            print(f"{result.device:<10} {cudagraph_str:<10} {sync_str:<6} "
                  f"{result.avg_time_ms:<12.3f} {result.std_time_ms:<12.3f} "
                  f"{result.min_time_ms:<12.3f} {result.max_time_ms:<12.3f} {result.speedup:<10.2f}x")


def save_results(results: list[BenchmarkResult], filename: str):
    """Save results to JSON file"""
    data = []
    for result in results:
        data.append({
            'name': result.name,
            'device': result.device,
            'use_cudagraph': result.use_cudagraph,
            'use_sync': result.use_sync,
            'avg_time_ms': result.avg_time_ms,
            'std_time_ms': result.std_time_ms,
            'min_time_ms': result.min_time_ms,
            'max_time_ms': result.max_time_ms,
            'speedup': result.speedup
        })
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='GPU vs CPU Benchmark with CUDAGraphs')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--iters', type=int, default=100, help='Number of benchmark iterations') 
    parser.add_argument('--output', type=str, default='benchmark_results.json', help='Output JSON file')
    parser.add_argument('--no-cudagraph', action='store_true', help='Skip CUDAGraph benchmarks')
    parser.add_argument('--workload', type=str, help='Run specific workload only')
    
    args = parser.parse_args()
    
    print("GPU vs CPU Benchmark with CUDAGraphs and Host Synchronization")
    print("=" * 60)
    
    # Check GPU availability
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    if cuda_available:
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"CUDAGraph Support: {hasattr(torch.cuda, 'CUDAGraph')}")
        print("torch.compile with reduce-overhead mode: Uses CUDAGraph automatically")
    elif mps_available:
        print("MPS (Metal Performance Shaders) available")
        print("CUDAGraph Support: No (MPS does not support CUDAGraph)")
        print("torch.compile with reduce-overhead mode: Uses MPS-optimized kernels")
    else:
        print("No GPU acceleration available - running CPU-only benchmarks")
        print("torch.compile with reduce-overhead mode: Uses CPU optimizations")
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Warmup Iterations: {args.warmup}")
    print(f"Benchmark Iterations: {args.iters}")
    
    # Create benchmark suite
    benchmark_suite = BenchmarkSuite(warmup_iters=args.warmup, benchmark_iters=args.iters)
    
    # Get workloads
    workloads = create_workloads()
    
    # Filter workloads if specified
    if args.workload:
        workloads = [w for w in workloads if args.workload.lower() in w['name'].lower()]
        if not workloads:
            print(f"No workloads found matching '{args.workload}'")
            return
    
    # Run benchmarks
    all_results = []
    for workload in workloads:
        if cuda_available or mps_available:
            # GPU benchmarking available
            try:
                results = benchmark_suite.benchmark_function(
                    workload['name'],
                    workload['fn'],
                    workload['args_cpu'],
                    workload['args_gpu'],
                    use_cudagraph=workload.get('use_cudagraph', False) and not args.no_cudagraph and cuda_available
                )
                all_results.extend(results)
            except Exception as e:
                print(f"Error benchmarking {workload['name']}: {e}")
        else:
            # CPU only
            results = benchmark_suite.benchmark_function(
                workload['name'],
                workload['fn'],
                workload['args_cpu'],
                workload['args_cpu'],  # Use CPU args for both
                use_cudagraph=False
            )
            all_results.extend(results)
    
    # Print and save results
    print_results(all_results)
    save_results(all_results, args.output)
    
    print(f"\nBenchmark completed! Total configurations tested: {len(all_results)}")


if __name__ == '__main__':
    main()
