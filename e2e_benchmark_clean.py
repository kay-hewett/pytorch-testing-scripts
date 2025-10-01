#!/usr/bin/env python3
"""
End-to-End Performance Analysis: Host Synchronization Impact

This benchmark demonstrates why isolated operator performance is misleading
and why host synchronizations dominate real-world performance characteristics.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
import statistics


class MLPModel(nn.Module):
    """Simple MLP for benchmarking"""
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


def benchmark_isolated_operations(device, batch_size=256, iters=50):
    """Benchmark isolated operations - the MISLEADING approach"""
    print(f"\n=== Isolated Operations Benchmark ({device}) ===")

    model = MLPModel().to(device)
    data = torch.randn(batch_size, 1024, device=device)
    target = torch.randint(0, 10, (batch_size,), device=device)

    # Warmup
    for _ in range(10):
        output = model(data)
        loss = F.cross_entropy(output, target)

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    # Benchmark
    times = []
    for _ in range(iters):
        start = time.perf_counter()

        output = model(data)
        loss = F.cross_entropy(output, target)

        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_time = statistics.mean(times)
    print(f"  Average time per forward pass: {avg_time:.3f}ms")
    return avg_time


def benchmark_with_frequent_syncs(device, batch_size=256, iters=50):
    """Benchmark with realistic host synchronizations - the REAL scenario"""
    print(f"\n=== Realistic Pipeline with Host Syncs ({device}) ===")

    model = MLPModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Simulate realistic training data
    data = torch.randn(batch_size, 1024, device=device)
    target = torch.randint(0, 10, (batch_size,), device=device)

    # Warmup
    for _ in range(10):
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    # Benchmark with realistic monitoring/logging
    times = []
    for i in range(iters):
        start = time.perf_counter()

        # Training step
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # Realistic monitoring (every 10 steps)
        if i % 10 == 0:
            # These operations FORCE host synchronization!
            loss_val = loss.item()  # HOST SYNC!
            accuracy = (output.argmax(dim=1) == target).float().mean().item()  # HOST SYNC!

            # Simulated logging/monitoring overhead
            _ = f"Step {i}, Loss: {loss_val:.4f}, Acc: {accuracy:.4f}"

        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_time = statistics.mean(times)
    print(f"  Average time per training step: {avg_time:.3f}ms")
    return avg_time


def benchmark_cudagraph_incompatible(device, batch_size=256, iters=50):
    """Benchmark operations that break CUDAGraph compatibility"""
    print(f"\n=== CUDAGraph-Incompatible Operations ({device}) ===")

    if device != "cuda":
        print("  Skipping - CUDAGraph only available on CUDA")
        return None

    model = MLPModel().to(device)

    times = []
    for i in range(iters):
        # Dynamic shapes - breaks CUDAGraph!
        dynamic_batch_size = batch_size + (i % 32)  # Varying batch sizes
        data = torch.randn(dynamic_batch_size, 1024, device=device)

        start = time.perf_counter()

        output = model(data)

        # Control flow - breaks CUDAGraph!
        if i % 20 == 0:
            output = output * 1.1  # Conditional operation

        # Memory allocation - breaks CUDAGraph!
        temp_tensor = torch.randn_like(output)
        result = output + temp_tensor

        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_time = statistics.mean(times)
    print(f"  Average time (CUDAGraph-incompatible): {avg_time:.3f}ms")
    return avg_time


def benchmark_cudagraph_compatible(device, batch_size=256, iters=50):
    """Benchmark with CUDAGraph optimization"""
    print(f"\n=== CUDAGraph-Compatible Benchmark ({device}) ===")

    if device != "cuda":
        print("  Skipping - CUDAGraph only available on CUDA")
        return None

    model = MLPModel().to(device)

    # Static tensors for CUDAGraph
    static_input = torch.randn(batch_size, 1024, device=device)
    static_target = torch.randint(0, 10, (batch_size,), device=device)

    # Warmup
    torch.cuda.synchronize()
    for _ in range(10):
        output = model(static_input)
        loss = F.cross_entropy(output, static_target)
    torch.cuda.synchronize()

    # Capture CUDAGraph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = model(static_input)
        static_loss = F.cross_entropy(static_output, static_target)

    # Benchmark graph replay
    times = []
    for _ in range(iters):
        start = time.perf_counter()

        # Update input data (allowed in CUDAGraph)
        static_input.copy_(torch.randn(batch_size, 1024, device=device))

        # Replay graph - VERY fast!
        graph.replay()

        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_time = statistics.mean(times)
    print(f"  Average time (CUDAGraph): {avg_time:.3f}ms")
    return avg_time


def analyze_hybrid_overhead():
    """Demonstrate performance regression with hybrid CPU/GPU approach"""
    print(f"\n{'='*80}")
    print("HYBRID APPROACH PERFORMANCE REGRESSION ANALYSIS")
    print(f"{'='*80}")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print("\nSimulating hybrid CPU/GPU pipeline:")
    print("- Small operations → CPU")
    print("- Large operations → GPU") 
    print("- Result: CONSTANT host synchronizations!")

    # Simulate a realistic mixed workload
    operations = [
        ("embedding_lookup", (32, 128)),      # Small - CPU
        ("attention_weights", (512, 512)),     # Large - GPU
        ("layer_norm", (512,)),               # Small - CPU
        ("ffn_forward", (1024, 1024)),        # Large - GPU
        ("softmax", (512,)),                  # Small - CPU
    ]

    total_sync_overhead = 0
    total_compute_time = 0

    for op_name, size in operations:
        # Decision based on size (the flawed approach)
        use_gpu = max(size) > 500

        print(f"\n{op_name} {size} → {'GPU' if use_gpu else 'CPU'}")

        # Simulate compute time
        compute_time = max(size) * 0.001  # Simplified model

        # Host sync overhead when switching devices
        sync_overhead = 0.1 if use_gpu else 0  # 0.1ms per GPU operation

        total_compute_time += compute_time
        total_sync_overhead += sync_overhead

        print(f"  Compute: {compute_time:.3f}ms, Sync overhead: {sync_overhead:.3f}ms")

    print(f"\nTotal compute time: {total_compute_time:.3f}ms")
    print(f"Total sync overhead: {total_sync_overhead:.3f}ms")
    print(f"Overhead percentage: {(total_sync_overhead/total_compute_time)*100:.1f}%")

    # Compare to GPU-only pipeline
    gpu_only_time = total_compute_time * 0.8  # Assume GPU is 25% faster
    print(f"\nGPU-only pipeline time: {gpu_only_time:.3f}ms (no sync overhead)")
    print(f"Hybrid vs GPU-only: {(total_compute_time + total_sync_overhead) / gpu_only_time:.2f}x slower")


def main():
    print("End-to-End Performance Analysis: Host Synchronization Impact")
    print("=" * 60)

    # Check available devices
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    if cuda_available:
        device = "cuda"
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
    elif mps_available:
        device = "mps"
        print("MPS Device: Apple Silicon GPU")
    else:
        device = "cpu"
        print("CPU Device: No GPU acceleration")

    print(f"PyTorch Version: {torch.__version__}")

    # Run comprehensive benchmarks
    results = {}

    # 1. Misleading isolated benchmark
    results['isolated'] = benchmark_isolated_operations(device)

    # 2. Realistic pipeline with host syncs
    results['realistic'] = benchmark_with_frequent_syncs(device)

    # 3. CUDAGraph incompatible operations
    results['incompatible'] = benchmark_cudagraph_incompatible(device)

    # 4. CUDAGraph compatible operations
    results['cudagraph'] = benchmark_cudagraph_compatible(device)

    # 5. Hybrid approach regression analysis
    analyze_hybrid_overhead()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: Why Isolated Benchmarks Are Misleading")
    print(f"{'='*80}")

    if device == "cuda" and results['cudagraph']:
        print(f"Isolated operations:     {results['isolated']:.3f}ms")
        print(f"Realistic pipeline:      {results['realistic']:.3f}ms ({results['realistic']/results['isolated']:.1f}x slower)")
        print(f"CUDAGraph incompatible:  {results['incompatible']:.3f}ms ({results['incompatible']/results['isolated']:.1f}x slower)")
        print(f"CUDAGraph optimized:     {results['cudagraph']:.3f}ms ({results['cudagraph']/results['isolated']:.1f}x faster)")

        print(f"\nCUDAGraph vs Realistic Pipeline: {results['realistic']/results['cudagraph']:.1f}x speedup!")
    else:
        print(f"Isolated operations:     {results['isolated']:.3f}ms")
        print(f"Realistic pipeline:      {results['realistic']:.3f}ms ({results['realistic']/results['isolated']:.1f}x slower)")

    print("\nKey Insights:")
    print("1. Host synchronizations dominate performance in realistic scenarios")
    print("2. Hybrid CPU/GPU approaches introduce constant sync overhead")
    print("3. CUDAGraphs require consistent device placement and static patterns")
    print("4. End-to-end pipeline design is more important than individual operator speed")


if __name__ == '__main__':
    main()
