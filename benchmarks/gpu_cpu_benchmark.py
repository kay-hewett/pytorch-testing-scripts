#!/usr/bin/env python3
"""
GPU vs CPU Performance Benchmarking Utilities for PyTorch

This module provides comprehensive benchmarking tools for comparing GPU and CPU performance
across different PyTorch operations. Originally developed as part of PyTorch PR #162107
(commit 6eb65749b7fc286ad3434a0faaf964f02f245f8e).

Features:
- Cross-platform device detection (CUDA, MPS, CPU)
- Matrix multiplication benchmarking with proper GPU synchronization
- Neural network training performance comparison
- Structured performance reporting with speedup calculations
- Professional error handling and edge case management

Example Usage:
    python benchmarks/gpu_cpu_benchmark.py
    
    # Or programmatically:
    from benchmarks.gpu_cpu_benchmark import run_matrix_benchmark
    results = run_matrix_benchmark([1000, 2000, 4000])
"""

import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    import torch
    import torch.nn as nn
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    warnings.warn("PyTorch not available. Benchmarking functionality will be disabled.")


class DeviceDetector:
    """
    Detects and manages the best available device for PyTorch operations.
    
    This class provides utilities for automatic device detection with fallback
    support, following the priority: CUDA > MPS > CPU.
    """
    
    @staticmethod
    def get_best_device() -> torch.device:
        """
        Get the best available device based on hardware capabilities.
        
        Priority order: CUDA > MPS (Apple Metal) > CPU
        
        Returns:
            torch.device: The best available device
            
        Raises:
            ImportError: If PyTorch is not available
        """
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available")
            
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """
        Get comprehensive information about available devices.
        
        Returns:
            Dict[str, Any]: Dictionary containing device information including:
                - pytorch_version: PyTorch version string
                - cuda_available: Whether CUDA is available
                - mps_available: Whether MPS (Apple Metal) is available
                - cpu_count: Number of CPU threads
                - cuda_version: CUDA version (if available)
                - gpu_count: Number of available GPUs (if CUDA available)
                - gpu_name: Name of primary GPU (if available)
        """
        if not HAS_PYTORCH:
            return {"error": "PyTorch not available"}
            
        info = {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "cpu_count": torch.get_num_threads(),
        }
        
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            if torch.cuda.device_count() > 0:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                
        return info
    
    @staticmethod
    def synchronize_device(device: torch.device) -> None:
        """
        Synchronize operations on the specified device.
        
        Args:
            device: The device to synchronize
        """
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        # CPU operations are synchronous by default


class MatrixBenchmark:
    """
    Matrix operation benchmarking utilities.
    
    This class provides methods for benchmarking matrix operations across
    different devices with proper timing and synchronization.
    """
    
    @staticmethod
    def benchmark_matrix_multiplication(
        size: int, 
        device: torch.device, 
        warmup: bool = True,
        iterations: int = 1,
        dtype: torch.dtype = torch.float32
    ) -> float:
        """
        Benchmark matrix multiplication on the specified device.
        
        Args:
            size: Size of square matrices (size x size)
            device: Device to run the benchmark on
            warmup: Whether to perform warmup iterations
            iterations: Number of benchmark iterations
            dtype: Data type for matrices
            
        Returns:
            float: Average time per iteration in seconds
            
        Raises:
            ImportError: If PyTorch is not available
            RuntimeError: If device is not available
        """
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available")
        
        try:
            # Create test matrices
            x = torch.randn(size, size, device=device, dtype=dtype)
            y = torch.randn(size, size, device=device, dtype=dtype)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to create tensors on {device}: {e}") from e
        
        # Warmup runs
        if warmup:
            for _ in range(3):
                _ = torch.mm(x, y)
                DeviceDetector.synchronize_device(device)
        
        # Benchmark runs
        DeviceDetector.synchronize_device(device)
        start_time = time.time()
        
        for _ in range(iterations):
            result = torch.mm(x, y)
            
        DeviceDetector.synchronize_device(device)
        end_time = time.time()
        
        return (end_time - start_time) / iterations
    
    @staticmethod
    def benchmark_elementwise_operations(
        size: int,
        device: torch.device,
        warmup: bool = True
    ) -> float:
        """
        Benchmark elementwise operations on the specified device.
        
        Args:
            size: Size of square matrices
            device: Device to run the benchmark on
            warmup: Whether to perform warmup iterations
            
        Returns:
            float: Time taken in seconds
        """
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available")
            
        x = torch.randn(size, size, device=device)
        y = torch.randn(size, size, device=device)
        
        if warmup:
            for _ in range(3):
                _ = x * y + torch.sin(x)
                DeviceDetector.synchronize_device(device)
        
        DeviceDetector.synchronize_device(device)
        start_time = time.time()
        
        result = x * y + torch.sin(x) + torch.cos(y)
        
        DeviceDetector.synchronize_device(device)
        end_time = time.time()
        
        return end_time - start_time


class NeuralNetworkBenchmark:
    """
    Neural network benchmarking utilities.
    
    This class provides methods for benchmarking neural network operations
    including training and inference across different devices.
    """
    
    @staticmethod
    def create_test_network(
        input_size: int = 1000, 
        hidden_sizes: List[int] = [2000, 2000, 2000, 1000],
        output_size: int = 100,
        dropout_rate: float = 0.1
    ) -> nn.Module:
        """
        Create a configurable test neural network for benchmarking.
        
        Args:
            input_size: Size of input layer
            hidden_sizes: List of hidden layer sizes
            output_size: Size of output layer
            dropout_rate: Dropout rate for regularization
            
        Returns:
            nn.Module: Configured neural network
        """
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Remove dropout from final layer
        layers = layers[:-1]
        layers.append(nn.Linear(prev_size, output_size))
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def benchmark_training(
        device: torch.device,
        epochs: int = 10,
        batch_size: int = 256,
        input_size: int = 1000,
        output_size: int = 100,
        learning_rate: float = 0.001
    ) -> Dict[str, float]:
        """
        Benchmark neural network training on the specified device.
        
        Args:
            device: Device to run training on
            epochs: Number of training epochs
            batch_size: Batch size for training
            input_size: Input layer size
            output_size: Output layer size (number of classes)
            learning_rate: Learning rate for optimizer
            
        Returns:
            Dict[str, float]: Dictionary containing timing results
                - total_time: Total training time
                - avg_epoch_time: Average time per epoch
                - final_loss: Final loss value
        """
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available")
            
        # Create model and move to device
        model = NeuralNetworkBenchmark.create_test_network(
            input_size=input_size, 
            output_size=output_size
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Create dummy data
        input_data = torch.randn(batch_size, input_size, device=device)
        target = torch.randint(0, output_size, (batch_size,), device=device)
        
        # Warmup
        model.train()
        for _ in range(2):
            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        DeviceDetector.synchronize_device(device)
        
        # Benchmark training
        start_time = time.time()
        final_loss = 0.0
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if epoch == epochs - 1:
                final_loss = loss.item()
        
        DeviceDetector.synchronize_device(device)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        return {
            "total_time": total_time,
            "avg_epoch_time": total_time / epochs,
            "final_loss": final_loss,
            "parameters": sum(p.numel() for p in model.parameters())
        }
    
    @staticmethod
    def benchmark_inference(
        device: torch.device,
        batch_size: int = 256,
        iterations: int = 100,
        input_size: int = 1000,
        output_size: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark neural network inference on the specified device.
        
        Args:
            device: Device to run inference on
            batch_size: Batch size for inference
            iterations: Number of inference iterations
            input_size: Input layer size
            output_size: Output layer size
            
        Returns:
            Dict[str, float]: Dictionary containing timing results
        """
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available")
            
        model = NeuralNetworkBenchmark.create_test_network(
            input_size=input_size,
            output_size=output_size
        ).to(device)
        
        model.eval()
        input_data = torch.randn(batch_size, input_size, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_data)
                DeviceDetector.synchronize_device(device)
        
        # Benchmark
        DeviceDetector.synchronize_device(device)
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(iterations):
                output = model(input_data)
        
        DeviceDetector.synchronize_device(device)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        return {
            "total_time": total_time,
            "avg_iteration_time": total_time / iterations,
            "throughput": (batch_size * iterations) / total_time
        }


class PerformanceReporter:
    """
    Performance reporting and formatting utilities.
    
    This class provides methods for formatting and displaying benchmark results
    in a professional, readable format.
    """
    
    @staticmethod
    def format_matrix_results(results: Dict[str, List[Tuple[int, str, float]]]) -> str:
        """
        Format matrix benchmark results into a readable table.
        
        Args:
            results: Dictionary mapping device names to result tuples
                    (size, device_type, time)
        
        Returns:
            str: Formatted results table
        """
        output = []
        output.append("📊 Matrix Multiplication Results:")
        output.append("=" * 70)
        output.append(f"{'Size':<10} {'CPU (s)':<10} {'GPU (s)':<10} {'Speedup':<10} {'Winner':<15}")
        output.append("-" * 70)
        
        cpu_wins = 0
        gpu_wins = 0
        
        # Extract CPU and GPU results
        cpu_results = []
        gpu_results = []
        
        for device_name, device_results in results.items():
            if 'cpu' in device_name.lower():
                cpu_results = device_results
            else:
                gpu_results = device_results
        
        if cpu_results and gpu_results:
            for (size_cpu, _, cpu_time), (size_gpu, _, gpu_time) in zip(cpu_results, gpu_results):
                if size_cpu != size_gpu:
                    continue
                    
                speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
                winner = "🚀 GPU" if speedup > 1.0 else "🖥️ CPU"
                
                if speedup > 1.0:
                    gpu_wins += 1
                else:
                    cpu_wins += 1
                
                # Calculate operations per second
                ops = 2 * size_cpu**3  # Matrix multiplication FLOPs
                cpu_gflops = ops / cpu_time / 1e9
                gpu_gflops = ops / gpu_time / 1e9
                
                output.append(
                    f"{size_cpu}x{size_cpu:<4} "
                    f"{cpu_time:<10.4f} "
                    f"{gpu_time:<10.4f} "
                    f"{speedup:<10.2f}x "
                    f"{winner:<15}"
                )
        
        output.append("-" * 70)
        output.append(f"🏆 Final Score: GPU wins: {gpu_wins}, CPU wins: {cpu_wins}")
        
        if gpu_wins > cpu_wins:
            output.append("🎉 GPU is faster for larger matrices!")
        elif cpu_wins > gpu_wins:
            output.append("🖥️ CPU is faster for these workloads!")
        else:
            output.append("⚖️  It's a tie between CPU and GPU!")
            
        return "\n".join(output)
    
    @staticmethod
    def format_training_results(
        cpu_results: Dict[str, float], 
        gpu_results: Dict[str, float]
    ) -> str:
        """
        Format neural network training results.
        
        Args:
            cpu_results: CPU training results dictionary
            gpu_results: GPU training results dictionary
            
        Returns:
            str: Formatted training results
        """
        cpu_time = cpu_results.get("total_time", 0.0)
        gpu_time = gpu_results.get("total_time", 0.0)
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        
        output = []
        output.append("🧠 Neural Network Training Results:")
        output.append("=" * 50)
        output.append(f"  📊 CPU Results:")
        output.append(f"    • Total time: {cpu_time:.4f}s")
        output.append(f"    • Avg epoch time: {cpu_results.get('avg_epoch_time', 0):.4f}s")
        output.append(f"    • Final loss: {cpu_results.get('final_loss', 0):.4f}")
        output.append(f"    • Parameters: {cpu_results.get('parameters', 0):,}")
        output.append("")
        
        output.append(f"  ⚡ GPU Results:")
        output.append(f"    • Total time: {gpu_time:.4f}s")
        output.append(f"    • Avg epoch time: {gpu_results.get('avg_epoch_time', 0):.4f}s")
        output.append(f"    • Final loss: {gpu_results.get('final_loss', 0):.4f}")
        output.append(f"    • Parameters: {gpu_results.get('parameters', 0):,}")
        output.append("")
        
        if speedup > 1.0:
            output.append(f"  🚀 GPU is {speedup:.2f}x faster for neural network training!")
        else:
            output.append(f"  🖥️ CPU is {1/speedup:.2f}x faster for neural network training!")
            
        return "\n".join(output)
    
    @staticmethod
    def generate_summary_report(
        device_info: Dict[str, Any],
        matrix_results: Dict[str, List[Tuple[int, str, float]]],
        training_results: Dict[str, Dict[str, float]]
    ) -> str:
        """
        Generate a comprehensive summary report.
        
        Args:
            device_info: System and device information
            matrix_results: Matrix benchmark results
            training_results: Training benchmark results
            
        Returns:
            str: Comprehensive report
        """
        report = []
        report.append("# PyTorch GPU vs CPU Performance Benchmark Report")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # System Information
        report.append("## System Information")
        report.append(f"- PyTorch Version: {device_info.get('pytorch_version', 'Unknown')}")
        report.append(f"- CUDA Available: {device_info.get('cuda_available', False)}")
        report.append(f"- MPS Available: {device_info.get('mps_available', False)}")
        report.append(f"- CPU Threads: {device_info.get('cpu_count', 'Unknown')}")
        
        if device_info.get('cuda_available'):
            report.append(f"- CUDA Version: {device_info.get('cuda_version', 'Unknown')}")
            report.append(f"- GPU Count: {device_info.get('gpu_count', 0)}")
            if 'gpu_name' in device_info:
                report.append(f"- GPU Name: {device_info['gpu_name']}")
        
        report.append("")
        
        # Matrix Results
        if matrix_results:
            report.append("## Matrix Multiplication Benchmarks")
            matrix_report = PerformanceReporter.format_matrix_results(matrix_results)
            report.append(f"```\n{matrix_report}\n```")
            report.append("")
        
        # Training Results
        if training_results:
            report.append("## Neural Network Training Benchmarks")
            training_report = PerformanceReporter.format_training_results(
                training_results.get('cpu', {}),
                training_results.get('gpu', {})
            )
            report.append(f"```\n{training_report}\n```")
            report.append("")
        
        report.append("## Recommendations")
        report.append("Based on the benchmark results:")
        report.append("- Use GPU for large matrix operations (>2000x2000)")
        report.append("- Use GPU for neural network training when available")
        report.append("- Consider CPU for small operations or when GPU memory is limited")
        report.append("- Profile your specific workloads for optimal performance")
        
        return "\n".join(report)


def run_matrix_benchmark(
    sizes: Optional[List[int]] = None,
    devices: Optional[List[str]] = None
) -> Dict[str, List[Tuple[int, str, float]]]:
    """
    Run matrix multiplication benchmarks across specified devices and sizes.
    
    This is the main entry point for matrix benchmarking, compatible with the
    PyTorch PR implementation.
    
    Args:
        sizes: List of matrix sizes to benchmark. Defaults to [500, 1000, 2000, 3000, 4000, 5000]
        devices: List of device types to test. Defaults to ['cpu', 'best_gpu']
        
    Returns:
        Dict[str, List[Tuple[int, str, float]]]: Results dictionary mapping
        device names to lists of (size, device_type, time) tuples
    """
    if not HAS_PYTORCH:
        print("❌ PyTorch not available. Cannot run benchmarks.")
        return {}
    
    if sizes is None:
        sizes = [500, 1000, 2000, 3000, 4000, 5000]
    
    detector = DeviceDetector()
    benchmark = MatrixBenchmark()
    
    # Determine devices to test
    cpu_device = torch.device('cpu')
    best_device = detector.get_best_device()
    
    results = {}
    
    print("🚀 PyTorch Matrix Multiplication Benchmark")
    print("=" * 60)
    print(f"Testing devices: CPU, {best_device}")
    print(f"Matrix sizes: {sizes}")
    print()
    
    # CPU benchmarks
    print("🖥️ Running CPU benchmarks...")
    cpu_results = []
    for size in sizes:
        try:
            cpu_time = benchmark.benchmark_matrix_multiplication(size, cpu_device)
            cpu_results.append((size, 'cpu', cpu_time))
            print(f"  {size}x{size}: {cpu_time:.6f}s")
        except Exception as e:
            print(f"  {size}x{size}: Failed ({e})")
            cpu_results.append((size, 'cpu', float('inf')))
    
    results['cpu'] = cpu_results
    
    # GPU benchmarks
    if best_device.type != 'cpu':
        print(f"\n⚡ Running {best_device} benchmarks...")
        gpu_results = []
        for size in sizes:
            try:
                gpu_time = benchmark.benchmark_matrix_multiplication(size, best_device)
                gpu_results.append((size, best_device.type, gpu_time))
                print(f"  {size}x{size}: {gpu_time:.6f}s")
            except Exception as e:
                print(f"  {size}x{size}: Failed ({e})")
                gpu_results.append((size, best_device.type, float('inf')))
        
        results[str(best_device)] = gpu_results
    
    # Display results
    print("\n" + PerformanceReporter.format_matrix_results(results))
    
    return results


def run_training_benchmark(
    epochs: int = 10,
    batch_size: int = 256
) -> Dict[str, Dict[str, float]]:
    """
    Run neural network training benchmarks.
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        Dict[str, Dict[str, float]]: Results dictionary with CPU and GPU results
    """
    if not HAS_PYTORCH:
        print("❌ PyTorch not available. Cannot run benchmarks.")
        return {}
    
    detector = DeviceDetector()
    benchmark = NeuralNetworkBenchmark()
    
    cpu_device = torch.device('cpu')
    best_device = detector.get_best_device()
    
    results = {}
    
    print("🧠 Neural Network Training Benchmark")
    print("=" * 50)
    print(f"Epochs: {epochs}, Batch Size: {batch_size}")
    print()
    
    # CPU training
    print("🖥️ Training on CPU...")
    try:
        cpu_results = benchmark.benchmark_training(
            device=cpu_device,
            epochs=epochs,
            batch_size=batch_size
        )
        results['cpu'] = cpu_results
        print(f"✅ CPU training completed in {cpu_results['total_time']:.4f}s")
    except Exception as e:
        print(f"❌ CPU training failed: {e}")
        results['cpu'] = {"total_time": float('inf'), "avg_epoch_time": 0, "final_loss": 0, "parameters": 0}
    
    # GPU training
    if best_device.type != 'cpu':
        print(f"⚡ Training on {best_device}...")
        try:
            gpu_results = benchmark.benchmark_training(
                device=best_device,
                epochs=epochs,
                batch_size=batch_size
            )
            results['gpu'] = gpu_results
            print(f"✅ GPU training completed in {gpu_results['total_time']:.4f}s")
        except Exception as e:
            print(f"❌ GPU training failed: {e}")
            results['gpu'] = {"total_time": float('inf'), "avg_epoch_time": 0, "final_loss": 0, "parameters": 0}
    
    # Display results
    if 'cpu' in results and 'gpu' in results:
        print("\n" + PerformanceReporter.format_training_results(results['cpu'], results['gpu']))
    
    return results


def main():
    """
    Main function to run comprehensive benchmarks.
    """
    print("🚀 PyTorch GPU vs CPU Performance Benchmark Suite")
    print("=" * 70)
    
    if not HAS_PYTORCH:
        print("❌ PyTorch is not available. Please install PyTorch to run benchmarks.")
        return
    
    # Device information
    detector = DeviceDetector()
    device_info = detector.get_device_info()
    
    print("💻 System Information:")
    print(f"  PyTorch Version: {device_info['pytorch_version']}")
    print(f"  CUDA Available: {device_info['cuda_available']}")
    print(f"  MPS Available: {device_info['mps_available']}")
    print(f"  CPU Threads: {device_info['cpu_count']}")
    
    if device_info['cuda_available']:
        print(f"  CUDA Version: {device_info.get('cuda_version', 'Unknown')}")
        print(f"  GPU Count: {device_info.get('gpu_count', 0)}")
        if 'gpu_name' in device_info:
            print(f"  GPU Name: {device_info['gpu_name']}")
    
    print()
    
    # Run benchmarks
    matrix_results = run_matrix_benchmark()
    print()
    training_results = run_training_benchmark()
    
    # Generate comprehensive report
    print("\n📝 Generating comprehensive report...")
    report = PerformanceReporter.generate_summary_report(
        device_info, matrix_results, training_results
    )
    
    # Save report
    report_path = Path("pytorch_benchmark_report.md")
    report_path.write_text(report, encoding='utf-8')
    print(f"📄 Report saved to: {report_path.absolute()}")
    
    print("\n✅ Benchmark suite completed successfully!")


if __name__ == "__main__":
    main()
