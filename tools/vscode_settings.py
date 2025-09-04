#!/usr/bin/env python3
"""
Enhanced VS Code settings manager with PyTorch performance benchmarking integration.
This combines the original VS Code settings management with GPU/CPU benchmarking utilities
from the PyTorch pull request commit 6eb65749b7fc286ad3434a0faaf964f02f245f8e.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    # VS Code settings allow comments and trailing commas, which are not valid JSON.
    import json5 as json  # type: ignore[import]
    HAS_JSON5 = True
except ImportError:
    import json  # type: ignore[no-redef]
    HAS_JSON5 = False

try:
    import torch
    import torch.nn as nn
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False


ROOT_FOLDER = Path(__file__).absolute().parent.parent
VSCODE_FOLDER = ROOT_FOLDER / ".vscode"
RECOMMENDED_SETTINGS = VSCODE_FOLDER / "settings_recommended.json"
SETTINGS = VSCODE_FOLDER / "settings.json"


class DeviceDetector:
    """Detects and manages the best available device for PyTorch operations."""
    
    @staticmethod
    def get_best_device() -> torch.device:
        """Get the best available device (CUDA > MPS > CPU)."""
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
        """Get comprehensive device information."""
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
            info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
            
        return info


class MatrixBenchmark:
    """Matrix operation benchmarking utilities."""
    
    @staticmethod
    def benchmark_matrix_multiplication(
        size: int, 
        device: torch.device, 
        warmup: bool = True
    ) -> float:
        """Benchmark matrix multiplication on specified device."""
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available")
            
        # Create test matrices
        x = torch.randn(size, size, device=device)
        y = torch.randn(size, size, device=device)
        
        # Warmup
        if warmup:
            for _ in range(3):
                torch.mm(x, y)
                if device.type in ['cuda', 'mps']:
                    torch.cuda.synchronize() if device.type == 'cuda' else torch.mps.synchronize()
        
        # Benchmark
        if device.type in ['cuda', 'mps']:
            torch.cuda.synchronize() if device.type == 'cuda' else torch.mps.synchronize()
            
        start_time = time.time()
        result = torch.mm(x, y)
        
        if device.type in ['cuda', 'mps']:
            torch.cuda.synchronize() if device.type == 'cuda' else torch.mps.synchronize()
            
        end_time = time.time()
        
        return end_time - start_time


class NeuralNetworkBenchmark:
    """Neural network benchmarking utilities."""
    
    @staticmethod
    def create_test_network(input_size: int = 1000, output_size: int = 100) -> nn.Module:
        """Create a test neural network for benchmarking."""
        return nn.Sequential(
            nn.Linear(input_size, 2000),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, output_size)
        )
    
    @staticmethod
    def benchmark_training(
        device: torch.device,
        epochs: int = 10,
        batch_size: int = 256
    ) -> float:
        """Benchmark neural network training."""
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available")
            
        # Create model and data
        model = NeuralNetworkBenchmark.create_test_network().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create dummy data
        input_data = torch.randn(batch_size, 1000, device=device)
        target = torch.randint(0, 100, (batch_size,), device=device)
        
        # Warmup
        for _ in range(2):
            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        if device.type in ['cuda', 'mps']:
            torch.cuda.synchronize() if device.type == 'cuda' else torch.mps.synchronize()
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(epochs):
            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        if device.type in ['cuda', 'mps']:
            torch.cuda.synchronize() if device.type == 'cuda' else torch.mps.synchronize()
            
        end_time = time.time()
        
        return end_time - start_time


class PerformanceReporter:
    """Performance reporting and formatting utilities."""
    
    @staticmethod
    def format_matrix_results(results: Dict[str, List[Tuple[int, str, float]]]) -> str:
        """Format matrix benchmark results."""
        output = []
        output.append("📊 Matrix Multiplication Results:")
        output.append("-" * 70)
        output.append(f"{'Size':<10} {'CPU (s)':<10} {'GPU (s)':<10} {'Speedup':<10} {'Winner':<10}")
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
                
                output.append(f"{size_cpu}x{size_cpu:<4} {cpu_time:<10.4f} {gpu_time:<10.4f} {speedup:<10.2f}x {winner}")
        
        output.append("-" * 70)
        output.append(f"🏆 Final Score: GPU wins: {gpu_wins}, CPU wins: {cpu_wins}")
        
        if gpu_wins > cpu_wins:
            output.append("🎉 GPU is faster for larger matrices!")
        else:
            output.append("🖥️ CPU is faster for these workloads!")
            
        return "\n".join(output)
    
    @staticmethod
    def format_training_results(cpu_time: float, gpu_time: float) -> str:
        """Format neural network training results."""
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        
        output = []
        output.append("🧠 Neural Network Training Results:")
        output.append(f"  • CPU time: {cpu_time:.4f}s")
        output.append(f"  • GPU time: {gpu_time:.4f}s")
        
        if speedup > 1.0:
            output.append(f"  🚀 GPU is {speedup:.2f}x faster for neural network training!")
        else:
            output.append(f"  🖥️ CPU is {1/speedup:.2f}x faster for neural network training!")
            
        return "\n".join(output)


class EnhancedVSCodeManager:
    """Enhanced VS Code settings manager with PyTorch benchmarking."""
    
    def __init__(self):
        self.device_detector = DeviceDetector()
        self.matrix_benchmark = MatrixBenchmark()
        self.nn_benchmark = NeuralNetworkBenchmark()
        self.reporter = PerformanceReporter()
    
    def deep_update(self, d: dict, u: dict) -> dict:  # type: ignore[type-arg]
        """Deep update dictionary (original VS Code settings functionality)."""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self.deep_update(d.get(k, {}), v)
            elif isinstance(v, list):
                d[k] = d.get(k, []) + v
            else:
                d[k] = v
        return d
    
    def update_vscode_settings(self) -> None:
        """Update VS Code settings (original functionality)."""
        if not RECOMMENDED_SETTINGS.exists():
            print("⚠️ No recommended settings file found")
            return
            
        recommended_settings = json.loads(RECOMMENDED_SETTINGS.read_text())
        
        try:
            current_settings_text = SETTINGS.read_text()
        except FileNotFoundError:
            current_settings_text = "{}"
        
        try:
            current_settings = json.loads(current_settings_text)
        except ValueError as ex:
            if HAS_JSON5:
                raise SystemExit("Failed to parse .vscode/settings.json.") from ex
            raise SystemExit(
                "Failed to parse .vscode/settings.json. "
                "Maybe it contains comments or trailing commas. "
                "Try `pip install json5` to install an extended JSON parser."
            ) from ex
        
        settings = self.deep_update(current_settings, recommended_settings)
        
        SETTINGS.write_text(
            json.dumps(settings, indent=4) + "\n",
            encoding="utf-8",
        )
        print("✅ VS Code settings updated successfully")
    
    def run_performance_benchmark(self, sizes: Optional[List[int]] = None) -> Dict[str, Any]:
        """Run comprehensive performance benchmark."""
        if not HAS_PYTORCH:
            return {"error": "PyTorch not available"}
        
        if sizes is None:
            sizes = [500, 1000, 2000, 3000, 4000, 5000]
        
        print("🚀 PyTorch GPU vs CPU Performance Benchmark")
        print("=" * 70)
        
        # Device info
        device_info = self.device_detector.get_device_info()
        print(f"PyTorch version: {device_info['pytorch_version']}")
        
        cpu_device = torch.device('cpu')
        best_device = self.device_detector.get_best_device()
        
        print(f"🖥️ CPU Device: {cpu_device}")
        print(f"⚡ GPU Device: {best_device}")
        print()
        
        # Matrix benchmarks
        cpu_results = []
        gpu_results = []
        
        for size in sizes:
            print(f"Benchmarking {size}x{size} matrices...")
            
            cpu_time = self.matrix_benchmark.benchmark_matrix_multiplication(size, cpu_device)
            gpu_time = self.matrix_benchmark.benchmark_matrix_multiplication(size, best_device)
            
            cpu_results.append((size, cpu_device.type, cpu_time))
            gpu_results.append((size, best_device.type, gpu_time))
        
        # Format and display matrix results
        results = {"cpu": cpu_results, str(best_device): gpu_results}
        matrix_report = self.reporter.format_matrix_results(results)
        print(matrix_report)
        print()
        
        # Neural network benchmark
        print("🧠 Neural Network Training Performance")
        print("=" * 70)
        
        print("🖥️ Training on CPU...")
        cpu_nn_time = self.nn_benchmark.benchmark_training(cpu_device)
        print(f"✅ CPU training completed in {cpu_nn_time:.4f}s")
        
        print(f"⚡ Training on {best_device}...")
        gpu_nn_time = self.nn_benchmark.benchmark_training(best_device)
        print(f"✅ GPU training completed in {gpu_nn_time:.4f}s")
        print()
        
        # Format and display training results
        training_report = self.reporter.format_training_results(cpu_nn_time, gpu_nn_time)
        print(training_report)
        print()
        
        return {
            "device_info": device_info,
            "matrix_results": results,
            "training_results": {
                "cpu_time": cpu_nn_time,
                "gpu_time": gpu_nn_time
            }
        }
    
    def generate_performance_report(self, output_file: Optional[Path] = None) -> str:
        """Generate a comprehensive performance report."""
        if not HAS_PYTORCH:
            return "PyTorch not available for benchmarking"
        
        results = self.run_performance_benchmark()
        
        report = []
        report.append("# PyTorch Performance Report")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if "device_info" in results:
            info = results["device_info"]
            report.append("## System Information")
            report.append(f"- PyTorch Version: {info.get('pytorch_version', 'Unknown')}")
            report.append(f"- CUDA Available: {info.get('cuda_available', False)}")
            report.append(f"- MPS Available: {info.get('mps_available', False)}")
            report.append(f"- CPU Threads: {info.get('cpu_count', 'Unknown')}")
            report.append("")
        
        if "matrix_results" in results:
            report.append("## Matrix Multiplication Benchmarks")
            matrix_report = self.reporter.format_matrix_results(results["matrix_results"])
            report.append(f"```\n{matrix_report}\n```")
            report.append("")
        
        if "training_results" in results:
            training = results["training_results"]
            report.append("## Neural Network Training Benchmarks")
            training_report = self.reporter.format_training_results(
                training["cpu_time"], 
                training["gpu_time"]
            )
            report.append(f"```\n{training_report}\n```")
            report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            output_file.write_text(report_text, encoding="utf-8")
            print(f"📝 Report saved to: {output_file}")
        
        return report_text


def run_matrix_benchmark(sizes: Optional[List[int]] = None) -> Dict[str, List[Tuple[int, str, float]]]:
    """Standalone function to run matrix benchmarks (PyTorch PR compatibility)."""
    manager = EnhancedVSCodeManager()
    results = manager.run_performance_benchmark(sizes)
    return results.get("matrix_results", {})


def main() -> None:
    """Main function with command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced VS Code settings manager with PyTorch benchmarking"
    )
    parser.add_argument(
        "--benchmark", 
        action="store_true", 
        help="Run PyTorch performance benchmark"
    )
    parser.add_argument(
        "--update-settings", 
        action="store_true", 
        help="Update VS Code settings"
    )
    parser.add_argument(
        "--report", 
        metavar="FILE", 
        help="Generate performance report to file"
    )
    parser.add_argument(
        "--sizes", 
        nargs="+", 
        type=int, 
        help="Matrix sizes for benchmarking"
    )
    
    args = parser.parse_args()
    
    manager = EnhancedVSCodeManager()
    
    # Default behavior: update settings if available, then run benchmark
    if not any([args.benchmark, args.update_settings, args.report]):
        if RECOMMENDED_SETTINGS.exists():
            args.update_settings = True
        args.benchmark = True
    
    if args.update_settings:
        manager.update_vscode_settings()
    
    if args.benchmark:
        manager.run_performance_benchmark(args.sizes)
    
    if args.report:
        report_file = Path(args.report)
        manager.generate_performance_report(report_file)


if __name__ == "__main__":
    main()
