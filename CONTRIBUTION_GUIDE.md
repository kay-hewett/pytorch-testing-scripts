# PyTorch Testing Scripts - Contribution Guide

Welcome to the PyTorch Testing Scripts repository! This guide will help you contribute effectively to our collection of PyTorch performance benchmarking and testing utilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Standards](#code-standards)
- [Benchmarking Guidelines](#benchmarking-guidelines)
- [Testing Requirements](#testing-requirements)
- [Submission Process](#submission-process)
- [Performance Considerations](#performance-considerations)
- [Documentation Standards](#documentation-standards)

## 🎯 Overview

This repository contains comprehensive PyTorch performance benchmarking utilities, including:

- **GPU vs CPU Performance Benchmarks**: Matrix operations, neural network training/inference
- **Cross-Platform Device Detection**: Support for CUDA, MPS (Apple Metal), and CPU
- **Professional Reporting**: Detailed performance analysis with speedup calculations
- **VS Code Integration**: Enhanced development tools with benchmarking capabilities

Our utilities are based on professional-grade code originally developed for PyTorch core (PR #162107, commit `6eb65749b7fc286ad3434a0faaf964f02f245f8e`).

## 🚀 Getting Started

### Prerequisites

```bash
# Required dependencies
pip install torch torchvision torchaudio
pip install json5  # For VS Code settings parsing

# Optional: For advanced reporting
pip install matplotlib seaborn pandas
```

### Repository Structure

```
pytorch-testing-scripts/
├── benchmarks/
│   ├── gpu_cpu_benchmark.py      # Comprehensive benchmarking suite
│   └── __init__.py
├── tools/
│   ├── enhanced_vscode_settings.py  # VS Code integration
│   └── __init__.py
├── docs/
│   ├── INTEGRATION_GUIDE.md
│   └── examples/
├── tests/
│   └── test_benchmarks.py
├── CONTRIBUTION_GUIDE.md
└── README.md
```

## 🛠️ Development Environment

### Setting Up Your Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kay-hewett/pytorch-testing-scripts.git
   cd pytorch-testing-scripts
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python benchmarks/gpu_cpu_benchmark.py
   ```

### Supported Platforms

- **macOS**: Apple Silicon (M1/M2/M3/M4) with MPS support
- **Linux**: CUDA-enabled systems
- **Windows**: CUDA-enabled systems
- **CPU-only**: All platforms (fallback support)

## 📝 Code Standards

### Python Code Quality

We maintain high code quality standards consistent with PyTorch core:

```python
# ✅ Good: Professional class structure with comprehensive docstrings
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
        
        Returns:
            torch.device: The best available device
            
        Raises:
            ImportError: If PyTorch is not available
        """
        # Implementation here...
```

### Type Hints

All functions must include comprehensive type hints:

```python
from typing import Dict, List, Optional, Tuple, Any

def benchmark_matrix_multiplication(
    size: int, 
    device: torch.device, 
    warmup: bool = True,
    iterations: int = 1,
    dtype: torch.dtype = torch.float32
) -> float:
    """Benchmark matrix multiplication with proper typing."""
    pass
```

## ⚡ Benchmarking Guidelines

### Proper GPU Synchronization

**Critical**: Always synchronize GPU operations for accurate timing:

```python
def benchmark_operation(device: torch.device) -> float:
    # Warmup
    for _ in range(3):
        result = operation()
        synchronize_device(device)  # Essential for accurate timing
    
    # Actual benchmark
    synchronize_device(device)
    start_time = time.time()
    
    result = operation()
    
    synchronize_device(device)  # Essential before recording end time
    end_time = time.time()
    
    return end_time - start_time

def synchronize_device(device: torch.device) -> None:
    """Synchronize operations on the specified device."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    # CPU operations are synchronous by default
```

### Benchmark Design Principles

1. **Warmup Iterations**: Always perform warmup runs to account for GPU initialization
2. **Multiple Iterations**: Run multiple iterations for statistical significance
3. **Memory Management**: Clean up GPU memory between benchmarks
4. **Cross-Platform Support**: Test on CUDA, MPS, and CPU devices
5. **Realistic Workloads**: Use representative problem sizes and model architectures

## 🧪 Testing Requirements

### Unit Tests

All benchmarking utilities must include comprehensive tests:

```python
import unittest
import torch
from benchmarks.gpu_cpu_benchmark import DeviceDetector, MatrixBenchmark

class TestDeviceDetector(unittest.TestCase):
    def test_device_detection(self):
        """Test device detection functionality."""
        device = DeviceDetector.get_best_device()
        self.assertIsInstance(device, torch.device)

if __name__ == '__main__':
    unittest.main()
```

## 📤 Submission Process

### Pull Request Guidelines

1. **Branch Naming**: Use descriptive names
   ```bash
   git checkout -b feature/improved-neural-network-benchmarks
   git checkout -b fix/gpu-synchronization-issue
   git checkout -b docs/update-contribution-guide
   ```

2. **Commit Messages**: Follow conventional commits
   ```bash
   git commit -m "feat: add advanced neural network benchmarking suite"
   git commit -m "fix: resolve MPS synchronization timing issue"
   git commit -m "docs: update benchmarking guidelines with examples"
   ```

## 🎯 Performance Considerations

### Optimization Guidelines

1. **Memory Efficiency**: 
   ```python
   # ✅ Good: Clean memory management
   del large_tensor
   torch.cuda.empty_cache() if torch.cuda.is_available() else None
   ```

2. **Numerical Stability**:
   ```python
   # ✅ Good: Use appropriate data types
   x = torch.randn(size, size, dtype=torch.float32, device=device)
   ```

## 🏆 Quality Standards

We maintain the same quality standards as PyTorch core:
- Comprehensive error handling
- Cross-platform compatibility
- Professional documentation
- Performance optimization
- Robust testing coverage

---

**Thank you for contributing to PyTorch Testing Scripts!** Your contributions help the entire PyTorch community build better, faster applications. 🚀

*Based on PyTorch PR #162107 (commit 6eb65749b7fc286ad3434a0faaf964f02f245f8e)*
