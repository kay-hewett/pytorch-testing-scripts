#!/usr/bin/env python3
"""
GPU Performance Test Script for PyTorch
Tests GPU (MPS/CUDA) vs CPU performance with different matrix sizes
"""

import torch
import time

def detect_best_device():
    """Detect the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda'), 'NVIDIA CUDA'
    elif torch.backends.mps.is_available():
        return torch.device('mps'), 'Apple Metal (MPS)'
    else:
        return torch.device('cpu'), 'CPU Only'

def benchmark_matrix_multiplication(size, device, warmup=True):
    """Benchmark matrix multiplication for a given size and device."""
    # Create tensors on the specified device
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)
    
    # Warmup run (important for GPU)
    if warmup and device.type != 'cpu':
        _ = torch.mm(x, y)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
    
    # Actual timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    
    start_time = time.time()
    result = torch.mm(x, y)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    
    end_time = time.time()
    
    return end_time - start_time

def run_performance_comparison():
    """Run comprehensive performance comparison between CPU and GPU."""
    print("=" * 70)
    print("🚀 GPU vs CPU Performance Comparison")
    print("=" * 70)
    
    gpu_device, gpu_name = detect_best_device()
    cpu_device = torch.device('cpu')
    
    print(f"�️  CPU Device: {cpu_device}")
    print(f"⚡ GPU Device: {gpu_device} ({gpu_name})")
    print()
    
    if gpu_device.type == 'cpu':
        print("❌ No GPU acceleration available. Testing CPU only.")
        sizes = [1000, 2000, 3000]
        for size in sizes:
            cpu_time = benchmark_matrix_multiplication(size, cpu_device)
            print(f"Size {size}x{size}: CPU = {cpu_time:.4f}s")
        return
    
    # Test different matrix sizes
    test_sizes = [500, 1000, 2000, 3000, 4000, 5000, 6000, 8000]
    
    print("� Results:")
    print("-" * 70)
    print(f"{'Size':<8} {'CPU (s)':<10} {'GPU (s)':<10} {'Speedup':<10} {'Winner':<10}")
    print("-" * 70)
    
    gpu_wins = 0
    cpu_wins = 0
    
    for size in test_sizes:
        try:
            # CPU benchmark
            cpu_time = benchmark_matrix_multiplication(size, cpu_device, warmup=False)
            
            # GPU benchmark  
            gpu_time = benchmark_matrix_multiplication(size, gpu_device, warmup=True)
            
            # Calculate speedup
            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                if speedup > 1.0:
                    winner = "🚀 GPU"
                    gpu_wins += 1
                else:
                    winner = "🖥️ CPU"
                    cpu_wins += 1
                    speedup = gpu_time / cpu_time  # Show how much slower GPU is
                    
            else:
                speedup = float('inf')
                winner = "🚀 GPU"
                gpu_wins += 1
            
            print(f"{size}x{size:<3} {cpu_time:<10.4f} {gpu_time:<10.4f} {speedup:<10.2f}x {winner}")
            
        except Exception as e:
            print(f"{size}x{size:<3} Error: {str(e)[:40]}")
            continue
    
    print("-" * 70)
    print(f"🏆 Final Score: GPU wins: {gpu_wins}, CPU wins: {cpu_wins}")
    
    if gpu_wins > cpu_wins:
        print("🎉 GPU is faster for larger matrices!")
    else:
        print("💻 CPU dominates for these matrix sizes on your system.")

def test_neural_network_performance():
    """Test neural network training performance."""
    print("\n" + "=" * 70)
    print("🧠 Neural Network Training Performance")
    print("=" * 70)
    
    gpu_device, gpu_name = detect_best_device()
    cpu_device = torch.device('cpu')
    
    if gpu_device.type == 'cpu':
        print("❌ No GPU available for neural network test")
        return
    
    import torch.nn as nn
    import torch.optim as optim
    
    # Create a larger neural network
    class LargeNet(nn.Module):
        def __init__(self):
            super(LargeNet, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(1000, 2000),
                nn.ReLU(),
                nn.Linear(2000, 2000),
                nn.ReLU(),
                nn.Linear(2000, 2000),
                nn.ReLU(),
                nn.Linear(2000, 1000),
                nn.ReLU(),
                nn.Linear(1000, 100)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    batch_size = 256
    epochs = 10
    
    print(f"Training setup:")
    print(f"  • Network: 5 layers (1000→2000→2000→2000→1000→100)")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Epochs: {epochs}")
    print(f"  • Parameters: {sum(p.numel() for p in LargeNet().parameters()):,}")
    print()
    
    # Test on CPU
    print("🖥️  Training on CPU...")
    model_cpu = LargeNet().to(cpu_device)
    optimizer_cpu = optim.Adam(model_cpu.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Create training data
    x_train = torch.randn(batch_size, 1000)
    y_train = torch.randn(batch_size, 100)
    
    start_time = time.time()
    for epoch in range(epochs):
        optimizer_cpu.zero_grad()
        output = model_cpu(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer_cpu.step()
    cpu_training_time = time.time() - start_time
    
    print(f"✅ CPU training completed in {cpu_training_time:.4f}s")
    
    # Test on GPU
    print(f"⚡ Training on {gpu_name}...")
    model_gpu = LargeNet().to(gpu_device)
    optimizer_gpu = optim.Adam(model_gpu.parameters(), lr=0.001)
    
    x_train_gpu = x_train.to(gpu_device)
    y_train_gpu = y_train.to(gpu_device)
    
    # Warmup
    output = model_gpu(x_train_gpu)
    loss = criterion(output, y_train_gpu)
    loss.backward()
    
    if gpu_device.type == 'cuda':
        torch.cuda.synchronize()
    elif gpu_device.type == 'mps':
        torch.mps.synchronize()
    
    start_time = time.time()
    for epoch in range(epochs):
        optimizer_gpu.zero_grad()
        output = model_gpu(x_train_gpu)
        loss = criterion(output, y_train_gpu)
        loss.backward()
        optimizer_gpu.step()
    
    if gpu_device.type == 'cuda':
        torch.cuda.synchronize()
    elif gpu_device.type == 'mps':
        torch.mps.synchronize()
        
    gpu_training_time = time.time() - start_time
    
    print(f"✅ GPU training completed in {gpu_training_time:.4f}s")
    
    # Compare results
    speedup = cpu_training_time / gpu_training_time
    print("\n📊 Neural Network Training Results:")
    print(f"  • CPU time: {cpu_training_time:.4f}s")
    print(f"  • GPU time: {gpu_training_time:.4f}s")
    if speedup > 1.0:
        print(f"  🚀 GPU is {speedup:.2f}x faster for neural network training!")
    else:
        print(f"  🖥️ CPU is {1/speedup:.2f}x faster for neural network training")

def main():
    """Run all performance tests."""
    print("🚀 PyTorch GPU Performance Test Suite")
    print(f"PyTorch version: {torch.__version__}")
    
    # Run matrix multiplication benchmark
    run_performance_comparison()
    
    # Run neural network benchmark
    test_neural_network_performance()
    
    print("\n" + "=" * 70)
    print("✅ Performance testing complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
