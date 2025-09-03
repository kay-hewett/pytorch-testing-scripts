#!/usr/bin/env python3
"""
Simple PyTorch test script to verify installation and basic functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def test_pytorch_basic():
    """Test basic PyTorch functionality."""
    print("🔥 PyTorch Version:", torch.__version__)
    print("🐍 Python Version:", torch.version.cuda if torch.cuda.is_available() else "CPU Only")
    print("💻 CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("🚀 CUDA Version:", torch.version.cuda)
        print("📊 GPU Count:", torch.cuda.device_count())
    print()

def test_tensor_operations():
    """Test basic tensor operations."""
    print("🧮 Testing Tensor Operations:")
    
    # Create tensors
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)
    
    print(f"  • Tensor x shape: {x.shape}")
    print(f"  • Tensor y shape: {y.shape}")
    
    # Basic operations
    z = x + y
    print(f"  • Addition result shape: {z.shape}")
    
    # Matrix multiplication
    a = torch.randn(3, 4)
    b = torch.randn(4, 2)
    c = torch.mm(a, b)
    print(f"  • Matrix multiplication ({a.shape} x {b.shape}) = {c.shape}")
    print()

def test_autograd():
    """Test automatic differentiation."""
    print("🎯 Testing Autograd:")
    
    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 2 + 3 * x + 1
    
    print(f"  • Input: x = {x.item()}")
    print("  • Function: y = x² + 3x + 1")
    print(f"  • Output: y = {y.item()}")
    
    y.backward()
    print(f"  • Gradient: dy/dx = {x.grad.item()} (expected: 2x + 3 = 7)")
    print()

def test_neural_network():
    """Test a simple neural network."""
    print("🧠 Testing Neural Network:")
    
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(4, 10)
            self.fc2 = nn.Linear(10, 3)
            self.dropout = nn.Dropout(0.1)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return F.softmax(x, dim=1)
    
    # Create model and test data
    model = SimpleNet()
    input_data = torch.randn(5, 4)  # batch_size=5, features=4
    
    # Forward pass
    output = model(input_data)
    print(f"  • Input shape: {input_data.shape}")
    print(f"  • Output shape: {output.shape}")
    print(f"  • Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test training step
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Dummy target
    target = torch.randint(0, 3, (5,))
    
    # Training step
    optimizer.zero_grad()
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"  • Loss: {loss.item():.4f}")
    print()

def test_cuda_if_available():
    """Test CUDA functionality if available."""
    if torch.cuda.is_available():
        print("🚀 Testing CUDA:")
        
        # Create tensors on GPU
        device = torch.device('cuda')
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # GPU computation
        z = torch.mm(x, y)
        print("  • GPU tensor computation successful")
        print(f"  • Result shape: {z.shape}")
        print(f"  • Result device: {z.device}")
        
        # Move back to CPU
        z_cpu = z.cpu()
        print(f"  • Moved to CPU: {z_cpu.device}")
        print()
    else:
        print("ℹ️  CUDA not available - running on CPU only")
        print()

def main():
    """Run all tests."""
    print("=" * 50)
    print("🔥 PyTorch Installation Test")
    print("=" * 50)
    
    try:
        test_pytorch_basic()
        test_tensor_operations()
        test_autograd()
        test_neural_network()
        test_cuda_if_available()
        
        print("✅ All tests passed! PyTorch is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
