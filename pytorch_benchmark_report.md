# PyTorch GPU vs CPU Performance Benchmark Report
Generated on: 2025-09-04 03:55:52

## System Information
- PyTorch Version: 2.8.0
- CUDA Available: False
- MPS Available: True
- CPU Threads: 10

## Matrix Multiplication Benchmarks
```
📊 Matrix Multiplication Results:
======================================================================
Size       CPU (s)    GPU (s)    Speedup    Winner         
----------------------------------------------------------------------
500x500  0.0002     0.0007     0.25      x 🖥️ CPU         
1000x1000 0.0008     0.0017     0.49      x 🖥️ CPU         
2000x2000 0.0052     0.0027     1.91      x 🚀 GPU          
3000x3000 0.0198     0.0085     2.34      x 🚀 GPU          
4000x4000 0.0429     0.0194     2.21      x 🚀 GPU          
5000x5000 0.0876     0.0386     2.27      x 🚀 GPU          
----------------------------------------------------------------------
🏆 Final Score: GPU wins: 4, CPU wins: 2
🎉 GPU is faster for larger matrices!
```

## Neural Network Training Benchmarks
```
🧠 Neural Network Training Results:
==================================================
  📊 CPU Results:
    • Total time: 0.2410s
    • Avg epoch time: 0.0241s
    • Final loss: 0.3796
    • Parameters: 12,107,100

  ⚡ GPU Results:
    • Total time: 0.0760s
    • Avg epoch time: 0.0076s
    • Final loss: 0.6566
    • Parameters: 12,107,100

  🚀 GPU is 3.17x faster for neural network training!
```

## Recommendations
Based on the benchmark results:
- Use GPU for large matrix operations (>2000x2000)
- Use GPU for neural network training when available
- Consider CPU for small operations or when GPU memory is limited
- Profile your specific workloads for optimal performance