# System Requirements

This document outlines the hardware and software requirements for running the local LLM analysis experiments.

## Table of Contents

- [1. Hardware Requirements](#1-hardware-requirements)
  - [1.1 Minimum Requirements](#11-minimum-requirements)
  - [1.2 Recommended Requirements](#12-recommended-requirements)
  - [1.3 GPU Requirements by Model Size](#13-gpu-requirements-by-model-size)
- [2. Software Requirements](#2-software-requirements)
  - [2.1 Core Dependencies](#21-core-dependencies)
  - [2.2 Operating System Requirements](#22-operating-system-requirements)
  - [2.3 CUDA Requirements (for GPU acceleration)](#23-cuda-requirements-for-gpu-acceleration)
  - [2.4 Python Package Requirements](#24-python-package-requirements)
- [3. Network Requirements](#3-network-requirements)
- [4. Storage Considerations](#4-storage-considerations)
  - [4.1 Model Storage](#41-model-storage)
  - [4.2 Experiment Data](#42-experiment-data)
- [5. Performance Expectations](#5-performance-expectations)
  - [5.1 CPU-only Performance](#51-cpu-only-performance)
  - [5.2 GPU Performance](#52-gpu-performance)
- [6. Troubleshooting Common Issues](#6-troubleshooting-common-issues)
  - [6.1 Out of Memory (OOM)](#61-out-of-memory-oom)
  - [6.2 Slow Performance](#62-slow-performance)
  - [6.3 Installation Issues](#63-installation-issues)

## 1. Hardware Requirements

### 1.1 Minimum Requirements
- **CPU**: Modern multi-core processor (Intel i5/AMD Ryzen 5 or equivalent)
- **RAM**: 16 GB system memory
- **Storage**: 50 GB free space for models and experiment data
- **GPU**: Optional but highly recommended for acceptable performance

### 1.2 Recommended Requirements
- **CPU**: High-performance multi-core processor (Intel i7/i9 or AMD Ryzen 7/9)
- **RAM**: 32 GB or more system memory
- **Storage**: 100 GB+ free space (SSD preferred for faster model loading)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070/4060 Ti or better)

### 1.3 GPU Requirements by Model Size

| Model Size | Minimum VRAM | Recommended VRAM | Example GPUs |
|------------|--------------|------------------|--------------|
| 1B-3B      | 4 GB         | 6 GB            | GTX 1660 Ti, RTX 3060 |
| 7B         | 8 GB         | 12 GB           | RTX 3070, RTX 4060 Ti |
| 13B        | 12 GB        | 16 GB           | RTX 3080, RTX 4070 Ti |
| 30B+       | 24 GB        | 32 GB           | RTX 3090, RTX 4090 |

**Note**: Requirements may vary based on:
- Precision settings (float16 vs float32)
- Batch size and sequence length
- Additional memory for hidden state extraction

## 2. Software Requirements

### 2.1 Core Dependencies
- **Python**: 3.8 or higher (3.10+ recommended)

### 2.2 Operating System Requirements

#### 🥇 Linux (Highly Recommended)
- **Ubuntu 20.04+** (most stable)
- **CentOS 8+** / **RHEL 8+** (enterprise environments)
- **Arch Linux** (advanced users)

#### 🖥️ Windows
**Strongly Recommended**: **WSL2 + Ubuntu 22.04**
- Linux environment compatibility with maximum stability
- CUDA-WSL support for full GPU utilization
- Improved development efficiency and easier troubleshooting

**Alternative**: **Windows 10/11 Native**
- Potential compatibility issues with some ML libraries
- PowerShell and cmd usage requirements
- Path management and filesystem differences

#### 🍎 macOS
**Intel Mac**: **macOS 10.15+** (full support)

**Apple Silicon (M1/M2/M3)**: **macOS 11.0+**
- Metal Performance Shaders (MPS) support
- ARM64 architecture - verify library compatibility
- Rosetta 2 fallback for Intel-based libraries

### 2.3 CUDA Requirements (for GPU acceleration)
- **CUDA Toolkit**: 11.8 or 12.1+
- **cuDNN**: Compatible version with CUDA
- **NVIDIA Driver**: Latest stable version (515+ for CUDA 11.8)

### 2.4 Python Package Requirements
See `experiments/color_generation/requirements.txt` for specific versions:
- PyTorch 1.9.0+
- Transformers 4.21.0+
- Additional ML and analysis libraries

## 3. Network Requirements
- **Internet connection** for:
  - Initial model downloads from HuggingFace Hub
  - Package installation via pip
- **Estimated bandwidth**: 5-50 GB for model downloads (depending on model size)

## 4. Storage Considerations

### 4.1 Model Storage
- **Local cache**: Models are cached locally after first download
- **Cache location**: `~/.cache/huggingface/` (can be configured)
- **Size estimates**:
  - 1B models: ~2-4 GB
  - 7B models: ~13-26 GB
  - 13B models: ~25-50 GB

### 4.2 Experiment Data
- **Results storage**: Configurable via `config.yaml`
- **Hidden states**: Can be memory-intensive (several GB per experiment)
- **Logs and outputs**: Typically < 1 GB

## 5. Performance Expectations

### 5.1 CPU-only Performance
- **Model loading**: 1-5 minutes
- **Generation**: 10-100 tokens/second (varies by model size)
- **Hidden state extraction**: Significant overhead, 2-5x slower

### 5.2 GPU Performance
- **Model loading**: 30 seconds - 2 minutes
- **Generation**: 50-500+ tokens/second
- **Hidden state extraction**: Minimal overhead with sufficient VRAM

## 6. Troubleshooting Common Issues

### 6.1 Out of Memory (OOM)
- Reduce model size or use quantization
- Lower batch size in config
- Use CPU offloading if available
- Close other applications consuming memory/VRAM

### 6.2 Slow Performance
- Ensure GPU utilization (`nvidia-smi`)
- Check CUDA/PyTorch installation
- Consider model quantization
- Use faster storage (SSD)

### 6.3 Installation Issues
- Verify Python and CUDA versions compatibility
- Use virtual environments to avoid conflicts
- Check firewall/proxy settings for downloads