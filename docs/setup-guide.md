# Environment Setup Guide

This guide provides step-by-step instructions for setting up your development environment to run local LLM analysis experiments.

## Table of Contents

- [1. Prerequisites](#1-prerequisites)
- [2. Quick Start](#2-quick-start)
  - [2.1 Clone the Repository](#21-clone-the-repository)
  - [2.2 Navigate to Experiment Directory](#22-navigate-to-experiment-directory)
  - [2.3 Set Up Python Environment](#23-set-up-python-environment)
  - [2.4 Verify Installation](#24-verify-installation)
- [3. Detailed Setup Instructions](#3-detailed-setup-instructions)
  - [3.1 Python Environment Management](#31-python-environment-management)
  - [3.2 GPU Setup (Recommended)](#32-gpu-setup-recommended)
  - [3.3 HuggingFace Hub Setup](#33-huggingface-hub-setup)
  - [3.4 Configuration](#34-configuration)
  - [3.5 Testing Your Setup](#35-testing-your-setup)
- [4. Troubleshooting](#4-troubleshooting)
  - [4.1 Common Issues](#41-common-issues)
  - [4.2 Performance Optimization](#42-performance-optimization)
  - [4.3 Getting Help](#43-getting-help)

## 1. Prerequisites

Before starting, ensure your system meets the [requirements](requirements.md).

## 2. Quick Start

### 2.1 Clone the Repository
```bash
git clone <repository-url>
cd local-llm-analysis
```

### 2.2 Navigate to Experiment Directory
```bash
cd experiments/color_generation
```

### 2.3 Set Up Python Environment
Choose one of the following methods:

#### Option A: Using venv (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using conda
```bash
# Create conda environment
conda create -n llm-analysis python=3.10
conda activate llm-analysis

# Install dependencies
pip install -r requirements.txt
```

### 2.4 Verify Installation
```bash
python scripts/test_generation.py --help
```

## 3. Detailed Setup Instructions

### 3.1 Python Environment Management

#### Virtual Environment Best Practices
- Always use a virtual environment to avoid package conflicts
- Use Python 3.10+ for best compatibility with modern ML libraries
- Keep your base Python installation clean

#### Installing Python Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install packages
pip install -r requirements.txt

# Verify key packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 3.2 GPU Setup (Recommended)

#### CUDA Installation
1. **Check GPU compatibility**:
   ```bash
   nvidia-smi
   ```

2. **Install CUDA Toolkit**:
   - Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Select your OS and follow installation instructions
   - Recommended: CUDA 11.8 or 12.1+

3. **Install cuDNN** (if required):
   - Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   - Follow installation instructions for your OS

#### PyTorch GPU Support
```bash
# Uninstall CPU-only PyTorch if installed
pip uninstall torch torchvision torchaudio

# Install PyTorch with CUDA support
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3.3 HuggingFace Hub Setup

#### Authentication (Optional but Recommended)
Some models require authentication or provide faster downloads:

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login to HuggingFace Hub
huggingface-cli login
# Follow prompts to enter your token
```

#### Cache Configuration
```bash
# Set custom cache directory (optional)
export TRANSFORMERS_CACHE="/path/to/your/cache"
export HF_HOME="/path/to/your/cache"

# Check current cache location
python -c "from transformers import file_utils; print(file_utils.TRANSFORMERS_CACHE)"
```

### 3.4 Configuration

#### Model Configuration
Edit `config.yaml` to match your system:

```yaml
model:
  backend: "transformers"
  model_name: "microsoft/DialoGPT-medium"  # Change to your chosen model
  device: "auto"  # auto, cpu, cuda, cuda:0
  torch_dtype: "float16"  # float32 for CPU, float16 for GPU
  load_in_8bit: false  # Set true to save VRAM if supported
```

#### Memory Optimization Settings
For systems with limited VRAM:

```yaml
model:
  torch_dtype: "float16"  # Reduces memory usage
  load_in_8bit: true      # Further reduces memory usage
  device_map: "auto"      # Automatic device placement

generation:
  max_new_tokens: 20      # Reduce for less memory usage
```

### 3.5 Testing Your Setup

#### Basic Functionality Test
```bash
# Test RGB tool
python src/tools/rgb.py 255 0 0

# Test basic configuration
python scripts/test_generation.py
```

#### GPU Performance Test
```bash
# Monitor GPU usage during test
nvidia-smi -l 1  # Run in separate terminal

# Run test with GPU monitoring
python scripts/test_generation.py --model microsoft/DialoGPT-medium
```

## 4. Troubleshooting

### 4.1 Common Issues

#### Import Errors
```bash
# Issue: ModuleNotFoundError
# Solution: Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Verify environment
which python
pip list
```

#### CUDA Issues
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# If false, reinstall PyTorch with CUDA support
```

#### Memory Issues
```bash
# Monitor memory usage
# Linux:
free -h
nvidia-smi

# Windows:
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /format:list
```

### 4.2 Performance Optimization

#### For Limited VRAM
- Use `torch_dtype: "float16"`
- Enable `load_in_8bit: true`
- Reduce `max_new_tokens`
- Use smaller models

#### For Limited RAM
- Close unnecessary applications
- Use swap/page file if available
- Consider CPU-only execution for small models

#### For Slow Storage
- Move cache to SSD if available
- Pre-download models during setup

### 4.3 Getting Help

If you encounter issues:

1. Check the [troubleshooting section](requirements.md#troubleshooting-common-issues) in requirements.md
2. Verify your system meets the minimum requirements
3. Check that all dependencies are correctly installed
4. Review error messages carefully - they often contain specific guidance
5. Create an issue in the repository with:
   - Your system specifications
   - Full error message
   - Steps to reproduce the problem