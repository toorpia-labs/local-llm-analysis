# OS-Specific Installation Instructions

This guide provides detailed installation instructions for different operating systems.

## Linux (Ubuntu/Debian)

### Prerequisites Installation

#### System Updates
```bash
sudo apt update && sudo apt upgrade -y
```

#### Python Development Tools
```bash
# Install Python and development tools
sudo apt install python3.10 python3.10-venv python3.10-dev python3-pip git -y

# Alternative: Use deadsnakes PPA for latest Python
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev -y
```

#### NVIDIA GPU Setup (if applicable)
```bash
# Install NVIDIA drivers
sudo ubuntu-drivers autoinstall

# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA Toolkit
sudo apt-get install cuda-toolkit-12-1

# Add CUDA to PATH (add to ~/.bashrc)
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd local-llm-analysis/experiments/color_generation

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install PyTorch with CUDA (if GPU available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verification
```bash
# Test installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python scripts/test_generation.py --help
```

## Windows

**IMPORTANT**: For Windows users, we **strongly recommend WSL2 + Ubuntu** for the best compatibility and performance. Native Windows installation is provided as an alternative.

### Option A: WSL2 + Ubuntu (Strongly Recommended) {#wsl2-setup}

WSL2 provides a Linux environment on Windows with excellent performance and compatibility.

#### 1. Enable WSL2
```powershell
# Run PowerShell as Administrator
wsl --install
# This installs WSL2 and Ubuntu by default
```

#### 2. Install Ubuntu 22.04
```powershell
# If you need a specific version
wsl --install -d Ubuntu-22.04
```

#### 3. Setup Ubuntu Environment
```bash
# Inside WSL2 Ubuntu terminal
sudo apt update && sudo apt upgrade -y

# Install Python and development tools
sudo apt install python3.10 python3.10-venv python3.10-dev python3-pip git -y
```

#### 4. GPU Support (CUDA-WSL)
```bash
# Install NVIDIA driver on Windows host (not in WSL2)
# Download from: https://www.nvidia.com/drivers/

# In WSL2, verify CUDA-WSL support
nvidia-smi  # Should show GPU information

# Install CUDA toolkit in WSL2
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-1
```

#### 5. VS Code Integration
```bash
# Install VS Code on Windows, then add WSL extension
# VS Code will automatically detect and connect to WSL2
```

### Option B: Native Windows Installation

#### Prerequisites Installation

#### Python Installation
1. Download Python 3.10+ from [python.org](https://www.python.org/downloads/)
2. **Important**: Check "Add Python to PATH" during installation
3. Verify installation:
   ```cmd
   python --version
   pip --version
   ```

#### Git Installation
1. Download Git from [git-scm.com](https://git-scm.com/download/win)
2. Use default settings during installation
3. Verify: `git --version`

#### Visual Studio Build Tools (Required for some packages)
1. Download from [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/)
2. Install "Build Tools for Visual Studio"
3. Select "C++ build tools" workload

#### NVIDIA GPU Setup (if applicable)
1. **NVIDIA Drivers**:
   - Download latest drivers from [NVIDIA](https://www.nvidia.com/drivers/)
   - Install and restart

2. **CUDA Toolkit**:
   - Download CUDA 12.1+ from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Select Windows → x86_64 → exe (network)
   - Follow installer instructions

3. **Verification**:
   ```cmd
   nvcc --version
   nvidia-smi
   ```

### Environment Setup
```cmd
REM Clone repository
git clone <repository-url>
cd local-llm-analysis\experiments\color_generation

REM Create virtual environment
python -m venv venv
venv\Scripts\activate

REM Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Install PyTorch with CUDA (if GPU available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### PowerShell Execution Policy (if needed)
If you encounter execution policy errors:
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Verification
```cmd
REM Test installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python scripts\test_generation.py --help
```

## macOS

macOS installation varies depending on your hardware. Both Intel and Apple Silicon Macs are supported.

### Check Your Mac Type
```bash
# Check your Mac architecture
uname -m
# Output: x86_64 (Intel) or arm64 (Apple Silicon)
```

### Prerequisites Installation

#### Homebrew Installation
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Python and Development Tools
```bash
# Install Python and Git
brew install python@3.10 git

# Verify installation
python3 --version
git --version
```

#### Xcode Command Line Tools
```bash
xcode-select --install
```

### Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd local-llm-analysis/experiments/color_generation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install PyTorch (CPU-only for macOS)
pip install torch torchvision torchaudio
```

### Apple Silicon (M1/M2/M3) Considerations

Apple Silicon Macs provide excellent performance for ML workloads with proper configuration.

#### Metal Performance Shaders (MPS)
PyTorch supports Apple's Metal Performance Shaders for GPU acceleration:

```bash
# Verify MPS support
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

#### Configuration for Apple Silicon
Update your `config.yaml`:
```yaml
model:
  device: "mps"  # Use Apple Metal Performance Shaders
  torch_dtype: "float32"  # MPS requires float32
  load_in_8bit: false     # Not supported on MPS
```

#### Performance Optimization
```bash
# Monitor unified memory usage
top -pid $(pgrep Python)

# Activity Monitor can also show memory pressure
# Keep memory pressure in green/yellow zones
```

#### Known Limitations
- **float16 precision**: Not fully supported, use float32
- **Quantization**: 8-bit/4-bit quantization may not work
- **Some libraries**: May fall back to CPU for unsupported operations

#### Troubleshooting
```bash
# If MPS issues occur, fall back to CPU
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Check for ARM64 vs x86_64 library conflicts
python -c "import torch; print(f'PyTorch built for: {torch.__version__}')"
```

### Verification
```bash
# Test installation
python -c "import torch; print(torch.__version__)"
python scripts/test_generation.py --help
```

## Docker Setup (Cross-Platform)

For consistent environments across systems:

### Dockerfile
```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY experiments/color_generation/requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy experiment code
COPY experiments/color_generation/ .

# Set entrypoint
CMD ["python", "scripts/test_generation.py"]
```

### Docker Commands
```bash
# Build image
docker build -t llm-analysis .

# Run container
docker run -it --rm llm-analysis

# With GPU support (Linux only)
docker run --gpus all -it --rm llm-analysis
```

## Common Issues by OS

### Linux Issues

#### Permission Denied
```bash
# Fix Python installation permissions
sudo chown -R $USER:$USER ~/.local
```

#### CUDA Library Issues
```bash
# Add to ~/.bashrc
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

### Windows Issues

#### Long Path Names
Enable long paths in Windows:
1. Open Group Policy Editor (`gpedit.msc`)
2. Navigate to: Computer Configuration → Administrative Templates → System → Filesystem
3. Enable "Enable Win32 long paths"

#### Antivirus Interference
Add exclusions for:
- Python installation directory
- Virtual environment directory
- Project directory

### macOS Issues

#### SSL Certificate Errors
```bash
# Update certificates
/Applications/Python\ 3.10/Install\ Certificates.command
```

#### Permission Issues with Homebrew
```bash
# Fix Homebrew permissions
sudo chown -R $(whoami) $(brew --prefix)/*
```

## Performance Optimization by OS

### Linux
- Use `taskset` for CPU affinity
- Configure GPU power management
- Use `numactl` for NUMA systems

### Windows
- Set high performance power plan
- Disable Windows Defender real-time scanning for project directory
- Use Windows Terminal for better console experience

### macOS
- Use Activity Monitor to check resource usage
- Consider thermal throttling on MacBooks
- Enable "Prevent computer from sleeping" during long experiments

## Next Steps

After completing OS-specific setup:

1. **Verify Installation**: Run `python scripts/test_generation.py`
2. **Configure Model**: Edit `config.yaml` for your hardware
3. **Choose Model**: See [model selection guide](model-candidates.md)
4. **Run Experiments**: Follow experiment-specific instructions