# System Requirements

This document outlines the hardware and software requirements for running the local LLM analysis experiments.

## Hardware Requirements

### Minimum Requirements
- **CPU**: Modern multi-core processor (Intel i5/AMD Ryzen 5 or equivalent)
- **RAM**: 16 GB system memory
- **Storage**: 50 GB free space for models and experiment data
- **GPU**: Optional but highly recommended for acceptable performance

### Recommended Requirements
- **CPU**: High-performance multi-core processor (Intel i7/i9 or AMD Ryzen 7/9)
- **RAM**: 32 GB or more system memory
- **Storage**: 100 GB+ free space (SSD preferred for faster model loading)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070/4060 Ti or better)

### GPU Requirements by Model Size

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

## Software Requirements

### Core Dependencies
- **Python**: 3.8 or higher (3.10+ recommended)
- **Operating System**:
  - Linux (Ubuntu 20.04+ recommended)
  - Windows 10/11
  - macOS 10.15+

### CUDA Requirements (for GPU acceleration)
- **CUDA Toolkit**: 11.8 or 12.1+
- **cuDNN**: Compatible version with CUDA
- **NVIDIA Driver**: Latest stable version (515+ for CUDA 11.8)

### Python Package Requirements
See `experiments/color_generation/requirements.txt` for specific versions:
- PyTorch 1.9.0+
- Transformers 4.21.0+
- Additional ML and analysis libraries

## Network Requirements
- **Internet connection** for:
  - Initial model downloads from HuggingFace Hub
  - Package installation via pip
- **Estimated bandwidth**: 5-50 GB for model downloads (depending on model size)

## Storage Considerations

### Model Storage
- **Local cache**: Models are cached locally after first download
- **Cache location**: `~/.cache/huggingface/` (can be configured)
- **Size estimates**:
  - 1B models: ~2-4 GB
  - 7B models: ~13-26 GB
  - 13B models: ~25-50 GB

### Experiment Data
- **Results storage**: Configurable via `config.yaml`
- **Hidden states**: Can be memory-intensive (several GB per experiment)
- **Logs and outputs**: Typically < 1 GB

## Performance Expectations

### CPU-only Performance
- **Model loading**: 1-5 minutes
- **Generation**: 10-100 tokens/second (varies by model size)
- **Hidden state extraction**: Significant overhead, 2-5x slower

### GPU Performance
- **Model loading**: 30 seconds - 2 minutes
- **Generation**: 50-500+ tokens/second
- **Hidden state extraction**: Minimal overhead with sufficient VRAM

## Troubleshooting Common Issues

### Out of Memory (OOM)
- Reduce model size or use quantization
- Lower batch size in config
- Use CPU offloading if available
- Close other applications consuming memory/VRAM

### Slow Performance
- Ensure GPU utilization (`nvidia-smi`)
- Check CUDA/PyTorch installation
- Consider model quantization
- Use faster storage (SSD)

### Installation Issues
- Verify Python and CUDA versions compatibility
- Use virtual environments to avoid conflicts
- Check firewall/proxy settings for downloads