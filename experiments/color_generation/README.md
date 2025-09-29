# Color Generation via MCP Experiment

## Objective

Analyze hidden layer states in local LLMs when they perform tool-calling tasks through Model Control Protocol (MCP). Specifically, investigate how internal representations change when the model generates tool calls versus regular text.

## Research Questions

1. How do hidden states differ between tool-calling and regular text generation?
2. Which layers show the most significant changes during tool call generation?
3. Can we identify patterns that predict successful tool call generation?

## Experimental Design

### Task Description
The LLM receives prompts requesting color generation:
- Input: "Generate red color using the RGB tool"
- Expected behavior: Model generates "rgb.py 255 0 0" or similar tool call
- Analysis: Extract hidden states for output tokens corresponding to tool calls

### Variables
- **Models**: Various HuggingFace Transformers models (to be determined experimentally)
- **Colors**: red, blue, green, yellow, purple, orange, white, black
- **Prompt variations**: Multiple templates for requesting colors
- **Repetitions**: Multiple runs for statistical analysis

## Setup Instructions

### 1. Platform-Specific Setup

Choose the setup method based on your operating system and hardware:

#### 🐧 Linux/WSL2 Users (CUDA Recommended)

```bash
cd experiments/color_generation

# Install base dependencies
pip install -r requirements.txt

# Install CUDA-enabled PyTorch
pip install -r requirements-cuda.txt

# Copy CUDA-optimized configuration
cp config-examples/config-cuda.yaml config.yaml

# Verify GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### 🍎 macOS Apple Silicon Users (MPS Recommended)

```bash
cd experiments/color_generation

# Install base dependencies
pip install -r requirements.txt

# Install MPS-enabled PyTorch
pip install -r requirements-mps.txt

# Copy MPS-optimized configuration
cp config-examples/config-mps.yaml config.yaml

# Verify MPS support
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

#### 💻 CPU-Only Setup (All Platforms)

```bash
cd experiments/color_generation

# Install base dependencies only
pip install -r requirements.txt

# Copy CPU configuration
cp config-examples/config-cpu.yaml config.yaml

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

#### 🐍 Conda Users (Alternative)

```bash
cd experiments/color_generation

# Create conda environment
conda env create -f environment.yml
conda activate llm-analysis

# Then follow platform-specific PyTorch installation above
```

### 2. Configuration Customization

After copying the appropriate config template, you can customize:

```yaml
model:
  model_name: "your-chosen-model"  # See docs/model-candidates.md for options
  # device and torch_dtype are already optimized for your platform
```

### 3. Test Tools
```bash
python scripts/setup_tools.py
```

### 4. Run Initial Test
```bash
python scripts/test_generation.py
```

## Repository Structure

```
color_generation/
├── README.md              # This file
├── config.yaml           # Experiment configuration
├── requirements.txt       # Dependencies
├── src/
│   ├── model_loaders/     # Model loading and hidden state extraction
│   ├── mcp_controllers/   # MCP task execution
│   ├── analyzers/         # Hidden state analysis
│   └── tools/             # External tools (rgb.py, etc.)
├── scripts/
│   ├── setup_tools.py     # Tool setup and verification
│   ├── test_generation.py # Basic functionality test
│   └── run_experiment.py  # Full experiment execution
└── results/               # Experimental outputs (auto-created)
```

## Usage

### Quick Test
```bash
python scripts/test_generation.py --model microsoft/DialoGPT-medium
```

### Full Experiment
```bash
python scripts/run_experiment.py --config config.yaml --output results/
```

## Analysis Outputs

- **Raw Results**: JSON files with all generated text and hidden states
- **Statistical Analysis**: Success rates, detection rates, by color and model
- **Visualizations**: Hidden state patterns and comparisons
- **Model Comparisons**: If multiple models are tested

## Expected Outcomes

This experiment will establish baseline methodologies for:
- MCP-controlled tool calling with local LLMs
- Hidden state extraction during specific token generation
- Comparative analysis across models and tasks

Results will inform design of future, more complex experiments in this research platform.

## Team Collaboration Guidelines

### Environment Consistency

When working in a team with mixed operating systems, follow these guidelines:

#### Sharing Configuration Files
- **DO**: Share the base `config.yaml` with model and experiment settings
- **DON'T**: Share platform-specific device or torch_dtype settings
- **RECOMMENDED**: Document your platform in issue comments when reporting results

#### Reproducing Results
Different platforms may produce slightly different results due to:
- **Precision differences**: CUDA (float16) vs MPS (float32) vs CPU (float32)
- **Hardware variations**: Different GPU architectures or memory configurations
- **Library versions**: Platform-specific PyTorch builds

#### Reporting Issues
When creating GitHub issues, always include:
```
## Environment Info
- Platform: Linux/WSL2/macOS/Windows
- Hardware: GPU model or "CPU-only"
- Python: [version]
- PyTorch: [version]
- Config: [config-cuda.yaml/config-mps.yaml/config-cpu.yaml]
```

#### Best Practices
1. **Consistent base requirements**: All team members use the same `requirements.txt`
2. **Platform-appropriate config**: Use the config template for your platform
3. **Document deviations**: Note any custom changes in issue descriptions
4. **Version control**: Only commit platform-independent files
5. **Results comparison**: Account for expected platform differences in analysis

### Troubleshooting Common Team Issues

#### "My results don't match yours"
1. Verify you're using the correct config template for your platform
2. Check PyTorch installation with verification commands above
3. Compare model and generation settings (not device-specific ones)
4. Consider precision differences between platforms

#### "The experiment won't run"
1. Verify platform setup was completed correctly
2. Check that config.yaml exists (should be copied from config-examples/)
3. Ensure all requirements files were installed for your platform
4. Run verification commands to check PyTorch installation

#### "Performance is very slow"
- **Linux/WSL2**: Verify CUDA is working with `nvidia-smi`
- **macOS**: Verify MPS is enabled and available
- **All platforms**: Consider using CPU config for small tests