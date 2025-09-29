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

### 1. Install Dependencies
```bash
cd experiments/color_generation
pip install -r requirements.txt
```

### 2. Configure Model
Edit `config.yaml` to specify your chosen model:
```yaml
model:
  model_name: "your-chosen-model"
  device: "cuda"  # or "cpu"
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