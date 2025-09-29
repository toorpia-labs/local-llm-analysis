# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research platform for analyzing hidden layer states in local Large Language Models through experimental approaches. The repository follows a modular experiment-based architecture where each experiment is self-contained and independent.

## Repository Architecture

- **Root Level**: Contains general documentation and shared resources
- **experiments/**: Self-contained experimental projects, each with independent dependencies and execution environments
- **docs/**: General documentation and GitHub Pages content

Each experiment operates independently with its own:
- README with specific setup and execution instructions
- requirements.txt with isolated dependencies
- config.yaml for experiment configuration
- Complete source code in src/ directory
- Scripts for setup, testing, and execution

## Current Experiments

### Color Generation MCP Experiment (`experiments/color_generation/`)

Analyzes hidden layer states during Model Control Protocol (MCP) tool calling for color generation tasks.

**Architecture**:
- `src/model_loaders/`: HuggingFace Transformers model loading and hidden state extraction
- `src/mcp_controllers/`: MCP task execution controllers
- `src/analyzers/`: Hidden state analysis components
- `src/tools/`: External tools (rgb.py for color generation)
- `scripts/`: Setup, testing, and experiment execution scripts

## Common Development Commands

### Setting up a new experiment
```bash
cd experiments/your_experiment_name/
pip install -r requirements.txt
```

### Testing experiment functionality
```bash
cd experiments/color_generation/
python scripts/test_generation.py
```

### Running experiments with custom models
```bash
python scripts/test_generation.py --model microsoft/DialoGPT-medium
```

### Testing RGB tool directly
```bash
python experiments/color_generation/src/tools/rgb.py 255 0 0
```

### Full experiment execution
```bash
cd experiments/color_generation/
python scripts/run_experiment.py --config config.yaml --output results/
```

## Key Configuration Files

- `experiments/*/config.yaml`: Contains model settings, generation parameters, analysis configuration, and task definitions
- `experiments/*/requirements.txt`: Python dependencies for each experiment
- Key config sections include model backend, generation parameters, hidden state extraction settings, and MCP tool configuration

## Development Guidelines

When working with this codebase:

1. **Experiment Independence**: Each experiment is completely self-contained. Navigate to the specific experiment directory before running any commands.

2. **Model Configuration**: Experiments use HuggingFace Transformers models configured via YAML. Check config.yaml for current model settings before running experiments.

3. **Hidden State Analysis**: The core research focus is extracting and analyzing hidden layer states during specific token generation (particularly tool calls).

4. **MCP Integration**: Experiments use Model Control Protocol for structured tool calling. The rgb.py tool serves as a reference implementation.

5. **Results Storage**: Experimental outputs are saved to results/ directories within each experiment, including raw results, statistical analysis, and visualizations.

## Testing Strategy

Always run the test script before executing full experiments:
```bash
python scripts/test_generation.py
```

This validates:
- Tool execution (rgb.py functionality)
- Model loading and basic generation
- MCP integration and task execution