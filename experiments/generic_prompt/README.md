# Generic Prompt Experiment

## Objective

Execute any arbitrary prompt repeatedly and extract hidden states to CSV files for analysis. This experiment framework provides a flexible tool for analyzing LLM behavior across multiple generations of the same prompt.

## Research Applications

- **Response Variability Analysis**: Study how models respond differently to identical prompts
- **Hidden State Patterns**: Analyze internal representations across repeated generations
- **Prompt Engineering**: Test and compare different prompt formulations
- **Model Behavior**: Investigate consistency and randomness in model outputs

## Features

- Execute any custom prompt multiple times
- Extract hidden states using TransformersLoader
- Export data to CSV format for downstream analysis
- Simple command-line interface
- No MCP tool dependency (pure text generation)

## Setup Instructions

### Install Dependencies

```bash
cd experiments/generic_prompt
pip install -r requirements.txt
```

### Configuration

The experiment uses `config.yaml` for model and generation settings. Key configuration options:

```yaml
model:
  model_name: "microsoft/Phi-4-mini-instruct"
  device: "auto"
  torch_dtype: "float16"

generation:
  max_new_tokens: 50
  temperature: 1.5    # Higher temperature for more variability
  do_sample: true

analysis:
  csv_export_layers: [-1]  # Export final layer
  csv_average_first_n_tokens: 3  # Average first 3 tokens
```

## Usage

### Basic Usage

Run a simple experiment with default settings (100 trials):

```bash
python scripts/run_generic_experiment.py --prompt "What is your name?"
```

### Custom Number of Trials

```bash
python scripts/run_generic_experiment.py --prompt "Explain quantum computing" --trials 50
```

### Custom Output Directory

```bash
python scripts/run_generic_experiment.py \
  --prompt "What is 2+2?" \
  --trials 20 \
  --output results/math_experiment
```

### Using Custom Config

```bash
python scripts/run_generic_experiment.py \
  --prompt "Hello world" \
  --trials 10 \
  --config my_custom_config.yaml
```

## Command Line Arguments

**Required:**
- `--prompt`: The prompt text to repeat for all trials

**Optional:**
- `--trials`: Number of trials to run (default: 100)
- `--output`: Output directory path (default: `results/generic_TIMESTAMP`)
- `--config`: Path to configuration file (default: `config.yaml`)

## Output Format

Each experiment run creates a directory with the following structure:

```
results/generic_20251215_120000/
├── results.json           # Experiment metadata and summary
├── generations.txt        # All generated texts
└── hidden_states/         # CSV files with hidden state vectors
    └── layer_-1.csv      # Last layer hidden states
```

### results.json

Contains experiment metadata, model information, and results for each trial:

```json
{
  "experiment": {
    "type": "generic_prompt",
    "timestamp": "20251215_120000",
    "prompt": "What is your name?",
    "n_trials": 100
  },
  "model": {
    "model_name": "microsoft/Phi-4-mini-instruct",
    "hidden_size": 3072
  },
  "results": {
    "total_trials": 100,
    "successful_trials": 100,
    "unique_responses": 45,
    "response_variability": true
  },
  "trials": [...]
}
```

### generations.txt

Plain text file with all generated outputs:

```
Prompt: What is your name?
Total trials: 100
================================================================================

=== Trial 1 ===
I am Claude, an AI assistant created by Anthropic.

=== Trial 2 ===
My name is Claude. I'm an AI assistant.

...
```

### hidden_states/layer_-1.csv

CSV file with hidden state vectors. Each row represents one trial:

```
No,success,c1,c2,c3,...,c3072
1,True,0.123,-0.456,0.789,...,0.234
2,True,0.145,-0.423,0.801,...,0.221
...
```

Columns:
- `No`: Trial number
- `success`: Always True for this experiment type
- `c1...cN`: Hidden state dimensions (N = hidden_size)

## Example Experiments

### 1. Name Response Variability

```bash
python scripts/run_generic_experiment.py \
  --prompt "What is your name?" \
  --trials 100
```

**Research Question**: How consistently does the model identify itself?

### 2. Mathematical Reasoning

```bash
python scripts/run_generic_experiment.py \
  --prompt "What is 2+2?" \
  --trials 50
```

**Research Question**: Does the model give consistent answers to simple math?

### 3. Creative Generation

```bash
python scripts/run_generic_experiment.py \
  --prompt "Tell me a short story" \
  --trials 100
```

**Research Question**: How diverse are creative outputs?

## Integration with Analysis

The CSV output format is compatible with:
- Issue #3 Analyzer module (planned)
- Pandas/NumPy for custom analysis
- Scikit-learn for ML analysis
- Matplotlib/Seaborn for visualization

Example analysis workflow:

```python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Load hidden states
df = pd.read_csv('results/generic_20251215_120000/hidden_states/layer_-1.csv')

# Extract features
X = df.iloc[:, 2:].values  # Skip 'No' and 'success' columns

# Dimensionality reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Visualize
import matplotlib.pyplot as plt
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Hidden States PCA')
plt.show()
```

## Technical Details

### Code Reuse

This experiment reuses components from the color_generation experiment:
- `TransformersLoader`: Model loading and hidden state extraction
- `CSVExporter`: CSV file generation
- Shared via symlink: `src -> ../color_generation/src`

### Key Differences from Color Generation Experiment

| Feature | Color Generation | Generic Prompt |
|---------|-----------------|----------------|
| MCP Tool Calling | ✅ Required | ❌ Not used |
| Prompt | Fixed color tasks | Any custom prompt |
| Success Criteria | Tool call detection | Always successful |
| Use Case | Specific research | General purpose |

## Troubleshooting

### Model Loading Issues

If you encounter CUDA/device errors:
```bash
# Edit config.yaml
model:
  device: "cpu"  # Force CPU usage
```

### Out of Memory

Reduce batch size or use smaller model:
```yaml
generation:
  max_new_tokens: 20  # Reduce output length
```

### Import Errors

Ensure you're running from the experiment directory:
```bash
cd experiments/generic_prompt
python scripts/run_generic_experiment.py --prompt "test" --trials 1
```

## Future Enhancements

- Support for batch processing multiple prompts
- Real-time progress visualization
- Automatic comparison between different temperatures
- Integration with Issue #3 Analyzer for automatic analysis

## Related

- Issue #1: TransformersLoader implementation
- Issue #6: This experiment framework
- Issue #3: Analyzer module (planned)
- Color Generation Experiment: `../color_generation/`
