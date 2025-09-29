# Local LLM Model Candidates

This document lists candidate models for the color generation MCP experiment, organized by category and characteristics relevant to tool calling and instruction following.

## Selection Criteria

For the color generation experiment, models should ideally have:
- **Function calling capability**: Native support for tool/function calling
- **Structured output generation**: Ability to generate formatted tool calls
- **Instruction following capability**: Can follow structured prompts reliably
- **Local execution feasibility**: Reasonable size for consumer hardware
- **Recent development**: Models released in 2024-2025 with active maintenance

## Latest High-Priority Models (2024-2025)

These are the most recent and capable models with strong tool calling abilities:

### Llama 3.3 Family (Dec 2024)
- **meta-llama/Llama-3.3-70B-Instruct** (70B parameters) ⭐ **TOP CHOICE**
  - Size: ~140GB
  - 405B-level performance in 70B size
  - **Native function calling support**
  - Multilingual capability
  - VRAM requirement: ~24GB+ (with 4-bit quantization)

### Qwen2.5 Family (Sep 2024, Jan 2025 tech report) ⭐ **HIGHLY RECOMMENDED**
- **Qwen/Qwen2.5-7B-Instruct** (7B parameters)
  - Size: ~14GB
  - **Excellent structured output generation**
  - Strong JSON/tool calling performance
  - VRAM requirement: ~8GB+

- **Qwen/Qwen2.5-14B-Instruct** (14B parameters)
  - Size: ~28GB
  - Enhanced reasoning and tool calling
  - VRAM requirement: ~16GB+

- **Qwen/Qwen2.5-32B-Instruct** (32B parameters)
  - Size: ~64GB
  - Top-tier performance for mid-range models
  - VRAM requirement: ~20GB+

- **Qwen/Qwen2.5-Coder-7B-Instruct** (7B parameters)
  - Size: ~14GB
  - Code-specialized with tool calling
  - VRAM requirement: ~8GB+

### Phi-4 Family (Jan-Feb 2025) ⭐ **COMPACT CHOICE**
- **microsoft/Phi-4-mini** (3.8B parameters) ⭐ **BUDGET TOP CHOICE**
  - Size: ~7GB
  - **Native function calling support**
  - Exceptional performance for size
  - VRAM requirement: ~4GB+

- **microsoft/Phi-4** (14B parameters)
  - Size: ~28GB
  - Mathematical reasoning specialized
  - Strong structured output
  - VRAM requirement: ~16GB+

## Llama 3.2 Family (Sep 2024)

Multimodal and edge-optimized models:

- **meta-llama/Llama-3.2-1B-Instruct** (1B parameters)
  - Size: ~2GB
  - Ultra-lightweight edge model
  - Good for parallel testing
  - VRAM requirement: ~2GB

- **meta-llama/Llama-3.2-3B-Instruct** (3B parameters)
  - Size: ~6GB
  - Balanced lightweight model
  - VRAM requirement: ~4GB+

- **meta-llama/Llama-3.2-11B-Vision-Instruct** (11B parameters)
  - Size: ~22GB
  - **Multimodal (text + vision)**
  - Future experiments potential
  - VRAM requirement: ~12GB+

## Alternative Models

### Mistral Family
- **mistralai/Mistral-7B-Instruct-v0.3** (7B parameters)
  - Size: ~14GB
  - Stable baseline model
  - Good instruction following
  - VRAM requirement: ~8GB+

- **mistralai/Mistral-Nemo-Instruct-2407** (12B parameters)
  - Size: ~24GB
  - 128K context window
  - Strong reasoning capabilities
  - VRAM requirement: ~14GB+

## Japanese Language Models

For potential Japanese language experiments:

### Recommended (2025)
- **Qwen/Qwen2.5-7B-Instruct** - Strong multilingual including Japanese
- **Qwen/Qwen2.5-14B-Instruct** - Best multilingual performance

### Japanese-Specific Models
- **elyza/ELYZA-japanese-Llama-2-7b-instruct** (7B parameters)
  - Size: ~13GB, VRAM: ~8GB+
- **stabilityai/japanese-stablelm-instruct-alpha-7b** (7B parameters)
  - Size: ~13GB, VRAM: ~8GB+

## Future Multimodal Experiments

### Vision-Language Models
- **Llama-3.2-11B-Vision-Instruct** - Primary choice for future multimodal experiments
- **Llama-3.2-90B-Vision-Instruct** - High-end multimodal (requires significant resources)
- **Qwen2.5-VL series** - Alternative multimodal options

## Lightweight Models

For resource-constrained environments:

### Qwen2.5 Small Models
- **Qwen/Qwen2.5-0.5B-Instruct** (0.5B parameters)
  - Size: ~1GB
  - Ultra-lightweight
  - VRAM requirement: ~1GB

- **Qwen/Qwen2.5-1.5B-Instruct** (1.5B parameters)
  - Size: ~3GB
  - Good small model performance
  - VRAM requirement: ~2GB

- **Qwen/Qwen2.5-3B-Instruct** (3B parameters)
  - Size: ~6GB
  - Balanced performance/size
  - VRAM requirement: ~4GB+

## Additional Research Options

### Alternative Families
- **google/gemma-2-9b-it** (9B parameters) - Google's latest open model
- **01-ai/Yi-1.5-9B-Chat** (9B parameters) - Improved Yi generation
- **DeepSeek-V2.5** series - Research-focused models with strong reasoning

## 🎯 Recommendations by Hardware (Updated 2025)

### Budget Setup (8GB GPU) ⭐ **BEST VALUE**
1. **microsoft/Phi-4-mini** - Native function calling, exceptional performance
2. **Qwen/Qwen2.5-7B-Instruct** - Outstanding structured output capabilities
3. **Llama-3.2-3B-Instruct** - Reliable baseline model

### Mid-Range Setup (16GB GPU) ⭐ **RECOMMENDED**
1. **Qwen/Qwen2.5-14B-Instruct** - Optimal balance for tool calling
2. **microsoft/Phi-4** (14B) - Mathematical reasoning + function calling
3. **Mistral-Nemo-Instruct-2407** - Long context capabilities

### High-End Setup (24GB+ GPU) ⭐ **MAXIMUM PERFORMANCE**
1. **Llama-3.3-70B-Instruct** (4-bit) - Ultimate tool calling performance
2. **Qwen/Qwen2.5-32B-Instruct** - Top-tier structured output
3. **Qwen/Qwen2.5-Coder-32B-Instruct** - Code-specialized variant

## 🏆 Top 3 Models for Color Generation Experiment

### 1st Place: **Phi-4-mini (3.8B)** 🥇
- **Why**: Native function calling, compact size, cost-effective
- **Best for**: Budget setups, rapid experimentation
- **VRAM**: ~4GB

### 2nd Place: **Qwen2.5-7B-Instruct** 🥈
- **Why**: Exceptional structured output, JSON generation expertise
- **Best for**: Mid-range setups, balanced performance
- **VRAM**: ~8GB

### 3rd Place: **Llama-3.3-70B-Instruct** 🥉
- **Why**: Maximum capability, 405B-level performance
- **Best for**: High-end setups, best possible results
- **VRAM**: ~24GB (4-bit quantization)

## 📋 Research Tasks for Students

### Priority 1: Function Calling Evaluation
- **Phi-4-mini**: Test native function calling with color generation prompts
- **Qwen2.5-7B**: Evaluate structured output quality and consistency
- **Llama-3.3-70B**: Benchmark maximum capability (if hardware allows)

### Priority 2: Performance Assessment
- Measure tool calling accuracy across different prompt templates
- Compare generation speed and VRAM usage
- Test quantization impacts (4-bit, 8-bit) on tool calling quality
- Evaluate tokenizer compatibility with RGB tool format

### Priority 3: Comparative Analysis
- Create standardized evaluation protocol for tool calling
- Document hardware requirements vs performance trade-offs
- Test model robustness with edge cases (invalid colors, malformed prompts)
- Analyze hidden state extraction overhead

### Methodology
1. **Start with Phi-4-mini** - Most promising budget option
2. **Scale up to Qwen2.5-7B** - Mid-range reference
3. **Test Llama-3.3-70B** - Maximum capability benchmark (if feasible)
4. **Document all findings** - Create model comparison matrix

## 🚀 Next Steps

### Immediate Actions
1. **Start with Phi-4-mini** - Download and test basic function calling
2. **Setup evaluation environment** - Standardized prompts and success metrics
3. **Document hardware specs** - Available VRAM and processing capabilities
4. **Create baseline tests** - Simple color generation tasks

### Team Assignments
- **Student 1**: Phi-4 series evaluation (mini + standard)
- **Student 2**: Qwen2.5 series comparison (3B, 7B, 14B)
- **Student 3**: Llama-3.2/3.3 assessment (based on available hardware)
- **Student 4**: Performance optimization (quantization, memory usage)

### Success Metrics
- Tool calling accuracy rate (% successful RGB generations)
- Response time and resource usage
- Hidden state extraction feasibility
- Robustness to prompt variations

## 📝 Important Notes

### Technical Considerations
- Model sizes are approximate (float16 precision)
- VRAM requirements include overhead for hidden state extraction
- **Function calling models perform significantly better** for structured tasks
- Quantization (4-bit/8-bit) can reduce VRAM by 50-75% with minimal quality loss

### Access Requirements
- Most models are freely available on HuggingFace Hub
- **Llama models require acceptance of license terms**
- Some models benefit from HuggingFace authentication for faster downloads

### Updated Focus (2025)
- **Prioritize models with native function calling** (Phi-4, Llama-3.3)
- **Qwen2.5 series offers best structured output** for tool calling tasks
- **Avoid legacy models** (CodeT5, StarCoder v1, Llama-2, DialoGPT)
- **Consider multimodal capabilities** for future experiment expansion

### Performance Expectation
Modern models (2024-2025) show **significantly better tool calling performance** compared to earlier generations, making them essential for MCP experiments.

---

*Last updated: January 2025*
*Focus: Function calling and structured output for MCP experiments*