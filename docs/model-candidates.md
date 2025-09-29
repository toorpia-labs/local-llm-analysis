# Local LLM Model Candidates

This document lists candidate models for the color generation MCP experiment, organized by category and characteristics relevant to tool calling and instruction following.

## Table of Contents

- [Local LLM Model Candidates](#local-llm-model-candidates)
  - [Table of Contents](#table-of-contents)
  - [1. Selection Criteria](#1-selection-criteria)
  - [2. Latest High-Priority Models (2024-2025)](#2-latest-high-priority-models-2024-2025)
    - [2.1 Llama 3.3 Family (Dec 2024)](#21-llama-33-family-dec-2024)
    - [2.2 Qwen2.5 Family (Sep 2024, Jan 2025 tech report) ⭐ **HIGHLY RECOMMENDED**](#22-qwen25-family-sep-2024-jan-2025-tech-report--highly-recommended)
    - [2.3 Phi-4 Family (Jan-Feb 2025) ⭐ **COMPACT CHOICE**](#23-phi-4-family-jan-feb-2025--compact-choice)
  - [3. Llama 3.2 Family (Sep 2024)](#3-llama-32-family-sep-2024)
  - [4. Alternative Models](#4-alternative-models)
    - [4.1 Mistral Family](#41-mistral-family)
  - [5. Japanese Language Models](#5-japanese-language-models)
    - [5.1 Recommended (2025)](#51-recommended-2025)
    - [5.2 Japanese-Specific Models](#52-japanese-specific-models)
  - [6. Future Multimodal Experiments](#6-future-multimodal-experiments)
    - [6.1 Vision-Language Models](#61-vision-language-models)
  - [7. Lightweight Models](#7-lightweight-models)
    - [7.1 Qwen2.5 Small Models](#71-qwen25-small-models)
  - [8. Additional Research Options](#8-additional-research-options)
    - [8.1 Alternative Families](#81-alternative-families)
  - [9. 🎯 Recommendations by Hardware (Updated 2025)](#9--recommendations-by-hardware-updated-2025)
    - [9.1 Budget Setup (8GB GPU) ⭐ **BEST VALUE**](#91-budget-setup-8gb-gpu--best-value)
    - [9.2 Mid-Range Setup (16GB GPU) ⭐ **RECOMMENDED**](#92-mid-range-setup-16gb-gpu--recommended)
    - [9.3 High-End Setup (24GB+ GPU) ⭐ **MAXIMUM PERFORMANCE**](#93-high-end-setup-24gb-gpu--maximum-performance)
  - [10. 🏆 Top 3 Models for Color Generation Experiment](#10--top-3-models-for-color-generation-experiment)
    - [10.1 1st Place: **Phi-4-mini (3.8B)** 🥇](#101-1st-place-phi-4-mini-38b-)
    - [10.2 2nd Place: **Qwen2.5-7B-Instruct** 🥈](#102-2nd-place-qwen25-7b-instruct-)
    - [10.3 3rd Place: **Llama-3.3-70B-Instruct** 🥉](#103-3rd-place-llama-33-70b-instruct-)
  - [11. 📋 Research Tasks for Students](#11--research-tasks-for-students)
    - [11.1 Priority 1: Function Calling Evaluation](#111-priority-1-function-calling-evaluation)
    - [11.2 Priority 2: Performance Assessment](#112-priority-2-performance-assessment)
    - [11.3 Priority 3: Comparative Analysis](#113-priority-3-comparative-analysis)
    - [11.4 Methodology](#114-methodology)
  - [12. 🚀 Next Steps](#12--next-steps)
    - [12.1 Immediate Actions](#121-immediate-actions)
    - [12.2 Success Metrics](#122-success-metrics)
  - [13. 📝 Important Notes](#13--important-notes)
    - [13.1 Technical Considerations](#131-technical-considerations)
    - [13.2 Access Requirements](#132-access-requirements)
    - [13.3 Updated Focus (2025)](#133-updated-focus-2025)
    - [13.4 Performance Expectation](#134-performance-expectation)

## 1. Selection Criteria

For the color generation experiment, models should ideally have:
- **Function calling capability**: Native support for tool/function calling
- **Structured output generation**: Ability to generate formatted tool calls
- **Instruction following capability**: Can follow structured prompts reliably
- **Local execution feasibility**: Reasonable size for consumer hardware
- **Recent development**: Models released in 2024-2025 with active maintenance

## 2. Latest High-Priority Models (2024-2025)

These are the most recent and capable models with strong tool calling abilities:

### 2.1 Llama 3.3 Family (Dec 2024)
- **meta-llama/Llama-3.3-70B-Instruct** (70B parameters) ⭐ **TOP CHOICE**
  - Size: ~140GB
  - 405B-level performance in 70B size
  - **Native function calling support**
  - Multilingual capability
  - VRAM requirement: ~24GB+ (with 4-bit quantization)

### 2.2 Qwen2.5 Family (Sep 2024, Jan 2025 tech report) ⭐ **HIGHLY RECOMMENDED**
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

### 2.3 Phi-4 Family (Jan-Feb 2025) ⭐ **COMPACT CHOICE**
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

## 3. Llama 3.2 Family (Sep 2024)

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

## 4. Alternative Models

### 4.1 Mistral Family
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

## 5. Japanese Language Models

For potential Japanese language experiments:

### 5.1 Recommended (2025)
- **Qwen/Qwen2.5-7B-Instruct** - Strong multilingual including Japanese
- **Qwen/Qwen2.5-14B-Instruct** - Best multilingual performance

### 5.2 Japanese-Specific Models
- **elyza/ELYZA-japanese-Llama-2-7b-instruct** (7B parameters)
  - Size: ~13GB, VRAM: ~8GB+
- **stabilityai/japanese-stablelm-instruct-alpha-7b** (7B parameters)
  - Size: ~13GB, VRAM: ~8GB+

## 6. Future Multimodal Experiments

### 6.1 Vision-Language Models
- **Llama-3.2-11B-Vision-Instruct** - Primary choice for future multimodal experiments
- **Llama-3.2-90B-Vision-Instruct** - High-end multimodal (requires significant resources)
- **Qwen2.5-VL series** - Alternative multimodal options

## 7. Lightweight Models

For resource-constrained environments:

### 7.1 Qwen2.5 Small Models
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

## 8. Additional Research Options

### 8.1 Alternative Families
- **google/gemma-2-9b-it** (9B parameters) - Google's latest open model
- **01-ai/Yi-1.5-9B-Chat** (9B parameters) - Improved Yi generation
- **DeepSeek-V2.5** series - Research-focused models with strong reasoning

## 9. 🎯 Recommendations by Hardware (Updated 2025)

### 9.1 Budget Setup (8GB GPU) ⭐ **BEST VALUE**
1. **microsoft/Phi-4-mini** - Native function calling, exceptional performance
2. **Qwen/Qwen2.5-7B-Instruct** - Outstanding structured output capabilities
3. **Llama-3.2-3B-Instruct** - Reliable baseline model

### 9.2 Mid-Range Setup (16GB GPU) ⭐ **RECOMMENDED**
1. **Qwen/Qwen2.5-14B-Instruct** - Optimal balance for tool calling
2. **microsoft/Phi-4** (14B) - Mathematical reasoning + function calling
3. **Mistral-Nemo-Instruct-2407** - Long context capabilities

### 9.3 High-End Setup (24GB+ GPU) ⭐ **MAXIMUM PERFORMANCE**
1. **Llama-3.3-70B-Instruct** (4-bit) - Ultimate tool calling performance
2. **Qwen/Qwen2.5-32B-Instruct** - Top-tier structured output
3. **Qwen/Qwen2.5-Coder-32B-Instruct** - Code-specialized variant

## 10. 🏆 Top 3 Models for Color Generation Experiment

### 10.1 1st Place: **Phi-4-mini (3.8B)** 🥇
- **Why**: Native function calling, compact size, cost-effective
- **Best for**: Budget setups, rapid experimentation
- **VRAM**: ~4GB

### 10.2 2nd Place: **Qwen2.5-7B-Instruct** 🥈
- **Why**: Exceptional structured output, JSON generation expertise
- **Best for**: Mid-range setups, balanced performance
- **VRAM**: ~8GB

### 10.3 3rd Place: **Llama-3.3-70B-Instruct** 🥉
- **Why**: Maximum capability, 405B-level performance
- **Best for**: High-end setups, best possible results
- **VRAM**: ~24GB (4-bit quantization)

## 11. 📋 Research Tasks for Students

### 11.1 Priority 1: Function Calling Evaluation
- **Phi-4-mini**: Test native function calling with color generation prompts
- **Qwen2.5-7B**: Evaluate structured output quality and consistency
- **Llama-3.3-70B**: Benchmark maximum capability (if hardware allows)

### 11.2 Priority 2: Performance Assessment
- Measure tool calling accuracy across different prompt templates
- Compare generation speed and VRAM usage
- Test quantization impacts (4-bit, 8-bit) on tool calling quality
- Evaluate tokenizer compatibility with RGB tool format

### 11.3 Priority 3: Comparative Analysis
- Create standardized evaluation protocol for tool calling
- Document hardware requirements vs performance trade-offs
- Test model robustness with edge cases (invalid colors, malformed prompts)
- Analyze hidden state extraction overhead

### 11.4 Methodology
**Sequential Single-Student Evaluation Approach:**
1. **Establish baseline with Phi-4-mini** - Master the color generation experiment protocol
2. **Expand to Qwen2.5-7B** - Apply learned methodology to mid-range model
3. **Scale to Llama-3.3-70B** - Maximum capability benchmark (if hardware permits)
4. **Create integrated comparison report** - Single consistent evaluation across all models
5. **Document experimental learnings** - Refine methodology for future experiments

**Benefits of this approach:**
- **Consistent evaluation criteria** across all models
- **Learning curve advantage** - improved prompt engineering skills applied to each subsequent model
- **No aggregation complexity** - single comprehensive report
- **Efficient resource utilization** - focus on most promising candidates first

## 12. 🚀 Next Steps

### 12.1 Immediate Actions
**Phase 1: Establish Foundation with Most Promising Model**
1. **Master Phi-4-mini** - Focus entirely on the #1 ranked model first
   - Perfect the color generation experiment setup
   - Establish robust evaluation protocols
   - Document comprehensive results and methodology
2. **Create standardized workflow** - Reusable procedures for subsequent models
3. **Hardware optimization** - Fine-tune configuration for available resources
4. **Validate experimental design** - Ensure methodology captures meaningful differences

**Phase 2: Sequential Model Expansion (Only After Phase 1 Success)**
1. **Apply proven methodology to Qwen2.5-7B** - Leverage established workflow
2. **Scale to Llama-3.3-70B** - Maximum capability assessment (if hardware permits)
3. **Generate integrated comparison report** - Single comprehensive analysis

### 12.2 Success Metrics
**For Single-Student Integrated Comparison Report:**
- **Tool calling accuracy** - % successful RGB generations across all tested models
- **Consistency analysis** - Variation in performance across multiple prompt templates
- **Resource efficiency** - VRAM usage and generation speed comparison
- **Hidden state viability** - Feasibility of state extraction for each model
- **Robustness assessment** - Performance on edge cases and malformed inputs
- **Learning curve documentation** - How experimental skills improved across models
- **Methodology refinement** - Improvements discovered during sequential testing

**Report Quality Indicators:**
- **Single evaluation standard** applied consistently across all models
- **Comprehensive model ranking** with clear justification
- **Actionable recommendations** for future experiments
- **Reusable methodology** documented for other researchers

## 13. 📝 Important Notes

### 13.1 Technical Considerations
- Model sizes are approximate (float16 precision)
- VRAM requirements include overhead for hidden state extraction
- **Function calling models perform significantly better** for structured tasks
- Quantization (4-bit/8-bit) can reduce VRAM by 50-75% with minimal quality loss

### 13.2 Access Requirements
- Most models are freely available on HuggingFace Hub
- **Llama models require acceptance of license terms**
- Some models benefit from HuggingFace authentication for faster downloads

### 13.3 Updated Focus (2025)
- **Prioritize models with native function calling** (Phi-4, Llama-3.3)
- **Qwen2.5 series offers best structured output** for tool calling tasks
- **Avoid legacy models** (CodeT5, StarCoder v1, Llama-2, DialoGPT)
- **Consider multimodal capabilities** for future experiment expansion

### 13.4 Performance Expectation
Modern models (2024-2025) show **significantly better tool calling performance** compared to earlier generations, making them essential for MCP experiments.

---

*Last updated: January 2025*
*Focus: Function calling and structured output for MCP experiments*