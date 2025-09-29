# Local LLM Analysis

Research platform for analyzing hidden layer states in local LLMs through various experimental approaches.

## Overview

This repository serves as a research hub for investigating the internal representations of locally-running Language Models. Each experiment is self-contained within its own directory, allowing for diverse approaches and independent development.

## Research Philosophy

- **Experimental Diversity**: Each experiment can use different models, techniques, and analysis methods
- **Self-Contained Experiments**: Complete independence allows for easy replication and migration
- **Incremental Discovery**: Start with simple experiments, evolve to complex analyses
- **Flexible Architecture**: No forced framework - each experiment defines its own structure

## Repository Structure

- `docs/`: General documentation and GitHub Pages content
- `experiments/`: Self-contained experimental projects
  - Each experiment has its own README, dependencies, and complete codebase
  - Experiments can be independently executed and analyzed
  - Successful experiments can be easily migrated to separate repositories

## Getting Started

1. Clone the repository
2. Navigate to the experiment of interest (e.g., `experiments/color_generation/`)
3. Follow the experiment's specific README for setup and execution

## Contributing New Experiments

1. Create a new directory under `experiments/`
2. Implement your experiment as a self-contained project
3. Include comprehensive documentation in the experiment's README
4. Add any experiment-specific dependencies to its own requirements.txt

## Future Development

As experiments mature and grow in scope, they may be promoted to independent repositories while maintaining their experimental history here.

## License

MIT License - see LICENSE file for details