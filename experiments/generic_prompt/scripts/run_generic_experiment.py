#!/usr/bin/env python3
"""
Generic experiment script for repeated prompt execution with hidden state extraction.

This script allows running any prompt multiple times and extracting hidden states
to CSV files for analysis.

Usage:
    python run_generic_experiment.py --prompt "Your prompt here" --trials 100
    python run_generic_experiment.py --prompt "What is 2+2?" --trials 50 --output results/math_experiment
"""

import sys
import os
import yaml
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add src to path
experiment_root = Path(__file__).parent.parent
sys.path.insert(0, str(experiment_root / 'src'))

from model_loaders.transformers_loader import TransformersLoader
from utils.csv_exporter import CSVExporter


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_experiment(prompt: str, n_trials: int, output_dir: str = None, config_path: str = 'config.yaml', model_name: str = None):
    """
    Run generic experiment with repeated prompt execution.

    Args:
        prompt: The prompt to repeat
        n_trials: Number of trials to run
        output_dir: Optional output directory (default: results/generic_TIMESTAMP)
        config_path: Path to config file (default: config.yaml)
        model_name: Optional model name to override config (default: use config.yaml)
    """
    # Change to experiment directory
    os.chdir(experiment_root)

    # Load configuration
    config = load_config(config_path)

    # Override model name if specified
    if model_name:
        config['model']['model_name'] = model_name

    setup_logging(config['output'].get('log_level', 'INFO'))
    logger = logging.getLogger(__name__)

    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Output directory setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = results_dir / f'generic_{timestamp}'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Results file and CSV directory
    results_file = output_dir / 'results.json'
    csv_dir = output_dir / 'hidden_states'
    generations_file = output_dir / 'generations.txt'

    logger.info("=" * 60)
    logger.info("GENERIC PROMPT EXPERIMENT")
    logger.info(f"Model: {config['model']['model_name']}")
    logger.info(f"Prompt: '{prompt}'")
    logger.info(f"Trials: {n_trials}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)

    try:
        # Load model once
        logger.info("Loading model...")
        loader = TransformersLoader(config)
        loader.load_model()

        # Get model info
        model_info = loader.get_model_info()
        hidden_size = model_info['hidden_size']
        logger.info(f"Model loaded: {model_info['model_name']}")
        logger.info(f"Hidden size: {hidden_size}")

        # Initialize CSV exporter
        csv_export_layers = config.get('analysis', {}).get('csv_export_layers', [-1])
        csv_n_tokens = config.get('analysis', {}).get('csv_average_first_n_tokens', 1)
        csv_exporter = CSVExporter(
            output_dir=csv_dir,
            hidden_size=hidden_size,
            export_layers=csv_export_layers,
            n_tokens_to_average=csv_n_tokens
        )
        logger.info(f"CSV exporter initialized: layers={csv_export_layers}, n_tokens={csv_n_tokens}")

        # Results storage
        results = []
        generations = []

        # Run trials
        logger.info(f"\nStarting {n_trials} trials...")
        for trial in range(1, n_trials + 1):
            logger.info(f"Trial {trial}/{n_trials}")

            try:
                # Generate text with hidden states
                generated_text, hidden_states = loader.generate_with_states(prompt)

                logger.info(f"  Generated: '{generated_text[:80]}{'...' if len(generated_text) > 80 else ''}'")

                # Export hidden states to CSV
                csv_written = csv_exporter.write_trial(
                    trial_no=trial,
                    success=True,  # Always True for simple text generation
                    generated_text=generated_text,
                    hidden_states=hidden_states
                )

                # Store trial result (without hidden_states to save memory)
                trial_data = {
                    'trial': trial,
                    'timestamp': datetime.now().isoformat(),
                    'prompt': prompt,
                    'generated_text': generated_text,
                    'csv_exported': csv_written
                }

                results.append(trial_data)
                generations.append(f"=== Trial {trial} ===\n{generated_text}\n")

                if not csv_written:
                    logger.warning(f"  CSV export failed for trial {trial}")

            except Exception as e:
                logger.error(f"Trial {trial} failed: {e}", exc_info=True)
                # Store error
                error_data = {
                    'trial': trial,
                    'timestamp': datetime.now().isoformat(),
                    'prompt': prompt,
                    'error': str(e),
                    'csv_exported': False
                }
                results.append(error_data)

        # Close CSV exporter
        csv_exporter.close()

        # Calculate statistics
        successful_trials = [r for r in results if 'generated_text' in r]
        failed_trials = [r for r in results if 'error' in r]
        unique_responses = set(r['generated_text'] for r in successful_trials)

        # Save results summary
        summary = {
            'experiment': {
                'type': 'generic_prompt',
                'timestamp': timestamp,
                'prompt': prompt,
                'n_trials': n_trials
            },
            'model': model_info,
            'config': {
                'temperature': config['generation'].get('temperature', 0.7),
                'max_new_tokens': config['generation'].get('max_new_tokens', 50),
                'csv_export_layers': csv_export_layers,
                'csv_average_first_n_tokens': csv_n_tokens
            },
            'results': {
                'total_trials': n_trials,
                'successful_trials': len(successful_trials),
                'failed_trials': len(failed_trials),
                'unique_responses': len(unique_responses),
                'response_variability': len(unique_responses) > 1
            },
            'trials': results
        }

        # Save JSON results
        logger.info("\nSaving results...")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save generations text
        with open(generations_file, 'w', encoding='utf-8') as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Total trials: {n_trials}\n")
            f.write("=" * 80 + "\n\n")
            f.write("\n".join(generations))

        # Cleanup model
        loader.cleanup()

        # Display summary
        logger.info("=" * 60)
        logger.info("EXPERIMENT COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total trials: {n_trials}")
        logger.info(f"Successful: {len(successful_trials)}")
        logger.info(f"Failed: {len(failed_trials)}")
        logger.info(f"Unique responses: {len(unique_responses)}")
        logger.info(f"Response variability: {'Yes' if len(unique_responses) > 1 else 'No'}")
        logger.info(f"\nResults saved to: {output_dir}")
        logger.info(f"  - results.json: {results_file}")
        logger.info(f"  - generations.txt: {generations_file}")
        logger.info(f"  - Hidden states CSV: {csv_dir}")

        # Show sample unique responses (up to 5)
        if len(unique_responses) > 0:
            logger.info(f"\nSample responses ({min(5, len(unique_responses))} of {len(unique_responses)} unique):")
            for i, response in enumerate(list(unique_responses)[:5], 1):
                preview = response[:100] + "..." if len(response) > 100 else response
                logger.info(f"  {i}. {preview}")

        return True

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run generic prompt experiment with hidden state extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 100 trials with a simple prompt
  python run_generic_experiment.py --prompt "What is your name?" --trials 100

  # Run 50 trials with custom output directory
  python run_generic_experiment.py --prompt "Explain quantum computing" --trials 50 --output results/quantum_exp

  # Use custom config file
  python run_generic_experiment.py --prompt "Hello" --trials 10 --config my_config.yaml
        """
    )

    parser.add_argument(
        '--prompt',
        type=str,
        required=True,
        help='The prompt to repeat for all trials'
    )

    parser.add_argument(
        '--trials',
        type=int,
        default=100,
        help='Number of trials to run (default: 100)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: results/generic_TIMESTAMP)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name to use (overrides config.yaml). Example: microsoft/Phi-4-mini-instruct'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.trials < 1:
        parser.error("--trials must be at least 1")

    # Run experiment
    success = run_experiment(
        prompt=args.prompt,
        n_trials=args.trials,
        output_dir=args.output,
        config_path=args.config,
        model_name=args.model
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
