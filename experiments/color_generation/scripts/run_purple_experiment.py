#!/usr/bin/env python3
"""
Run 100 MCP trials for purple color generation and save results to file.
"""

import sys
import os
import yaml
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
experiment_root = Path(__file__).parent.parent
sys.path.insert(0, str(experiment_root / 'src'))

from model_loaders.transformers_loader import TransformersLoader
from mcp_controllers.base import MCPController


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


async def run_purple_trials():
    """Run 100 independent MCP trials for purple color generation."""
    # Change to experiment directory
    os.chdir(experiment_root)

    # Load configuration
    config = load_config('config.yaml')
    setup_logging(config['output'].get('log_level', 'INFO'))
    logger = logging.getLogger(__name__)

    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f'results_purple_experiment_{timestamp}.dat'

    logger.info("=" * 60)
    logger.info("PURPLE COLOR MCP EXPERIMENT")
    logger.info(f"Model: {config['model']['model_name']}")
    logger.info(f"Task: Generate purple color (expected: rgb.py 128 0 128)")
    logger.info(f"Results file: {results_file}")
    logger.info("=" * 60)

    try:
        # Load model once
        logger.info("Loading model...")
        loader = TransformersLoader(config)
        loader.load_model()

        # Initialize MCP controller
        controller = MCPController(loader, config)
        logger.info("Model loaded and MCP controller initialized")

        # Results storage
        results = []
        prompt = "Generate purple color"

        # Run 100 trials
        logger.info("Starting 100 trials...")
        for trial in range(1, 101):
            logger.info(f"Trial {trial}/100")

            # Execute MCP task
            result = await controller.execute_mcp_task(prompt, 'rgb')

            # Store trial result
            trial_data = {
                'trial': trial,
                'timestamp': datetime.now().isoformat(),
                'prompt': result['prompt'],
                'generated_text': result['generated_text'],
                'tool_call_detected': result['tool_call_detected'],
                'tool_args': result['tool_args'],
                'success': result['success'],
                'tool_result': result['tool_result'],
                'error': result['error']
            }

            results.append(trial_data)

            # Progress update every 10 trials
            if trial % 10 == 0:
                successful = sum(1 for r in results if r['success'])
                success_rate = successful / trial * 100
                logger.info(f"Progress: {trial}/100 completed, Success rate: {success_rate:.1f}%")

        # Calculate final statistics
        successful_trials = [r for r in results if r['success']]
        failed_trials = [r for r in results if not r['success']]
        success_rate = len(successful_trials) / len(results) * 100

        # Analyze response variability
        unique_responses = set(r['generated_text'] for r in results)

        # Analyze RGB value accuracy for purple (128, 0, 128)
        correct_purple_count = 0
        incorrect_rgb_count = 0
        for r in results:
            if r['tool_args'] and r['tool_call_detected']:
                rgb_vals = (r['tool_args'].get('red', -1),
                           r['tool_args'].get('green', -1),
                           r['tool_args'].get('blue', -1))
                if rgb_vals == (128, 0, 128):
                    correct_purple_count += 1
                elif all(0 <= val <= 255 for val in rgb_vals):
                    incorrect_rgb_count += 1

        # Summary statistics
        summary = {
            'experiment_info': {
                'timestamp': timestamp,
                'model': config['model']['model_name'],
                'prompt': prompt,
                'expected_rgb': [128, 0, 128],
                'total_trials': len(results)
            },
            'results_summary': {
                'successful_trials': len(successful_trials),
                'failed_trials': len(failed_trials),
                'success_rate': success_rate,
                'correct_purple_rgb': correct_purple_count,
                'incorrect_rgb': incorrect_rgb_count,
                'rgb_accuracy': correct_purple_count / len(results) * 100,
                'unique_responses': len(unique_responses),
                'response_variability': len(unique_responses) > 1
            },
            'detailed_results': results
        }

        # Save results to file
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Display summary
        logger.info("=" * 60)
        logger.info("PURPLE COLOR EXPERIMENT COMPLETED")
        logger.info(f"Total trials: {len(results)}")
        logger.info(f"Successful: {len(successful_trials)} ({success_rate:.1f}%)")
        logger.info(f"Failed: {len(failed_trials)}")
        logger.info(f"Correct purple RGB (128,0,128): {correct_purple_count} ({correct_purple_count/len(results)*100:.1f}%)")
        logger.info(f"Incorrect RGB values: {incorrect_rgb_count}")
        logger.info(f"Unique responses: {len(unique_responses)}")
        logger.info(f"Response variability: {'Yes' if len(unique_responses) > 1 else 'No'}")
        logger.info(f"Results saved to: {results_file}")

        # Show unique responses
        if len(unique_responses) <= 10:
            logger.info("\nUnique LLM responses:")
            for i, response in enumerate(unique_responses, 1):
                logger.info(f"  {i}: '{response}'")

        # Show RGB value analysis
        if incorrect_rgb_count > 0:
            logger.info("\nIncorrect RGB values found:")
            incorrect_vals = set()
            for r in results:
                if r['tool_args'] and r['tool_call_detected']:
                    rgb_vals = (r['tool_args'].get('red', -1),
                               r['tool_args'].get('green', -1),
                               r['tool_args'].get('blue', -1))
                    if rgb_vals != (128, 0, 128) and all(0 <= val <= 255 for val in rgb_vals):
                        incorrect_vals.add(rgb_vals)

            for i, vals in enumerate(sorted(incorrect_vals), 1):
                logger.info(f"  {i}: rgb.py {vals[0]} {vals[1]} {vals[2]}")

        # Cleanup
        loader.cleanup()

        return summary

    except Exception as e:
        logger.error(f"Purple experiment failed: {e}")
        return None


def main():
    """Main function."""
    return asyncio.run(run_purple_trials())


if __name__ == '__main__':
    result = main()
    if result:
        print(f"\n✓ Purple color experiment completed successfully!")
        print(f"Success rate: {result['results_summary']['success_rate']:.1f}%")
        print(f"Purple RGB accuracy: {result['results_summary']['rgb_accuracy']:.1f}%")
        print(f"Response variability: {'Yes' if result['results_summary']['response_variability'] else 'No'}")
    else:
        print("\n✗ Purple experiment failed!")
        sys.exit(1)