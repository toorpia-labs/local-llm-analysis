#!/usr/bin/env python3
"""
Test CSV export functionality with a small number of trials.
"""

import sys
import os
import yaml
import asyncio
import logging
import csv
from datetime import datetime
from pathlib import Path

# Add src to path
experiment_root = Path(__file__).parent.parent
sys.path.insert(0, str(experiment_root / 'src'))

from model_loaders.transformers_loader import TransformersLoader
from mcp_controllers.base import MCPController
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


async def test_csv_export():
    """Test CSV export with 5 trials."""
    # Change to experiment directory
    os.chdir(experiment_root)

    # Load configuration
    config = load_config('config.yaml')
    setup_logging(config['output'].get('log_level', 'INFO'))
    logger = logging.getLogger(__name__)

    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # CSV directory for test
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_dir = results_dir / f'test_csv_{timestamp}'

    logger.info("=" * 60)
    logger.info("CSV EXPORT TEST (20 trials)")
    logger.info(f"Model: {config['model']['model_name']}")
    logger.info(f"CSV output: {csv_dir}")
    logger.info("=" * 60)

    try:
        # Load model
        logger.info("Loading model...")

        # Increase temperature for testing variation
        original_temp = config['generation']['temperature']
        config['generation']['temperature'] = 1.5
        logger.info(f"Temperature increased: {original_temp} -> 1.5 (for testing)")

        loader = TransformersLoader(config)
        loader.load_model()

        # Get model info
        model_info = loader.get_model_info()
        hidden_size = model_info['hidden_size']
        logger.info(f"Hidden size: {hidden_size}")

        # Initialize MCP controller
        controller = MCPController(loader, config)

        # Initialize CSV exporter
        csv_export_layers = config.get('analysis', {}).get('csv_export_layers', [-1])
        csv_n_tokens = config.get('analysis', {}).get('csv_average_first_n_tokens', 1)

        logger.info(f"CSV export layers: {csv_export_layers}")
        logger.info(f"Averaging first {csv_n_tokens} tokens")

        csv_exporter = CSVExporter(
            output_dir=csv_dir,
            hidden_size=hidden_size,
            export_layers=csv_export_layers,
            n_tokens_to_average=csv_n_tokens
        )

        # Run 20 test trials
        prompt = "Generate red color"
        logger.info(f"\nRunning 20 test trials with prompt: '{prompt}'")

        for trial in range(1, 21):
            logger.info(f"\nTrial {trial}/20")

            # Execute MCP task
            result = await controller.execute_mcp_task(prompt, 'rgb')

            logger.info(f"  Generated: '{result['generated_text'][:50]}...'")
            logger.info(f"  Success: {result['success']}")

            # Export to CSV
            hidden_states = result.get('hidden_states', {})
            csv_written = csv_exporter.write_trial(
                trial_no=trial,
                success=result['success'],
                generated_text=result['generated_text'],
                hidden_states=hidden_states
            )

            logger.info(f"  CSV exported: {csv_written}")

        # Close CSV files
        csv_exporter.close()

        # Validate CSV files
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATING CSV FILES")

        for layer_idx in csv_export_layers:
            csv_file = csv_dir / f"layer_{layer_idx}.csv"

            if not csv_file.exists():
                logger.error(f"✗ CSV file not found: {csv_file}")
                continue

            # Read and validate CSV
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

            logger.info(f"\n✓ Layer {layer_idx} CSV: {csv_file}")
            logger.info(f"  Total rows: {len(rows)} (1 header + {len(rows)-1} data)")

            # Check header
            header = rows[0]
            logger.info(f"  Columns: {len(header)}")
            logger.info(f"  Header format: {header[:3]}...{header[-3:]}")

            expected_columns = 2 + hidden_size  # No, success + c1...cN
            if len(header) == expected_columns:
                logger.info(f"  ✓ Header column count correct: {expected_columns}")
            else:
                logger.error(f"  ✗ Header column count mismatch: expected {expected_columns}, got {len(header)}")

            # Analyze data rows
            if len(rows) > 1:
                # Count success vs failure
                success_rows = [r for r in rows[1:] if r[1] == 'True']
                failure_rows = [r for r in rows[1:] if r[1] == 'False']

                logger.info(f"\n  Data analysis:")
                logger.info(f"    Success trials: {len(success_rows)}")
                logger.info(f"    Failure trials: {len(failure_rows)}")

                # Check if success rows have identical hidden states
                if len(success_rows) > 1:
                    first_success = success_rows[0][2:7]  # First 5 values
                    all_same = all(r[2:7] == first_success for r in success_rows)
                    logger.info(f"    All success rows identical: {all_same}")
                    if not all_same:
                        logger.info(f"      Success row 1 values: {success_rows[0][2:7]}")
                        logger.info(f"      Success row 2 values: {success_rows[1][2:7]}")

                # Check if failure rows have different hidden states
                if len(failure_rows) > 1:
                    different = len(set(tuple(r[2:7]) for r in failure_rows)) > 1
                    logger.info(f"    Failure rows have variation: {different}")
                    for i, row in enumerate(failure_rows[:3], 1):
                        logger.info(f"      Failure {i} (trial {row[0]}): {row[2:7]}")
                elif len(failure_rows) == 1:
                    logger.info(f"    Single failure row (trial {failure_rows[0][0]}): {failure_rows[0][2:7]}")

                # Show first data row preview
                data_row = rows[1]
                logger.info(f"\n  First data row preview:")
                logger.info(f"    No: {data_row[0]}")
                logger.info(f"    Success: {data_row[1]}")
                logger.info(f"    First 5 hidden values: {data_row[2:7]}")
                logger.info(f"    Last 3 hidden values: {data_row[-3:]}")

                # Validate numeric values
                try:
                    hidden_values = [float(v) for v in data_row[2:]]
                    logger.info(f"  ✓ All {len(hidden_values)} hidden state values are numeric")
                except ValueError as e:
                    logger.error(f"  ✗ Non-numeric values found: {e}")

        # Cleanup
        loader.cleanup()

        logger.info("\n" + "=" * 60)
        logger.info("✓ CSV EXPORT TEST COMPLETED SUCCESSFULLY")
        logger.info(f"CSV files saved to: {csv_dir}")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


def main():
    """Main function."""
    success = asyncio.run(test_csv_export())
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
