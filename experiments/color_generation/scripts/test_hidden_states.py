#!/usr/bin/env python3
"""
Test hidden states extraction functionality.
"""

import sys
import os
import yaml
import logging
import numpy as np
from pathlib import Path

# Add src to path
experiment_root = Path(__file__).parent.parent
sys.path.insert(0, str(experiment_root / 'src'))

from model_loaders.transformers_loader import TransformersLoader


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


def test_hidden_states_extraction():
    """Test hidden states extraction with a simple prompt."""
    # Change to experiment directory
    os.chdir(experiment_root)

    # Load configuration
    config = load_config('config.yaml')
    setup_logging(config['output'].get('log_level', 'INFO'))
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("HIDDEN STATES EXTRACTION TEST")
    logger.info("=" * 60)

    try:
        # Load model
        logger.info("Loading model...")
        loader = TransformersLoader(config)
        loader.load_model()

        model_info = loader.get_model_info()
        logger.info(f"Model: {model_info['model_name']}")
        logger.info(f"Layers: {model_info['num_layers']}")
        logger.info(f"Hidden size: {model_info['hidden_size']}")

        # Test prompt
        test_prompt = "Generate red color"
        logger.info(f"\nTest prompt: '{test_prompt}'")

        # Generate with hidden states
        logger.info("Generating with hidden states extraction...")
        generated_text, hidden_states = loader.generate_with_states(test_prompt)

        logger.info(f"Generated text: '{generated_text}'")
        logger.info(f"\nHidden states structure:")

        # Analyze hidden states
        if hidden_states:
            logger.info(f"Number of generation steps: {len(hidden_states)}")

            for step_key in list(hidden_states.keys())[:3]:  # Show first 3 steps
                step_data = hidden_states[step_key]
                logger.info(f"\n{step_key}:")

                for layer_key, layer_data in step_data.items():
                    logger.info(f"  {layer_key}:")
                    logger.info(f"    Shape: {layer_data['shape']}")
                    logger.info(f"    Mean: {layer_data['mean']:.4f}")
                    logger.info(f"    Std: {layer_data['std']:.4f}")

                    # Check if raw vectors are saved
                    if 'vector' in layer_data:
                        vector = layer_data['vector']
                        logger.info(f"    ✓ Raw vector saved! Shape: {vector.shape}, dtype: {vector.dtype}")
                        logger.info(f"    Vector size: {vector.nbytes / 1024:.2f} KB")
                    else:
                        logger.info(f"    ✗ Raw vector NOT saved (save_raw_states=false)")

            # Calculate total data size
            total_size = 0
            vector_count = 0
            for step_data in hidden_states.values():
                for layer_data in step_data.values():
                    if 'vector' in layer_data:
                        total_size += layer_data['vector'].nbytes
                        vector_count += 1

            if vector_count > 0:
                logger.info(f"\n📊 Total hidden states data:")
                logger.info(f"  Vectors saved: {vector_count}")
                logger.info(f"  Total size: {total_size / 1024:.2f} KB ({total_size / (1024**2):.2f} MB)")

        else:
            logger.info("No hidden states extracted!")

        # Test extract_hidden_states API
        logger.info("\n" + "=" * 60)
        logger.info("Testing extract_hidden_states(input_ids, attention_mask) API...")

        # Tokenize test input
        test_input = "Test input"
        inputs = loader.tokenizer(test_input, return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        logger.info(f"Test input: '{test_input}'")
        logger.info(f"Input IDs shape: {input_ids.shape}")

        # Extract hidden states
        extracted = loader.extract_hidden_states(input_ids, attention_mask)

        logger.info(f"Extracted hidden states:")
        for layer_key, layer_data in extracted.items():
            logger.info(f"  {layer_key}:")
            logger.info(f"    Shape: {layer_data['shape']}")
            logger.info(f"    Mean: {layer_data['mean']:.4f}")
            if 'vector' in layer_data:
                logger.info(f"    ✓ Raw vector saved: {layer_data['vector'].shape}")
            else:
                logger.info(f"    ✗ Raw vector NOT saved")

        # Cleanup
        loader.cleanup()

        logger.info("\n" + "=" * 60)
        logger.info("✓ HIDDEN STATES EXTRACTION TEST PASSED")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


def main():
    """Main function."""
    success = test_hidden_states_extraction()
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
