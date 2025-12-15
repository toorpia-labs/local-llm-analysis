#!/usr/bin/env python3
"""
Test if hidden states are truly different across trials or being cached/reused.
"""

import sys
import os
import yaml
import asyncio
import logging
import hashlib
from pathlib import Path

# Add src to path
experiment_root = Path(__file__).parent.parent
sys.path.insert(0, str(experiment_root / 'src'))

from model_loaders.transformers_loader import TransformersLoader
from mcp_controllers.base import MCPController


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def hash_hidden_states(hidden_states):
    """Create a hash of hidden states to detect if they're identical."""
    if not hidden_states:
        return None

    # Extract first few vectors and create a hash
    data_str = str(hidden_states)
    return hashlib.md5(data_str.encode()).hexdigest()[:16]


async def test_uniqueness():
    """Test if hidden states are unique across trials."""
    os.chdir(experiment_root)

    config = yaml.safe_load(open('config.yaml'))
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("TESTING HIDDEN STATES UNIQUENESS")
    logger.info("=" * 60)

    # Load model
    logger.info("Loading model...")
    loader = TransformersLoader(config)
    loader.load_model()

    controller = MCPController(loader, config)

    # Run 5 trials and collect hidden states hashes
    prompt = "Generate red color"
    hashes = []
    first_5_values = []

    for trial in range(1, 6):
        logger.info(f"\nTrial {trial}/5")

        result = await controller.execute_mcp_task(prompt, 'rgb')
        hidden_states = result.get('hidden_states', {})

        # Get hash
        hs_hash = hash_hidden_states(hidden_states)
        hashes.append(hs_hash)

        # Get first step, first layer, first 5 values
        if 'step_0' in hidden_states and 'layer_-1' in hidden_states['step_0']:
            vec = hidden_states['step_0']['layer_-1'].get('vector', None)
            if vec is not None:
                first_5 = vec[0, -1, :5].tolist()
                first_5_values.append(first_5)
            else:
                first_5_values.append(None)
        else:
            first_5_values.append(None)

        logger.info(f"  Generated: '{result['generated_text'][:50]}...'")
        logger.info(f"  Success: {result['success']}")
        logger.info(f"  Hidden states hash: {hs_hash}")
        if first_5_values[-1]:
            logger.info(f"  First 5 values: {[f'{v:.4f}' for v in first_5_values[-1]]}")

    # Analysis
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS")
    logger.info("=" * 60)

    unique_hashes = set(hashes)
    logger.info(f"Total trials: {len(hashes)}")
    logger.info(f"Unique hashes: {len(unique_hashes)}")

    if len(unique_hashes) == 1:
        logger.error("✗ ALL HIDDEN STATES ARE IDENTICAL!")
        logger.error("  This suggests either:")
        logger.error("    1. Hidden states are being cached/reused")
        logger.error("    2. Model is deterministic (same input → same output → same states)")
        logger.error("    3. We're extracting the wrong data")
    elif len(unique_hashes) == len(hashes):
        logger.info("✓ All hidden states are unique!")
    else:
        logger.info(f"⚠ Some hidden states are duplicated:")
        for i, h in enumerate(hashes, 1):
            count = hashes.count(h)
            if count > 1:
                logger.info(f"  Trial {i} hash {h}: appears {count} times")

    # Check first 5 values
    logger.info("\nFirst 5 values comparison:")
    for i, vals in enumerate(first_5_values, 1):
        if vals:
            logger.info(f"  Trial {i}: {[f'{v:.4f}' for v in vals]}")

    if first_5_values[0] and all(v == first_5_values[0] for v in first_5_values if v):
        logger.error("✗ First 5 values are identical across all trials!")
    else:
        logger.info("✓ First 5 values show variation")

    # Cleanup
    loader.cleanup()

    return len(unique_hashes) > 1


def main():
    success = asyncio.run(test_uniqueness())
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
