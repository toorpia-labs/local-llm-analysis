#!/usr/bin/env python3
"""
Test script for color generation experiment.
"""

import sys
import os
import yaml
import argparse
import logging
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




async def main():
    parser = argparse.ArgumentParser(description='Test MCP color generation')
    parser.add_argument('--config', default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--prompt', default='Generate red color',
                       help='Test prompt')

    args = parser.parse_args()

    # Change to experiment directory
    os.chdir(experiment_root)

    # Load configuration
    config = load_config(args.config)
    setup_logging(config['output'].get('log_level', 'INFO'))
    logger = logging.getLogger(__name__)

    logger.info("MCP Color Generation Test")
    logger.info(f"Model: {config['model']['model_name']}")
    logger.info(f"Prompt: '{args.prompt}'")
    logger.info("=" * 60)

    try:
        # Load model
        loader = TransformersLoader(config)
        loader.load_model()

        # Initialize MCP controller
        controller = MCPController(loader, config)

        # Execute MCP task
        result = await controller.execute_mcp_task(args.prompt, 'rgb')

        # Display results
        logger.info("FULL PROMPT SENT TO LLM:")
        logger.info(result['full_prompt'])
        logger.info("=" * 60)
        logger.info("LLM RAW OUTPUT:")
        logger.info(f"'{result['generated_text']}'")
        logger.info("=" * 60)
        logger.info(f"Tool call detected: {result['tool_call_detected']}")
        if result['tool_call_detected']:
            logger.info(f"Detected RGB: {result['tool_args']}")
            logger.info(f"Tool success: {result['success']}")
            if result['success']:
                logger.info(f"Tool result: {result['tool_result']}")

        # Cleanup
        loader.cleanup()

        return 0

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1


if __name__ == '__main__':
    import asyncio
    sys.exit(asyncio.run(main()))