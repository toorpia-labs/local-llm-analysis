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


def test_tool_execution():
    """Test RGB tool execution."""
    tool_path = experiment_root / 'src' / 'tools' / 'rgb.py'
    
    if not tool_path.exists():
        return False, f"Tool not found: {tool_path}"
    
    try:
        import subprocess
        result = subprocess.run(
            ['python', str(tool_path), '255', '0', '0'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            return True, "RGB tool working correctly"
        else:
            return False, f"Tool failed: {result.stderr}"
            
    except Exception as e:
        return False, f"Tool test error: {e}"


def test_model_loading(config: dict):
    """Test model loading and basic generation."""
    logger = logging.getLogger(__name__)
    
    try:
        # Load model
        loader = TransformersLoader(config)
        model = loader.load_model()
        
        model_info = loader.get_model_info()
        logger.info(f"Model loaded: {model_info['model_name']}")
        logger.info(f"Layers: {model_info['num_layers']}, Hidden size: {model_info['hidden_size']}")
        
        # Test generation
        test_prompt = "Generate red color"
        generated_text, hidden_states = loader.generate_with_states(test_prompt)
        
        logger.info(f"Test prompt: '{test_prompt}'")
        logger.info(f"Generated: '{generated_text}'")
        logger.info(f"Hidden states extracted: {list(hidden_states.keys())}")
        
        return True, {
            'model_info': model_info,
            'generated_text': generated_text,
            'hidden_states_count': len(hidden_states)
        }
        
    except Exception as e:
        return False, str(e)


def test_mcp_integration(config: dict):
    """Test MCP controller integration."""
    logger = logging.getLogger(__name__)
    
    try:
        # Load model
        loader = TransformersLoader(config)
        loader.load_model()
        
        # Initialize MCP controller
        controller = MCPController(loader, config)
        
        # Test simple task
        import asyncio
        
        async def run_test():
            result = await controller.execute_mcp_task(
                "Generate red color using the RGB tool.", 
                "rgb"
            )
            return result
        
        result = asyncio.run(run_test())
        
        logger.info(f"MCP Task completed")
        logger.info(f"Generated: '{result['generated_text']}'")
        logger.info(f"Tool call detected: {result['tool_call_detected']}")
        logger.info(f"Tool success: {result['success']}")
        
        return True, result
        
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Test color generation experiment')
    parser.add_argument('--config', default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--model', type=str,
                       help='Override model name')
    
    args = parser.parse_args()
    
    # Change to experiment directory
    os.chdir(experiment_root)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override model if specified
    if args.model:
        config['model']['model_name'] = args.model
    
    setup_logging(config['output'].get('log_level', 'INFO'))
    logger = logging.getLogger(__name__)
    
    logger.info("Starting color generation experiment tests")
    logger.info(f"Experiment root: {experiment_root}")
    logger.info(f"Model: {config['model']['model_name']}")
    
    # Test 1: Tool execution
    logger.info("=" * 50)
    logger.info("TEST 1: RGB Tool Execution")
    success, message = test_tool_execution()
    logger.info(f"Result: {'✓ PASS' if success else '✗ FAIL'} - {message}")
    
    if not success:
        logger.error("Tool test failed, stopping tests")
        return 1
    
    # Test 2: Model loading
    logger.info("=" * 50)
    logger.info("TEST 2: Model Loading and Generation")
    success, result = test_model_loading(config)
    logger.info(f"Result: {'✓ PASS' if success else '✗ FAIL'}")
    
    if not success:
        logger.error(f"Model test failed: {result}")
        return 1
    
    # Test 3: MCP integration
    logger.info("=" * 50)
    logger.info("TEST 3: MCP Integration")
    success, result = test_mcp_integration(config)
    logger.info(f"Result: {'✓ PASS' if success else '✗ FAIL'}")
    
    if not success:
        logger.error(f"MCP test failed: {result}")
        return 1
    
    # Summary
    logger.info("=" * 50)
    logger.info("ALL TESTS PASSED ✓")
    logger.info("Experiment is ready to run!")
    logger.info("Next: python scripts/run_experiment.py")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())