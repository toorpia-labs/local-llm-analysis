#!/usr/bin/env python3
"""
Test script for generic prompt experiment.

Tests the experiment framework with 3 different prompts to validate functionality.
"""

import sys
import os
from pathlib import Path
import subprocess
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test prompts
TEST_PROMPTS = [
    "What is your name?",
    "What is 2+2?",
    "Hello"
]

# Number of trials for testing (small number for quick validation)
TEST_TRIALS = 3


def run_test_experiment(prompt: str, trial_num: int):
    """
    Run a test experiment with the given prompt.

    Args:
        prompt: The prompt to test
        trial_num: Test number (for naming output directory)

    Returns:
        True if successful, False otherwise
    """
    experiment_root = Path(__file__).parent.parent
    script_path = experiment_root / 'scripts' / 'run_generic_experiment.py'
    output_dir = experiment_root / 'results' / f'test_{trial_num}'

    logger.info(f"\nTest {trial_num}/3: Testing prompt '{prompt}'")
    logger.info("=" * 60)

    try:
        # Run the experiment script
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                '--prompt', prompt,
                '--trials', str(TEST_TRIALS),
                '--output', str(output_dir)
            ],
            cwd=str(experiment_root),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            logger.info(f"✓ Test {trial_num} passed: '{prompt}'")
            logger.info(f"  Output directory: {output_dir}")

            # Verify output files exist
            results_file = output_dir / 'results.json'
            generations_file = output_dir / 'generations.txt'
            csv_dir = output_dir / 'hidden_states'

            if not results_file.exists():
                logger.error(f"  ✗ Missing results.json")
                return False

            if not generations_file.exists():
                logger.error(f"  ✗ Missing generations.txt")
                return False

            if not csv_dir.exists():
                logger.error(f"  ✗ Missing hidden_states directory")
                return False

            csv_files = list(csv_dir.glob('*.csv'))
            if len(csv_files) == 0:
                logger.error(f"  ✗ No CSV files generated")
                return False

            logger.info(f"  ✓ All output files present")
            logger.info(f"  ✓ {len(csv_files)} CSV file(s) generated")
            return True

        else:
            logger.error(f"✗ Test {trial_num} failed: '{prompt}'")
            logger.error(f"  Return code: {result.returncode}")
            if result.stderr:
                logger.error(f"  Error output:\n{result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"✗ Test {trial_num} timed out after 5 minutes")
        return False

    except Exception as e:
        logger.error(f"✗ Test {trial_num} failed with exception: {e}")
        return False


def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("GENERIC PROMPT EXPERIMENT - TEST SUITE")
    logger.info("=" * 60)
    logger.info(f"Testing with {TEST_TRIALS} trials per prompt")
    logger.info(f"Total prompts to test: {len(TEST_PROMPTS)}")

    results = []

    # Run tests for each prompt
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        success = run_test_experiment(prompt, i)
        results.append((prompt, success))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed

    for prompt, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: '{prompt}'")

    logger.info("")
    logger.info(f"Total: {len(results)} tests")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")

    if failed == 0:
        logger.info("\n✓ ALL TESTS PASSED")
        return 0
    else:
        logger.error(f"\n✗ {failed} TEST(S) FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
