#!/usr/bin/env python3
"""
MCP Controller for tool calling detection and task execution.
"""

import asyncio
import json
import re
import subprocess
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


class MCPController:
    """Controller for MCP-based tool calling and task execution."""

    def __init__(self, model_loader, config: Dict[str, Any]):
        """
        Initialize MCPController.

        Args:
            model_loader: Model loader instance (TransformersLoader)
            config: Configuration dictionary
        """
        self.model_loader = model_loader
        self.config = config
        self.mcp_config = config.get('mcp', {})
        self.tools_directory = Path(self.mcp_config.get('tools_directory', 'src/tools'))
        self.timeout = self.mcp_config.get('timeout', 30)
        self.logger = logging.getLogger(__name__)
        self.system_prompt = self._build_mcp_system_prompt()

    def _build_mcp_system_prompt(self) -> str:
        """Build MCP system prompt with tool definitions."""
        return """You are a color generation assistant with access to tools.

AVAILABLE TOOL:
- Name: rgb.py
- Description: Generates RGB color output
- Usage: rgb.py <red> <green> <blue>
- Parameters: red, green, blue (integers 0-255)

CRITICAL INSTRUCTIONS:
When asked to generate or output a color:
1. Convert the color name to RGB values (0-255 range)
2. Output ONLY: rgb.py <red> <green> <blue>
3. NO explanations, NO additional text, NO conversation
4. STOP generating after the command
5. One line only, then STOP

FORMAT EXAMPLES:
"Output blue" → rgb.py 0 0 255
"Show me green" → rgb.py 0 255 0
"Generate yellow color" → rgb.py 255 255 0"""

    async def execute_mcp_task(self, prompt: str, tool_name: str) -> Dict[str, Any]:
        """
        Execute an MCP task with tool calling.

        Args:
            prompt: The task prompt for the model
            tool_name: Expected tool name (e.g., 'rgb')

        Returns:
            Dict containing execution results
        """
        try:
            # Build full prompt with system instructions
            full_prompt = f"{self.system_prompt}\n\nUser: {prompt}\nAssistant:"

            # Generate text with the model
            generated_text, hidden_states = self.model_loader.generate_with_states(full_prompt)

            # Detect and parse tool calls
            tool_call_detected, tool_args = self._detect_tool_call(generated_text, tool_name)

            result = {
                'prompt': prompt,
                'full_prompt': full_prompt,
                'generated_text': generated_text,
                'tool_call_detected': tool_call_detected,
                'tool_args': tool_args,
                'hidden_states': hidden_states,
                'success': False,
                'tool_result': None,
                'error': None
            }

            # Check if generated text is exactly the RGB command format
            if self._is_exact_rgb_command(generated_text):
                if tool_call_detected and tool_args:
                    # Execute the tool
                    tool_success, tool_result = await self._execute_tool(tool_name, tool_args)
                    result['success'] = tool_success
                    result['tool_result'] = tool_result
                    if not tool_success:
                        result['error'] = tool_result
                else:
                    result['error'] = 'Tool call detected but failed to parse arguments'
            else:
                result['success'] = False
                result['error'] = 'Output does not match exact RGB command format'

            return result

        except Exception as e:
            self.logger.error(f"Error executing MCP task: {e}")
            return {
                'prompt': prompt,
                'generated_text': '',
                'tool_call_detected': False,
                'tool_args': None,
                'hidden_states': {},
                'success': False,
                'tool_result': None,
                'error': str(e)
            }

    def _detect_tool_call(self, text: str, expected_tool: str) -> Tuple[bool, Optional[Dict]]:
        """
        Detect tool calls in generated text.

        Args:
            text: Generated text to analyze
            expected_tool: Expected tool name

        Returns:
            Tuple of (detected, arguments)
        """
        # Multiple patterns for tool call detection - ordered by specificity
        patterns = [
            # MCP command format: rgb.py 255 0 0 (highest priority)
            r'rgb\.py\s+(\d+)\s+(\d+)\s+(\d+)',
            # RGB function pattern: rgb(255, 0, 0) or RGB(255, 0, 0)
            r'rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)',
            # RGB with parentheses and names: RGB(red_255, green_0, blue_0)
            r'RGB\s*\(\s*\w*[_\s]*(\d+)\s*,\s*\w*[_\s]*(\d+)\s*,\s*\w*[_\s]*(\d+)\s*\)',
            # RGB values with colon: RGB values: 255, 0, 0
            r'RGB\s+values?\s*:?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
            # Simple number sequence with commas: 255, 0, 0
            r'(?:^|\s|:)(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})(?:\s|$|\.)',
            # Three numbers separated by spaces: 255 0 0
            r'(?:^|\s)(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})(?:\s|$)',
            # JSON-like pattern: {"red": 255, "green": 0, "blue": 0}
            r'\{\s*["\']?red["\']?\s*:\s*(\d+)\s*,\s*["\']?green["\']?\s*:\s*(\d+)\s*,\s*["\']?blue["\']?\s*:\s*(\d+)\s*\}',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    red, green, blue = map(int, match.groups())
                    # Validate RGB values
                    if all(0 <= val <= 255 for val in [red, green, blue]):
                        return True, {'red': red, 'green': green, 'blue': blue}
                except (ValueError, AttributeError):
                    continue

        return False, None

    def _is_exact_rgb_command(self, text: str) -> bool:
        """
        Check if text is exactly 'rgb.py XXX XXX XXX' format only.

        Args:
            text: Generated text to validate

        Returns:
            True if text matches exact format, False otherwise
        """
        # Remove leading/trailing whitespace
        text = text.strip()

        # Check exact pattern: rgb.py followed by exactly 3 integers (0-255)
        pattern = r'^rgb\.py\s+(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})$'
        match = re.match(pattern, text)

        if match:
            # Validate RGB values are in valid range
            try:
                red, green, blue = map(int, match.groups())
                return all(0 <= val <= 255 for val in [red, green, blue])
            except ValueError:
                return False

        return False

    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Execute a tool with given arguments.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments for the tool

        Returns:
            Tuple of (success, result)
        """
        try:
            if tool_name == 'rgb':
                return await self._execute_rgb_tool(args)
            else:
                return False, f"Unknown tool: {tool_name}"

        except Exception as e:
            self.logger.error(f"Tool execution error: {e}")
            return False, str(e)

    async def _execute_rgb_tool(self, args: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Execute the RGB color generation tool.

        Args:
            args: RGB arguments dict with 'red', 'green', 'blue' keys

        Returns:
            Tuple of (success, result)
        """
        try:
            red = args['red']
            green = args['green']
            blue = args['blue']

            # Path to the RGB tool
            tool_path = self.tools_directory / 'rgb.py'

            if not tool_path.exists():
                return False, f"RGB tool not found: {tool_path}"

            # Execute the tool
            process = await asyncio.create_subprocess_exec(
                'python3', str(tool_path), str(red), str(green), str(blue),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout
            )

            if process.returncode == 0:
                try:
                    result = json.loads(stdout.decode())
                    return True, result
                except json.JSONDecodeError:
                    return True, stdout.decode().strip()
            else:
                return False, stderr.decode().strip()

        except asyncio.TimeoutError:
            return False, f"Tool execution timeout ({self.timeout}s)"
        except Exception as e:
            return False, str(e)

    async def run_color_experiment(self, colors: list, repetitions: int = 1) -> Dict[str, Any]:
        """
        Run color generation experiment with multiple repetitions.

        Args:
            colors: List of color names to test
            repetitions: Number of repetitions per color

        Returns:
            Dict containing experiment results
        """
        prompt_templates = self.config.get('tasks', {}).get('prompt_templates', [
            "Generate {color} color using the RGB tool."
        ])

        results = []
        total_tasks = len(colors) * len(prompt_templates) * repetitions
        completed_tasks = 0

        self.logger.info(f"Starting experiment: {len(colors)} colors, {repetitions} repetitions each")

        for color in colors:
            for template in prompt_templates:
                for rep in range(repetitions):
                    prompt = template.format(color=color)

                    self.logger.info(f"Task {completed_tasks + 1}/{total_tasks}: {color} (rep {rep + 1})")

                    result = await self.execute_mcp_task(prompt, 'rgb')
                    result['color'] = color
                    result['repetition'] = rep + 1
                    result['template'] = template

                    results.append(result)
                    completed_tasks += 1

                    if completed_tasks % 10 == 0:
                        success_rate = sum(1 for r in results if r['success']) / len(results) * 100
                        self.logger.info(f"Progress: {completed_tasks}/{total_tasks}, Success rate: {success_rate:.1f}%")

        # Calculate summary statistics
        successful = [r for r in results if r['success']]
        success_rate = len(successful) / len(results) * 100 if results else 0

        summary = {
            'total_tasks': len(results),
            'successful_tasks': len(successful),
            'success_rate': success_rate,
            'results': results,
            'colors_tested': colors,
            'repetitions': repetitions
        }

        self.logger.info(f"Experiment completed: {len(successful)}/{len(results)} tasks successful ({success_rate:.1f}%)")

        return summary