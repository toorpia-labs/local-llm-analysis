#!/usr/bin/env python3
"""
Transformers model loader for hidden states extraction.
"""

import torch
import logging
from typing import Dict, Any, Tuple, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)


class TransformersLoader:
    """Model loader for HuggingFace Transformers models with hidden states extraction."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TransformersLoader.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.generation_config = config.get('generation', {})
        self.analysis_config = config.get('analysis', {})

        self.model_name = self.model_config.get('model_name', 'microsoft/DialoGPT-medium')
        self.device = self._get_device()

        self.tokenizer = None
        self.model = None
        self.logger = logging.getLogger(__name__)

    def _get_device(self) -> str:
        """Determine the appropriate device for model execution."""
        device_config = self.model_config.get('device', 'auto')

        if device_config == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device_config

    def load_model(self) -> Any:
        """
        Load the model and tokenizer.

        Returns:
            The loaded model
        """
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            self.logger.info(f"Device: {self.device}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.model_config.get('trust_remote_code', False)
            )

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            torch_dtype = getattr(torch, self.model_config.get('torch_dtype', 'float32'))

            model_kwargs = {
                'torch_dtype': torch_dtype,
                'trust_remote_code': self.model_config.get('trust_remote_code', False),
                'output_hidden_states': True,  # Always enable hidden states
            }

            # Add device mapping for multi-GPU
            if 'cuda' in self.device:
                model_kwargs['device_map'] = 'auto'

            # Load in 8-bit if specified
            if self.model_config.get('load_in_8bit', False):
                model_kwargs['load_in_8bit'] = True

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            # Move to device if not using device_map
            if 'device_map' not in model_kwargs:
                self.model = self.model.to(self.device)

            self.model.eval()
            self.logger.info("Model loaded successfully")

            return self.model

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dict containing model information
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        config = self.model.config

        return {
            'model_name': self.model_name,
            'num_layers': getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 'unknown')),
            'hidden_size': getattr(config, 'hidden_size', getattr(config, 'n_embd', 'unknown')),
            'vocab_size': getattr(config, 'vocab_size', 'unknown'),
            'max_position_embeddings': getattr(config, 'max_position_embeddings', getattr(config, 'n_positions', 'unknown')),
            'device': str(self.model.device) if hasattr(self.model, 'device') else self.device,
            'dtype': str(self.model.dtype) if hasattr(self.model, 'dtype') else 'unknown'
        }

    def generate_with_states(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text and extract hidden states.

        Args:
            prompt: Input prompt

        Returns:
            Tuple of (generated_text, hidden_states)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                padding=True,
                truncation=True
            )

            # Move inputs to the same device as the model
            # Handle device_map='auto' case where model might be distributed
            if hasattr(self.model, 'device'):
                model_device = self.model.device
            else:
                # For distributed models, use the device of the first parameter
                model_device = next(self.model.parameters()).device

            inputs = inputs.to(model_device)

            # Generation configuration
            gen_config = GenerationConfig(
                max_new_tokens=self.generation_config.get('max_new_tokens', 50),
                do_sample=self.generation_config.get('do_sample', True),
                temperature=self.generation_config.get('temperature', 0.7),
                top_p=self.generation_config.get('top_p', 0.9),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

            # Generate with hidden states
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config
                )

            # Extract generated text
            generated_tokens = outputs.sequences[0]
            input_length = inputs['input_ids'].shape[1]
            new_tokens = generated_tokens[input_length:]

            generated_text = self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True
            ).strip()

            # Extract hidden states if available
            hidden_states = {}
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                hidden_states = self._extract_hidden_states(
                    outputs.hidden_states,
                    new_tokens
                )

            self.logger.debug(f"Generated: '{generated_text}'")

            return generated_text, hidden_states

        except Exception as e:
            self.logger.error(f"Error during generation: {e}")
            raise

    def extract_hidden_states(self, input_ids, attention_mask=None) -> Dict[str, Any]:
        """
        Extract hidden states for given input IDs (Issue #1 API requirement).

        Args:
            input_ids: Input token IDs (torch.Tensor or list)
            attention_mask: Optional attention mask

        Returns:
            Dict containing extracted hidden states
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Convert to tensor if needed
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)

            # Ensure proper shape (add batch dimension if needed)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

            # Move to model device
            if hasattr(self.model, 'device'):
                device = self.model.device
            else:
                device = next(self.model.parameters()).device

            input_ids = input_ids.to(device)

            # Create attention mask if not provided
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            elif not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.tensor(attention_mask)

            attention_mask = attention_mask.to(device)

            # Forward pass to get hidden states
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

            # Extract hidden states from all layers
            hidden_states = outputs.hidden_states

            # Process and structure the hidden states
            return self._process_hidden_states(hidden_states, input_ids)

        except Exception as e:
            self.logger.error(f"Error extracting hidden states: {e}")
            raise

    def _process_hidden_states(self, hidden_states_tuple, input_ids) -> Dict[str, Any]:
        """
        Process hidden states into structured format.

        Args:
            hidden_states_tuple: Tuple of hidden states from model
            input_ids: Input token IDs

        Returns:
            Dict containing processed hidden states
        """
        try:
            target_layers = self.analysis_config.get('target_layers', [-1])
            save_raw = self.analysis_config.get('save_raw_states', False)

            processed = {}

            if hidden_states_tuple:
                for layer_idx in target_layers:
                    if abs(layer_idx) <= len(hidden_states_tuple):
                        layer_states = hidden_states_tuple[layer_idx]
                        layer_states_cpu = layer_states.cpu().numpy()

                        layer_data = {
                            'shape': layer_states_cpu.shape,
                            'mean': float(layer_states_cpu.mean()),
                            'std': float(layer_states_cpu.std()),
                            'min': float(layer_states_cpu.min()),
                            'max': float(layer_states_cpu.max())
                        }

                        if save_raw:
                            layer_data['vector'] = layer_states_cpu

                        processed[f'layer_{layer_idx}'] = layer_data

            return processed

        except Exception as e:
            self.logger.warning(f"Error processing hidden states: {e}")
            return {}

    def _extract_hidden_states(self, hidden_states_tuple, generated_tokens) -> Dict[str, Any]:
        """
        Extract and process hidden states from generation output.

        Args:
            hidden_states_tuple: Hidden states from model generation
            generated_tokens: The generated token IDs

        Returns:
            Dict containing processed hidden states
        """
        try:
            # Configuration
            target_layers = self.analysis_config.get('target_layers', [-1])
            save_raw = self.analysis_config.get('save_raw_states', False)
            extract_for = self.analysis_config.get('extract_for_tokens', 'output_only')

            extracted_states = {}

            if hidden_states_tuple:
                # Determine which steps to extract based on extract_for_tokens
                if extract_for == 'output_only':
                    # Only extract for newly generated tokens
                    steps_to_extract = range(len(hidden_states_tuple))
                elif extract_for == 'all':
                    # Extract for all tokens (including input)
                    steps_to_extract = range(len(hidden_states_tuple))
                else:
                    steps_to_extract = range(len(hidden_states_tuple))

                # Get states from target layers
                for step_idx in steps_to_extract:
                    if step_idx < len(hidden_states_tuple):
                        step_states = hidden_states_tuple[step_idx]
                        if step_states is not None:
                            step_extracted = {}

                            for layer_idx in target_layers:
                                if abs(layer_idx) <= len(step_states):
                                    layer_states = step_states[layer_idx]
                                    # Convert to CPU
                                    layer_states_cpu = layer_states.cpu().numpy()

                                    layer_data = {
                                        'shape': layer_states_cpu.shape,
                                        'mean': float(layer_states_cpu.mean()),
                                        'std': float(layer_states_cpu.std()),
                                        'min': float(layer_states_cpu.min()),
                                        'max': float(layer_states_cpu.max())
                                    }

                                    # Save raw vector data if requested
                                    if save_raw:
                                        # Store the full hidden state vector
                                        # Shape: (batch_size, seq_len, hidden_dim)
                                        layer_data['vector'] = layer_states_cpu

                                    step_extracted[f'layer_{layer_idx}'] = layer_data

                            extracted_states[f'step_{step_idx}'] = step_extracted

            return extracted_states

        except Exception as e:
            self.logger.warning(f"Error extracting hidden states: {e}")
            return {}

    def cleanup(self):
        """Clean up model resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model

        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("Model resources cleaned up")