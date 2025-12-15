#!/usr/bin/env python3
"""
CSV exporter for hidden states data.
"""

import csv
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional


class CSVExporter:
    """Export hidden states to CSV files, one file per layer."""

    def __init__(
        self,
        output_dir: Path,
        hidden_size: int,
        export_layers: List[int],
        n_tokens_to_average: int = 1
    ):
        """
        Initialize CSV exporter.

        Args:
            output_dir: Directory to save CSV files
            hidden_size: Hidden state dimensionality (e.g., 3072)
            export_layers: List of layer indices to export (e.g., [-1, -2])
            n_tokens_to_average: Number of initial tokens to average
        """
        self.output_dir = Path(output_dir)
        self.hidden_size = hidden_size
        self.export_layers = export_layers
        self.n_tokens_to_average = n_tokens_to_average
        self.logger = logging.getLogger(__name__)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Open CSV files for each layer
        self.csv_files = {}
        self.csv_writers = {}

        for layer_idx in export_layers:
            csv_path = self.output_dir / f"layer_{layer_idx}.csv"
            csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
            csv_writer = csv.writer(csv_file)

            # Write header
            header = ['No', 'success']
            header.extend([f'c{i+1}' for i in range(hidden_size)])
            csv_writer.writerow(header)

            self.csv_files[layer_idx] = csv_file
            self.csv_writers[layer_idx] = csv_writer

        self.logger.info(f"CSV exporter initialized: {len(export_layers)} layers, {hidden_size} dimensions")
        self.logger.info(f"Output directory: {self.output_dir}")

    def write_trial(
        self,
        trial_no: int,
        success: bool,
        generated_text: str,
        hidden_states: Dict[str, Any]
    ) -> bool:
        """
        Write a trial's hidden states to CSV files.

        Args:
            trial_no: Trial number
            success: Whether the trial succeeded
            generated_text: Generated text from the trial
            hidden_states: Hidden states dictionary from generate_with_states()

        Returns:
            True if data was written, False if skipped
        """
        try:
            # Check if we have valid hidden states
            if not hidden_states:
                self.logger.warning(f"Trial {trial_no}: No hidden states available, skipping CSV export")
                return False

            # Extract and average hidden states for each layer
            for layer_idx in self.export_layers:
                layer_key = f'layer_{layer_idx}'

                # Collect vectors from first n tokens
                vectors = []
                for step_idx in range(min(self.n_tokens_to_average, len(hidden_states))):
                    step_key = f'step_{step_idx}'
                    if step_key in hidden_states:
                        step_data = hidden_states[step_key]
                        if layer_key in step_data:
                            layer_data = step_data[layer_key]
                            if 'vector' in layer_data:
                                # Extract vector: shape is (batch_size, seq_len, hidden_dim)
                                # We want the last position in the sequence
                                vector = layer_data['vector']
                                # Take the last position: [0, -1, :]
                                if vector.ndim == 3:
                                    vectors.append(vector[0, -1, :])
                                elif vector.ndim == 2:
                                    vectors.append(vector[0, :])
                                else:
                                    self.logger.warning(f"Unexpected vector shape: {vector.shape}")

                if not vectors:
                    self.logger.warning(
                        f"Trial {trial_no}, layer {layer_idx}: No vectors found, skipping"
                    )
                    return False

                # Average the vectors
                averaged_vector = np.mean(vectors, axis=0)

                # Verify dimensionality
                if len(averaged_vector) != self.hidden_size:
                    self.logger.warning(
                        f"Trial {trial_no}, layer {layer_idx}: "
                        f"Vector size mismatch (expected {self.hidden_size}, got {len(averaged_vector)})"
                    )
                    return False

                # Prepare row
                row = [trial_no, success]
                row.extend(averaged_vector.tolist())

                # Write to CSV
                self.csv_writers[layer_idx].writerow(row)

            # Flush all files
            for csv_file in self.csv_files.values():
                csv_file.flush()

            return True

        except Exception as e:
            self.logger.error(f"Trial {trial_no}: Error writing to CSV: {e}")
            return False

    def close(self):
        """Close all CSV files."""
        for layer_idx, csv_file in self.csv_files.items():
            csv_file.close()
            csv_path = self.output_dir / f"layer_{layer_idx}.csv"
            self.logger.info(f"Closed CSV file: {csv_path}")

        self.csv_files.clear()
        self.csv_writers.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
