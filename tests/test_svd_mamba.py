"""
Test for SVD-capable Mamba implementation

This module contains tests to verify that the SVF adaptation
works correctly with Mamba state-space model parameters.
"""

import torch
import unittest
import sys
import os

# Add the project root to the path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.mambaformer_layer import MambaformerLayer
from model.svd_mamba import SVDMamba


class TestSVDMamba(unittest.TestCase):
    """Test cases for SVD-capable Mamba implementation"""
    
    def setUp(self):
        """Set up common test components"""
        self.d_model = 32
        self.n_heads = 4
        self.d_state = 16
        self.seq_len = 10
        self.batch_size = 2
        
        # Create a random input tensor
        self.x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Create Mamba layer
        self.mamba = SVDMamba(
            d_model=self.d_model,
            d_state=self.d_state,
            dropout=0.0  # No dropout for testing
        )
        
        # Create MambaformerLayer
        self.layer = MambaformerLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_state=self.d_state,
            dropout=0.0  # No dropout for testing
        )
    
    def test_svd_mamba_forward(self):
        """Test that SVDMamba forward pass works"""
        output = self.mamba(self.x)
        self.assertEqual(output.shape, self.x.shape)
    
    def test_get_singular_values(self):
        """Test that SVDMamba can get singular values"""
        singular_values = self.mamba.get_singular_values()
        self.assertIsInstance(singular_values, dict)
        self.assertGreater(len(singular_values), 0)
        
        # Verify that all values are tensors
        for name, value in singular_values.items():
            self.assertIsInstance(value, torch.Tensor)
    
    def test_update_weights(self):
        """Test that SVDMamba can update weights with modified singular values"""
        # Get original singular values
        original_singular_values = self.mamba.get_singular_values()
        
        # Create expert vector (double all singular values)
        expert_vector = {name: torch.ones_like(S) * 2.0 for name, S in original_singular_values.items()}
        
        # Update weights
        self.mamba.update_weights(expert_vector)
        
        # Get new singular values
        new_singular_values = self.mamba.get_singular_values()
        
        # Verify that singular values have been updated
        for name, S in new_singular_values.items():
            if name in original_singular_values:
                # The new values should be approximately 2x the original values
                # (approximately because SVD decomposition and reconstruction isn't exact)
                ratio = S / original_singular_values[name]
                self.assertTrue(torch.allclose(ratio, torch.ones_like(ratio) * 2.0, rtol=1e-1, atol=1e-1))
    
    def test_mambaformer_layer_integration(self):
        """Test that MambaformerLayer integrates SVDMamba correctly"""
        # Get original output
        original_output = self.layer(self.x)
        
        # Get singular values
        singular_values = self.layer.get_singular_values()
        self.assertIn("mamba_block", singular_values)
        self.assertIn("fc1", singular_values)
        self.assertIn("fc2", singular_values)
        
        # Create expert vector (all ones)
        expert_vector = torch.ones(self.d_model)
        
        # Forward pass with expert vector
        adapted_output = self.layer(self.x, expert_vector)
        
        # Shapes should be the same
        self.assertEqual(adapted_output.shape, original_output.shape)
        
        # Outputs should be different
        self.assertFalse(torch.allclose(adapted_output, original_output))
    
    def test_mambaformer_layer_dict_expert(self):
        """Test that MambaformerLayer handles dictionary expert vectors"""
        # Get singular values
        singular_values = self.layer.get_singular_values()
        
        # Create expert vector dictionary
        expert_vector = {
            "mamba_block": {
                name: torch.ones_like(S) * 2.0
                for name, S in singular_values["mamba_block"].items()
            },
            "fc1": torch.ones_like(singular_values["fc1"]) * 1.5,
            "fc2": torch.ones_like(singular_values["fc2"]) * 0.5
        }
        
        # Forward pass with expert vector
        output = self.layer(self.x, expert_vector)
        
        # Shape should be correct
        self.assertEqual(output.shape, self.x.shape)


if __name__ == "__main__":
    unittest.main()
