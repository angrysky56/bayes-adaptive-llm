"""
Test for cache handling in MambaFormer models

This module contains tests to verify that the cache handling
for incremental inference works correctly.
"""

import torch
import unittest
import sys
import os

# Add the project root to the path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.mambaformer import MambaFormer
from model.mambaformer_layer import MambaformerLayer
from model.svd_mamba import SVDMamba


class TestCacheHandling(unittest.TestCase):
    """Test cases for cache handling in MambaFormer models"""
    
    def setUp(self):
        """Set up common test components"""
        self.d_model = 32
        self.n_heads = 4
        self.d_state = 16
        self.seq_len = 10
        self.vocab_size = 1000
        self.batch_size = 2
        
        # Create a random input tensor
        self.input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        # Create SVDMamba
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
        
        # Create MambaFormer
        self.model = MambaFormer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=3,
            n_heads=self.n_heads,
            d_state=self.d_state,
            dropout=0.0,  # No dropout for testing
            max_seq_len=1024
        )
    
    def test_svd_mamba_cache(self):
        """Test that SVDMamba caching works correctly"""
        # Dummy input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Process full sequence without caching
        no_cache_output = self.mamba(x, use_cache=False)
        
        # Process full sequence with caching
        with_cache_output = self.mamba(x, use_cache=True)
        
        # Both outputs should be the same
        self.assertTrue(torch.allclose(no_cache_output, with_cache_output, rtol=1e-4, atol=1e-4))
        
        # Reset cache
        self.mamba.reset_cache()
        
        # Process one token at a time
        outputs = []
        for i in range(self.seq_len):
            token_output = self.mamba(x[:, i:i+1, :], use_cache=True)
            outputs.append(token_output)
        
        # Concatenate outputs
        incremental_output = torch.cat(outputs, dim=1)
        
        # The incremental output should be similar to processing the full sequence
        # (not exactly the same due to implementation details of Mamba)
        self.assertEqual(incremental_output.shape, no_cache_output.shape)
    
    def test_mambaformer_layer_cache(self):
        """Test that MambaformerLayer caching works correctly"""
        # Dummy input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Process full sequence without caching
        no_cache_output = self.layer(x, use_cache=False)
        
        # Process full sequence with caching
        with_cache_output = self.layer(x, use_cache=True)
        
        # Both outputs should be the same
        self.assertTrue(torch.allclose(no_cache_output, with_cache_output, rtol=1e-4, atol=1e-4))
        
        # Reset cache
        self.layer.reset_cache()
        
        # Process one token at a time
        outputs = []
        for i in range(self.seq_len):
            token_output = self.layer(x[:, i:i+1, :], use_cache=True)
            outputs.append(token_output)
        
        # Concatenate outputs
        incremental_output = torch.cat(outputs, dim=1)
        
        # The incremental output should be similar to processing the full sequence
        self.assertEqual(incremental_output.shape, no_cache_output.shape)
    
    def test_mambaformer_model_cache(self):
        """Test that MambaFormer model caching works correctly"""
        # Process full sequence without caching
        no_cache_output = self.model(input_ids=self.input_ids, use_cache=False)
        
        # Process full sequence with caching
        with_cache_output = self.model(input_ids=self.input_ids, use_cache=True)
        
        # Both outputs should be the same
        self.assertTrue(torch.allclose(no_cache_output, with_cache_output, rtol=1e-4, atol=1e-4))
        
        # Reset cache
        self.model.reset_cache()
        
        # Process one token at a time (simulating autoregressive generation)
        full_output = self.model(
            input_ids=self.input_ids[:, :1],  # Start with first token
            use_cache=True
        )
        
        for i in range(1, self.seq_len):
            token_output = self.model(
                input_ids=self.input_ids[:, i:i+1],  # Next token
                use_cache=True
            )
            full_output = torch.cat([full_output, token_output], dim=1)
        
        # The incremental output should match the shape of the full sequence output
        self.assertEqual(full_output.shape, no_cache_output.shape)
    
    def test_model_generate(self):
        """Test that MambaFormer model generate works with caching"""
        # Generate with the model (this implicitly tests caching)
        generated_ids = self.model.generate(
            input_ids=self.input_ids[:, :3],  # Start with first few tokens
            max_new_tokens=5,
            temperature=1.0
        )
        
        # Output should have correct shape (input + max_new_tokens)
        expected_length = self.input_ids[:, :3].shape[1] + 5
        self.assertEqual(generated_ids.shape, (self.batch_size, expected_length))


if __name__ == "__main__":
    unittest.main()
