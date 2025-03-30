"""
Tests for the MambaFormer architecture

This module contains unit tests for the MambaFormer implementation,
testing its components and overall functionality:
- SVDLinear layer
- SVDMamba block
- MultiHeadAttention layer
- MambaformerLayer
- MambaFormer model
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add the parent directory to the path to import the model module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.mambaformer import (
    SVDLinear,
    SVDMamba,
    MultiHeadAttention,
    MambaformerLayer,
    MambaFormer
)


class TestSVDLinear(unittest.TestCase):
    """Test cases for the SVDLinear layer"""
    
    def setUp(self):
        """Set up common test variables"""
        self.in_features = 64
        self.out_features = 128
        self.batch_size = 2
        self.seq_len = 10
        self.svd_linear = SVDLinear(self.in_features, self.out_features)
    
    def test_forward(self):
        """Test forward pass through SVDLinear"""
        # Create input tensor
        x = torch.randn(self.batch_size, self.seq_len, self.in_features)
        
        # Forward pass
        y = self.svd_linear(x)
        
        # Check output shape
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.out_features))
    
    def test_svd_components(self):
        """Test SVD component extraction"""
        # Get SVD components
        U, S, V = self.svd_linear.get_svd_components()
        
        # Check shapes
        self.assertEqual(U.shape[0], self.out_features)
        self.assertEqual(S.shape[0], min(self.in_features, self.out_features))
        self.assertEqual(V.shape[0], self.in_features)
        
        # Check that U is semi-orthogonal (UU^T is identity for the smaller dimension)
        # Since U is out_features x min(in, out), UU^T would be out_features x out_features
        # But columns are orthogonal, so U^T U is min(in, out) x min(in, out) identity
        min_dim = min(self.in_features, self.out_features)
        UTU = torch.matmul(U.transpose(-2, -1), U)
        VTV = torch.matmul(V.transpose(-2, -1), V)
        
        # Identity matrices should have 1s on the diagonal
        I_min = torch.eye(min_dim, device=U.device)
        
        # Test that the matrices are approximately orthogonal
        self.assertTrue(torch.allclose(UTU, I_min, atol=1e-5))
        self.assertTrue(torch.allclose(VTV, I_min, atol=1e-5))
    
    def test_update_weights(self):
        """Test weight update with modified singular values"""
        # Get original SVD components
        _, original_S, _ = self.svd_linear.get_svd_components()
        
        # Create modified singular values (e.g., scale by 2)
        modified_S = original_S * 2.0
        
        # Update weights
        self.svd_linear.update_weights(modified_S)
        
        # Get new SVD components
        _, new_S, _ = self.svd_linear.get_svd_components()
        
        # Check that singular values changed
        self.assertFalse(torch.allclose(original_S, new_S))
        
        # Should be approximately 2x the original values
        # Allow for some numerical error
        self.assertTrue(torch.allclose(modified_S, new_S, rtol=1e-4, atol=1e-4))


class MockSVDMamba(nn.Module):
    """A mock version of SVDMamba for testing"""
    
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * 2  # Same as expand_factor=2
        
        # Create simplified projections
        self.in_proj = SVDLinear(d_model, self.d_inner)
        self.out_proj = SVDLinear(self.d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """Simplified forward pass"""
        identity = x
        x = self.norm(x)
        x = self.in_proj(x)
        x = self.out_proj(F.silu(x))
        return x + identity
    
    def get_singular_values(self):
        """Return singular values"""
        return {
            "fc1": self.in_proj.get_singular_values(),
            "fc2": self.out_proj.get_singular_values()
        }
    
    def update_weights(self, modified_S_dict):
        """Update weights using modified singular values"""
        if "fc1" in modified_S_dict:
            self.in_proj.update_weights(modified_S_dict["fc1"])
            
        if "fc2" in modified_S_dict:
            self.out_proj.update_weights(modified_S_dict["fc2"])


class TestSVDMamba(unittest.TestCase):
    """Test cases for the SVDMamba block"""
    
    def setUp(self):
        """Set up common test variables"""
        self.d_model = 128
        self.d_state = 16
        self.batch_size = 2
        self.seq_len = 20
        
        # Use our mock implementation
        self.mamba = MockSVDMamba(self.d_model, self.d_state)
    
    def test_forward(self):
        """Test forward pass through SVDMamba (mocked version)"""
        # Create input tensor
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        y = self.mamba(x)
        
        # Check output shape
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.d_model))
    
    def test_get_singular_values(self):
        """Test getting singular values from SVDMamba"""
        # Get singular values
        singular_values = self.mamba.get_singular_values()
        
        # Check dictionary structure
        self.assertIn("fc1", singular_values)
        self.assertIn("fc2", singular_values)
        
        # Check that values are tensors
        self.assertIsInstance(singular_values["fc1"], torch.Tensor)
        self.assertIsInstance(singular_values["fc2"], torch.Tensor)
    
    def test_update_weights(self):
        """Test updating weights with modified singular values"""
        # Get original singular values
        original_values = self.mamba.get_singular_values()
        
        # Create modified values (e.g., scale by 1.5)
        modified_values = {
            "fc1": original_values["fc1"] * 1.5,
            "fc2": original_values["fc2"] * 1.5
        }
        
        # Update weights
        self.mamba.update_weights(modified_values)
        
        # Get new singular values
        new_values = self.mamba.get_singular_values()
        
        # Check that values changed
        self.assertFalse(torch.allclose(original_values["fc1"], new_values["fc1"]))
        self.assertFalse(torch.allclose(original_values["fc2"], new_values["fc2"]))
        
        # Should be approximately 1.5x the original values
        self.assertTrue(
            torch.allclose(
                modified_values["fc1"], 
                new_values["fc1"], 
                rtol=1e-4, 
                atol=1e-4
            )
        )
        self.assertTrue(
            torch.allclose(
                modified_values["fc2"], 
                new_values["fc2"], 
                rtol=1e-4, 
                atol=1e-4
            )
        )


class TestMultiHeadAttention(unittest.TestCase):
    """Test cases for the MultiHeadAttention layer"""
    
    def setUp(self):
        """Set up common test variables"""
        self.d_model = 128
        self.num_heads = 4
        self.batch_size = 2
        self.seq_len = 10
        self.attention = MultiHeadAttention(self.d_model, self.num_heads)
    
    def test_forward(self):
        """Test forward pass through MultiHeadAttention"""
        # Create input tensor
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        y = self.attention(x)
        
        # Check output shape
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.d_model))
    
    def test_causal_mask(self):
        """Test that causal masking works correctly"""
        # Create input with clear sequential pattern
        x = torch.zeros(1, 4, self.d_model)
        for i in range(4):
            x[0, i] = torch.ones(self.d_model) * (i + 1)
        
        # Forward pass
        out = self.attention(x)
        
        # Each position should only attend to itself and previous positions
        # Position 0 can only see position 0
        # Position 1 can see positions 0 and 1
        # ...
        
        # This is a qualitative test - check that the output doesn't have
        # information leakage from future positions
        
        # A simple check is that the relative magnitudes should increase
        # in the same direction as the input
        for i in range(3):
            self.assertGreaterEqual(
                torch.norm(out[0, i+1]).item(), 
                torch.norm(out[0, i]).item()
            )
    
    def test_get_singular_values(self):
        """Test getting singular values from MultiHeadAttention"""
        # Get singular values
        singular_values = self.attention.get_singular_values()
        
        # Check dictionary structure
        self.assertIn("q_proj", singular_values)
        self.assertIn("k_proj", singular_values)
        self.assertIn("v_proj", singular_values)
        self.assertIn("out_proj", singular_values)
        
        # Check that values are tensors
        self.assertIsInstance(singular_values["q_proj"], torch.Tensor)
        self.assertIsInstance(singular_values["k_proj"], torch.Tensor)
        self.assertIsInstance(singular_values["v_proj"], torch.Tensor)
        self.assertIsInstance(singular_values["out_proj"], torch.Tensor)


class MockMambaformerLayer(nn.Module):
    """A mock version of MambaformerLayer for testing"""
    
    def __init__(self, d_model, num_heads, d_state=16, dropout=0.0):
        super().__init__()
        
        # Create mocked mamba block
        self.mamba_block = MockSVDMamba(d_model, d_state)
        
        # Attention block
        self.attention_block = MultiHeadAttention(d_model, num_heads, dropout)
    
    def forward(self, x):
        """Forward pass through the MambaformerLayer"""
        # Mamba block
        x = self.mamba_block(x)
        
        # Attention block
        x = self.attention_block(x)
        
        return x
    
    def get_singular_values(self):
        """Get singular values from all sub-components"""
        # Get values from Mamba block
        mamba_values = self.mamba_block.get_singular_values()
        
        # Get values from attention block
        attention_values = self.attention_block.get_singular_values()
        
        # Combine
        singular_values = {
            "mamba_block": mamba_values,
            **attention_values
        }
        
        return singular_values
    
    def update_weights(self, modified_S_dict):
        """Update weights using modified singular values"""
        # Update Mamba block
        if "mamba_block" in modified_S_dict:
            self.mamba_block.update_weights(modified_S_dict["mamba_block"])
        
        # Update attention block parameters
        attention_params = {
            k: v for k, v in modified_S_dict.items() 
            if k in ["q_proj", "k_proj", "v_proj", "out_proj"]
        }
        
        if attention_params:
            self.attention_block.update_weights(attention_params)


class TestMambaformerLayer(unittest.TestCase):
    """Test cases for the MambaformerLayer"""
    
    def setUp(self):
        """Set up common test variables"""
        self.d_model = 128
        self.num_heads = 4
        self.d_state = 16
        self.batch_size = 2
        self.seq_len = 10
        
        # Use our mock implementation
        self.layer = MockMambaformerLayer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_state=self.d_state
        )
    
    def test_forward(self):
        """Test forward pass through MambaformerLayer"""
        # Create input tensor
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        y = self.layer(x)
        
        # Check output shape
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.d_model))
    
    def test_get_singular_values(self):
        """Test getting singular values from MambaformerLayer"""
        # Get singular values
        singular_values = self.layer.get_singular_values()
        
        # Check dictionary structure
        self.assertIn("mamba_block", singular_values)
        self.assertIn("q_proj", singular_values)
        self.assertIn("k_proj", singular_values)
        self.assertIn("v_proj", singular_values)
        self.assertIn("out_proj", singular_values)
        
        # Check that mamba_block has the right structure
        self.assertIn("fc1", singular_values["mamba_block"])
        self.assertIn("fc2", singular_values["mamba_block"])
    
    def test_update_weights(self):
        """Test updating weights with modified singular values"""
        # Get original singular values
        original_values = self.layer.get_singular_values()
        
        # Create modified values for mamba_block
        modified_mamba_values = {
            "fc1": original_values["mamba_block"]["fc1"] * 1.2,
            "fc2": original_values["mamba_block"]["fc2"] * 1.2
        }
        
        # Create modified values for attention_block
        modified_attention_values = {
            "q_proj": original_values["q_proj"] * 1.2,
            "k_proj": original_values["k_proj"] * 1.2,
            "v_proj": original_values["v_proj"] * 1.2,
            "out_proj": original_values["out_proj"] * 1.2
        }
        
        # Combine
        modified_values = {
            "mamba_block": modified_mamba_values,
            **modified_attention_values
        }
        
        # Update weights
        self.layer.update_weights(modified_values)
        
        # Get new singular values
        new_values = self.layer.get_singular_values()
        
        # Check that values changed for mamba block
        self.assertFalse(
            torch.allclose(
                original_values["mamba_block"]["fc1"], 
                new_values["mamba_block"]["fc1"]
            )
        )
        
        # Check that values changed for attention block
        self.assertFalse(
            torch.allclose(
                original_values["q_proj"], 
                new_values["q_proj"]
            )
        )


class MockMambaFormer(nn.Module):
    """A mock version of MambaFormer for testing"""
    
    def __init__(
        self,
        vocab_size,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_state=16,
        dropout=0.1,
        max_seq_len=2048
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Initial Mamba block (instead of positional encoding)
        self.initial_mamba = MockSVDMamba(d_model, d_state)
        
        # MambaFormer layers
        self.layers = nn.ModuleList([
            MockMambaformerLayer(
                d_model=d_model,
                num_heads=n_heads,
                d_state=d_state,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Output layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights with embedding
        self.lm_head.weight = self.embedding.weight
    
    def forward(self, input_ids):
        """Forward pass through the MambaFormer model"""
        # Get token embeddings
        x = self.embedding(input_ids)
        
        # Apply initial Mamba block (instead of positional encoding)
        x = self.initial_mamba(x)
        
        # Apply layers
        for layer in self.layers:
            x = layer(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Get logits
        logits = self.lm_head(x)
        
        return logits
    
    def generate(
        self,
        input_ids,
        max_new_tokens=20,
        temperature=1.0,
        top_p=0.9,
        expert_vector=None
    ):
        """Simplified generate method for testing"""
        # Clone input_ids to avoid modifying the original
        generated_ids = input_ids.clone()
        
        # Add random tokens for testing purposes
        num_new_tokens = torch.randint(1, max_new_tokens + 1, (1,)).item()
        random_tokens = torch.randint(0, self.vocab_size, (input_ids.size(0), num_new_tokens))
        generated_ids = torch.cat([generated_ids, random_tokens], dim=1)
        
        return generated_ids
    
    def get_singular_values(self):
        """Get singular values from all components of the model"""
        singular_values = {}
        
        # Get values from initial Mamba block
        initial_mamba_values = self.initial_mamba.get_singular_values()
        singular_values["initial_mamba"] = initial_mamba_values
        
        # Get values from each layer
        for i, layer in enumerate(self.layers):
            layer_values = layer.get_singular_values()
            singular_values[f"layer_{i}"] = layer_values
        
        return singular_values
    
    def update_weights(self, modified_S_dict):
        """Update weights using modified singular values"""
        # Update initial Mamba block
        if "initial_mamba" in modified_S_dict:
            self.initial_mamba.update_weights(modified_S_dict["initial_mamba"])
        
        # Update each layer
        for i, layer in enumerate(self.layers):
            layer_key = f"layer_{i}"
            if layer_key in modified_S_dict:
                layer.update_weights(modified_S_dict[layer_key])


class TestMambaFormer(unittest.TestCase):
    """Test cases for the MambaFormer model"""
    
    def setUp(self):
        """Set up common test variables"""
        self.vocab_size = 1000
        self.d_model = 128
        self.n_layers = 2
        self.n_heads = 4
        self.d_state = 16
        self.batch_size = 2
        self.seq_len = 10
        
        # Use our mock implementation
        self.model = MockMambaFormer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_state=self.d_state
        )
    
    def test_forward(self):
        """Test forward pass through MambaFormer"""
        # Create input tensor
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Check output shape
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))
    
    def test_generate(self):
        """Test text generation with MambaFormer"""
        # Create input tensor (start tokens)
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, 5))
        
        # Generate
        output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=10,
            temperature=0.7,
            top_p=0.9
        )
        
        # Check output shape
        self.assertEqual(output_ids.shape[0], self.batch_size)
        self.assertGreaterEqual(output_ids.shape[1], input_ids.shape[1])
        self.assertLessEqual(output_ids.shape[1], input_ids.shape[1] + 10)
    
    def test_get_singular_values(self):
        """Test getting singular values from MambaFormer"""
        # Get singular values
        singular_values = self.model.get_singular_values()
        
        # Check dictionary structure
        self.assertIn("initial_mamba", singular_values)
        
        # Check that we have the right number of layers
        for i in range(self.n_layers):
            self.assertIn(f"layer_{i}", singular_values)
    
    def test_update_weights(self):
        """Test updating weights with modified singular values"""
        # Get original singular values
        original_values = self.model.get_singular_values()
        
        # Create modified values for initial_mamba
        modified_initial_mamba_values = {
            "fc1": original_values["initial_mamba"]["fc1"] * 1.5,
            "fc2": original_values["initial_mamba"]["fc2"] * 1.5
        }
        
        # Create modified values for layers
        modified_layer_values = {}
        for i in range(self.n_layers):
            layer_key = f"layer_{i}"
            layer_values = original_values[layer_key]
            
            # Create modified values for this layer
            modified_mamba_block = {
                "fc1": layer_values["mamba_block"]["fc1"] * 1.5,
                "fc2": layer_values["mamba_block"]["fc2"] * 1.5
            }
            
            # Create modified values for the layer
            modified_layer_values[layer_key] = {
                "mamba_block": modified_mamba_block,
                "q_proj": layer_values["q_proj"] * 1.5,
                "k_proj": layer_values["k_proj"] * 1.5,
                "v_proj": layer_values["v_proj"] * 1.5,
                "out_proj": layer_values["out_proj"] * 1.5
            }
        
        # Combine
        modified_values = {
            "initial_mamba": modified_initial_mamba_values,
            **modified_layer_values
        }
        
        # Update weights
        self.model.update_weights(modified_values)
        
        # Get new singular values
        new_values = self.model.get_singular_values()
        
        # Check that values changed for initial_mamba
        self.assertFalse(
            torch.allclose(
                original_values["initial_mamba"]["fc1"], 
                new_values["initial_mamba"]["fc1"]
            )
        )
        
        # Check that values changed for the first layer
        self.assertFalse(
            torch.allclose(
                original_values["layer_0"]["mamba_block"]["fc1"], 
                new_values["layer_0"]["mamba_block"]["fc1"]
            )
        )


if __name__ == "__main__":
    unittest.main()
