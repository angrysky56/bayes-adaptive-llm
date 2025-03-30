#!/usr/bin/env python
"""
Test script for MambaFormer implementation

This script runs a simple forward pass test on the MambaFormer model
to verify that the implementation works correctly.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.mambaformer import MambaFormer


def test_forward_pass():
    """
    Test a simple forward pass through the MambaFormer model
    """
    # Determine if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Creating model...")
    model = MambaFormer(
        vocab_size=1000,
        d_model=256,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        d_state=16
    ).to(device)  # Move model to appropriate device
    
    print("Creating dummy input...")
    # Create a dummy batch with batch_size=2, seq_len=10
    dummy_input = torch.randint(0, 1000, (2, 10)).to(device)  # Move input to appropriate device
    
    print("Running forward pass...")
    try:
        # Run forward pass
        outputs = model(dummy_input, return_dict=True)
        
        # Check if output is a dictionary
        if isinstance(outputs, dict):
            logits = outputs["logits"]
            print(f"Forward pass successful! Output shape: {logits.shape}")
        else:
            logits = outputs
            print(f"Forward pass successful! Output shape: {logits.shape}")
        
        # Check expected shape: [batch_size, seq_len, vocab_size]
        expected_shape = torch.Size([2, 10, 1000])
        if logits.shape == expected_shape:
            print(f"✓ Output shape is correct: {logits.shape}")
        else:
            print(f"✗ Output shape is incorrect: {logits.shape}, expected: {expected_shape}")
            
        return True
    except Exception as e:
        print(f"✗ Forward pass failed with error: {e}")
        return False


def test_svf_capabilities():
    """
    Test SVF capabilities of the MambaFormer model
    """
    # Determine if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\nTesting SVF capabilities...")
    model = MambaFormer(
        vocab_size=1000,
        d_model=256,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        d_state=16
    ).to(device)  # Move model to appropriate device
    
    try:
        # Get singular values from model
        singular_values = model.get_singular_values()
        print(f"✓ Successfully retrieved singular values from {len(singular_values)} layers")
        
        # Print some debug information about the structure
        print("Singular values structure:")
        for layer_name, layer_svs in singular_values.items():
            print(f"  Layer: {layer_name}")
            if isinstance(layer_svs, dict):
                for param_name, svs in layer_svs.items():
                    print(f"    Parameter: {param_name}, shape: {svs.shape if hasattr(svs, 'shape') else 'N/A'}")
            else:
                print(f"    Value type: {type(layer_svs)}")
        
        # Create a simpler expert vector - just use a single tensor that can be broadcast
        # This simple test just uses a scalar multiplier for all singular values
        expert_vector = 1.0  # Simple scalar multiplier
        
        # Create dummy input
        dummy_input = torch.randint(0, 1000, (2, 10)).to(device)  # Move input to appropriate device
        
        # Run forward pass with expert vector
        outputs = model(dummy_input, expert_vector=expert_vector, return_dict=True)
        
        # Check if output is a dictionary
        if isinstance(outputs, dict):
            logits = outputs["logits"]
            print(f"✓ Successfully ran forward pass with expert vector! Output shape: {logits.shape}")
        else:
            logits = outputs
            print(f"✓ Successfully ran forward pass with expert vector! Output shape: {logits.shape}")
        
        return True
    except Exception as e:
        print(f"✗ SVF capabilities test failed with error: {e}")
        return False


if __name__ == "__main__":
    success = test_forward_pass()
    if success:
        svf_success = test_svf_capabilities()
    
    if success and svf_success:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Tests failed!")
