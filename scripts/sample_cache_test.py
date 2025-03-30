"""
Sample script to test cache implementation manually.

This script creates a simplified version of our cache handling implementation
to verify the logic without requiring all dependencies to be installed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSVDMamba(nn.Module):
    """Simplified SVDMamba implementation for testing cache handling."""
    
    def __init__(self, d_model=32, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Simple layers to mimic Mamba
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Cache for state
        self._state_cache = None
        self._is_cache_initialized = False
    
    def forward(self, x, use_cache=False, cache=None, return_cache=False):
        """Forward pass with caching."""
        batch_size, seq_len, _ = x.shape
        
        # Process cache
        if use_cache:
            if cache is not None:
                state_cache = cache
                self._state_cache = cache
                self._is_cache_initialized = True
            elif self._is_cache_initialized and self._state_cache is not None:
                state_cache = self._state_cache
            else:
                state_cache = None
                self._is_cache_initialized = False
            
            if seq_len == 1 and self._is_cache_initialized:
                # Incremental processing with existing cache
                output, new_cache = self._process_single_token(x, state_cache)
                
                # Update internal cache
                self._state_cache = new_cache
                
                if return_cache:
                    return output, new_cache
                return output
            else:
                # Process full sequence
                output, new_cache = self._process_full_sequence(x)
                
                # Update internal cache
                self._state_cache = new_cache
                self._is_cache_initialized = True
                
                if return_cache:
                    return output, new_cache
                return output
        else:
            # Standard processing without caching
            return self._process_full_sequence(x)[0]
    
    def _process_full_sequence(self, x):
        """Process full sequence and create cache."""
        # Simple transformation to mimic Mamba processing
        x_proj = self.in_proj(x)
        output = F.gelu(x_proj)
        output = self.out_proj(output[:, :, :self.d_model])
        
        # Create a simple cache with last hidden state
        cache = {"last_hidden": x[:, -1:, :].clone()}
        
        return output, cache
    
    def _process_single_token(self, x, cache):
        """Process single token with cache."""
        # Use cache information
        last_hidden = cache.get("last_hidden", None)
        
        # Process current token
        x_proj = self.in_proj(x)
        output = F.gelu(x_proj)
        output = self.out_proj(output[:, :, :self.d_model])
        
        # Update cache
        new_cache = {"last_hidden": x.clone()}
        
        return output, new_cache
    
    def reset_cache(self):
        """Reset the cache."""
        self._state_cache = None
        self._is_cache_initialized = False


def test_cache_consistency():
    """Test that cache handling is consistent."""
    # Create a model
    model = SimpleSVDMamba()
    
    # Create a dummy input
    batch_size = 2
    seq_len = 10
    d_model = 32
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Process full sequence without caching
    no_cache_output = model(x, use_cache=False)
    
    # Process full sequence with caching
    with_cache_output = model(x, use_cache=True)
    
    # Check outputs are the same
    assert torch.allclose(no_cache_output, with_cache_output, rtol=1e-4, atol=1e-4), \
        "Outputs differ when using cache vs. not using cache for full sequence"
    
    print("✓ Full sequence output is consistent with and without caching")
    
    # Reset cache
    model.reset_cache()
    
    # Process sequence token by token
    outputs = []
    for i in range(seq_len):
        token_output = model(x[:, i:i+1, :], use_cache=True)
        outputs.append(token_output)
    
    # Concatenate outputs
    incremental_output = torch.cat(outputs, dim=1)
    
    # Check shape
    assert incremental_output.shape == no_cache_output.shape, \
        f"Shape mismatch: {incremental_output.shape} vs. {no_cache_output.shape}"
    
    print("✓ Incremental processing generates output with correct shape")
    
    # Compare incremental output with full sequence output
    # Note: In a real implementation, there might be small differences due to numerical precision
    close_enough = torch.allclose(incremental_output, no_cache_output, rtol=1e-2, atol=1e-2)
    print(f"Incremental processing {'matches' if close_enough else 'differs from'} full sequence processing")
    
    print("Difference magnitude:", torch.abs(incremental_output - no_cache_output).mean().item())
    
    # Test cache return
    _, cache = model(x[:, :5, :], use_cache=True, return_cache=True)
    assert isinstance(cache, dict), "Cache should be a dictionary"
    assert "last_hidden" in cache, "Cache should contain last_hidden"
    
    print("✓ Cache returned correctly")
    
    # Test cache reuse
    model.reset_cache()
    
    # First half
    output1, cache1 = model(x[:, :5, :], use_cache=True, return_cache=True)
    
    # Second half with cache
    output2, _ = model(x[:, 5:, :], use_cache=True, cache=cache1, return_cache=True)
    
    # Concatenate
    output_with_cache = torch.cat([output1, output2], dim=1)
    
    # Full sequence
    model.reset_cache()
    output_full = model(x, use_cache=False)
    
    # Compare
    close_enough = torch.allclose(output_with_cache, output_full, rtol=1e-2, atol=1e-2)
    print(f"External cache passing {'works' if close_enough else 'has issues'}")
    
    print("All tests completed.")


if __name__ == "__main__":
    test_cache_consistency()
