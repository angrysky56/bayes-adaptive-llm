"""
MambaformerLayer Implementation

This module implements the MambaformerLayer, which combines Mamba state space 
models (SSMs) with Transformer attention blocks. This implementation is designed 
to work with Singular Value Fine-tuning (SVF) for expert adaptation.

References:
1. Mamba: https://arxiv.org/abs/2312.00752
2. MambaFormer: https://arxiv.org/abs/2402.04248
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Optional, Tuple, Union, Any

from .svd_mamba import SVDMamba
from .layer import SVDLinear


class MambaformerLayer(nn.Module):
    """
    MambaformerLayer: A hybrid layer combining Mamba and Transformer attention
    
    This layer combines:
    1. A Mamba block for efficient sequence processing
    2. A multi-head attention block for retrieval/general ICL
    3. A feed-forward network with SVF capability
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_state: int,
        dropout: float = 0.1
    ):
        """
        Initialize a MambaformerLayer
        
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            d_state: Dimension of the state in Mamba blocks
            dropout: Dropout probability
        """
        super().__init__()
        
        # SVD-capable Mamba block
        self.mamba_block = SVDMamba(
            d_model=d_model, 
            d_state=d_state,
            dropout=dropout
        )
        
        # Attention block
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # SVF-capable linear layers for feedforward
        self.fc1 = SVDLinear(d_model, d_model * 4)
        self.fc2 = SVDLinear(d_model * 4, d_model)
        
        # Layer norms
        self.norm_mamba = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Attention KV cache for incremental inference
        self.attn_kv_cache = None
        self.cache_length = 0
        
        # Expert vector cache to avoid reapplying during incremental inference
        self.last_expert = None

    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        expert_vector: Optional[Union[torch.Tensor, Dict[str, Any]]] = None, 
        use_cache: bool = False,
        layer_cache: Optional[Dict[str, Any]] = None,
        return_cache: bool = False
    ):
        """
        Forward pass through the MambaformerLayer
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            expert_vector: Optional expert vector for SVF adaptation
            use_cache: Whether to use the cache for incremental inference
            layer_cache: Optional cache dict from a previous forward pass
            return_cache: Whether to return updated cache
            
        Returns:
            output: Output tensor [batch_size, seq_len, d_model]
            new_cache: Updated cache if return_cache=True, else None
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize cache structures if needed
        new_cache = {} if return_cache else None
        
        # Process layer_cache if provided
        if layer_cache is not None:
            self.attn_kv_cache = layer_cache.get("attn_kv_cache", None)
            self.cache_length = layer_cache.get("cache_length", 0)
            mamba_cache = layer_cache.get("mamba_cache", None)
        else:
            mamba_cache = None
        
        # Apply expert vector adaptation (SVF) if provided
        # Only apply if expert vector is different from the last one
        if expert_vector is not None and expert_vector != self.last_expert:
            if isinstance(expert_vector, dict):
                # Apply separate expert vectors to each SVF-capable layer
                if 'mamba_block' in expert_vector:
                    mamba_S = self.mamba_block.get_singular_values()
                    modified_mamba_S = {}
                    for param_name, S in mamba_S.items():
                        if param_name in expert_vector['mamba_block']:
                            modified_mamba_S[param_name] = S * expert_vector['mamba_block'][param_name]
                        else:
                            # Use the same expert vector for all parameters if not specified
                            if isinstance(expert_vector['mamba_block'], torch.Tensor):
                                modified_mamba_S[param_name] = S * expert_vector['mamba_block']
                    
                    self.mamba_block.update_weights(modified_mamba_S)
                
                if 'fc1' in expert_vector:
                    modified_S = self.fc1.get_singular_values() * expert_vector['fc1']
                    self.fc1.update_weights(modified_S)
                
                if 'fc2' in expert_vector:
                    modified_S = self.fc2.get_singular_values() * expert_vector['fc2']
                    self.fc2.update_weights(modified_S)
            else:
                # Apply same expert vector to all SVF-capable layers
                mamba_S = self.mamba_block.get_singular_values()
                modified_mamba_S = {param_name: S * expert_vector for param_name, S in mamba_S.items()}
                self.mamba_block.update_weights(modified_mamba_S)
                
                modified_S_fc1 = self.fc1.get_singular_values() * expert_vector
                self.fc1.update_weights(modified_S_fc1)
                
                modified_S_fc2 = self.fc2.get_singular_values() * expert_vector
                self.fc2.update_weights(modified_S_fc2)
            
            # Cache the applied expert vector
            self.last_expert = expert_vector
            
            # Expert changes invalidate the cache
            if use_cache:
                self.reset_cache()
        
        # Mamba block with caching
        residual = x
        if use_cache:
            # Forward through Mamba with caching
            if return_cache:
                mamba_out, new_mamba_cache = self.mamba_block(
                    x, 
                    use_cache=True, 
                    cache=mamba_cache, 
                    return_cache=True
                )
                if new_cache is not None:
                    new_cache["mamba_cache"] = new_mamba_cache
            else:
                mamba_out = self.mamba_block(
                    x, 
                    use_cache=True, 
                    cache=mamba_cache, 
                    return_cache=False
                )
        else:
            # Standard forward without caching
            mamba_out = self.mamba_block(x)
        
        x = residual + mamba_out
        x = self.norm_mamba(x)
        
        # Attention block with KV caching
        residual = x
        
        if use_cache and seq_len == 1 and self.attn_kv_cache is not None:
            # Incremental inference with cached attention
            # Extract cached past key and value states
            if isinstance(self.attn_kv_cache, tuple):
                past_key, past_value = self.attn_kv_cache
                
                # Generate key and value for current token
                key_states, value_states = self._get_attn_kv(x)
                
                # Concatenate with past keys and values
                key_states = torch.cat([past_key, key_states], dim=1)
                value_states = torch.cat([past_value, value_states], dim=1)
                
                # Update cache
                self.attn_kv_cache = (key_states, value_states)
                if return_cache and new_cache is not None:
                    new_cache["attn_kv_cache"] = self.attn_kv_cache
                
                # Compute attention using cached keys and values and attention mask
                attn_mask = self._prepare_attention_mask(attention_mask, x.shape[0], x.shape[1]) if attention_mask is not None else None
                attn_out, _ = self.attn(x, key_states, value_states, attn_mask=attn_mask)
            else:
                # First token or cache not initialized properly
                attn_mask = self._prepare_attention_mask(attention_mask, x.shape[0], x.shape[1]) if attention_mask is not None else None
                attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
                
                # Initialize cache with current key and value
                key_states, value_states = self._get_attn_kv(x)
                self.attn_kv_cache = (key_states, value_states)
                if return_cache and new_cache is not None:
                    new_cache["attn_kv_cache"] = self.attn_kv_cache
            
            # Update cache length
            self.cache_length += 1
            if return_cache and new_cache is not None:
                new_cache["cache_length"] = self.cache_length
        else:
            # Standard attention forward pass with attention mask
            attn_mask = self._prepare_attention_mask(attention_mask, x.shape[0], x.shape[1]) if attention_mask is not None else None
            attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
            
            # Initialize/update cache if needed
            if use_cache:
                # Initialize attention KV cache
                key_states, value_states = self._get_attn_kv(x)
                self.attn_kv_cache = (key_states, value_states)
                self.cache_length = seq_len
                
                if return_cache and new_cache is not None:
                    new_cache["attn_kv_cache"] = self.attn_kv_cache
                    new_cache["cache_length"] = self.cache_length
        
        x = residual + attn_out
        x = self.norm_attn(x)
        
        # Feed-forward block (same for both cases)
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = residual + x
        x = self.norm_ff(x)
        
        if return_cache:
            return x, new_cache
        return x
    
    def _prepare_attention_mask(self, attention_mask, batch_size, seq_length):
        """
        Convert attention mask from HuggingFace format (1=attend, 0=mask) to PyTorch format
        
        Args:
            attention_mask: Attention mask in shape [batch_size, seq_length] (1=attend, 0=mask)
            batch_size: Batch size
            seq_length: Sequence length
            
        Returns:
            Attention mask in PyTorch format for nn.MultiheadAttention
        """
        # If attention_mask is None, return None
        if attention_mask is None:
            return None
        
        # PyTorch's MultiheadAttention accepts attention masks in two formats:
        # 1. attn_mask: a 2D mask of shape (L, S) or a 3D mask of shape (N*num_heads, L, S)
        # 2. key_padding_mask: a 2D mask of shape (N, S)
        
        # Ensure attention_mask is a bool or float tensor as required by PyTorch
        if not isinstance(attention_mask, (bool, torch.BoolTensor, torch.FloatTensor)):
            if hasattr(attention_mask, 'dtype'):
                if attention_mask.dtype == torch.int64 or attention_mask.dtype == torch.int32:
                    attention_mask = attention_mask.to(torch.bool)
                else:
                    attention_mask = attention_mask.to(torch.float)
        
        # For key_padding_mask, we need to invert the mask (in HF, 1=attend, 0=mask; in PyTorch, True=mask, False=attend)
        # We'll use key_padding_mask as it's easier to work with for padding masks
        if attention_mask.dim() == 2:
            # Check if the mask has the correct dimensions for attn_mask
            # PyTorch expects attn_mask to be of shape [seq_length, seq_length]
            if attention_mask.size(0) != seq_length or attention_mask.size(1) != seq_length:
                # Create a proper square attention mask
                proper_mask = attention_mask.new_zeros(seq_length, seq_length)
                
                # If this is a causal mask (future tokens should be masked)
                if hasattr(self, 'is_causal') and self.is_causal:
                    # Create causal mask (lower triangular)
                    causal_mask = torch.triu(
                        torch.ones(seq_length, seq_length, dtype=torch.bool, device=attention_mask.device),
                        diagonal=1
                    )
                    # Apply causal masking (PyTorch uses -inf for masked positions)
                    proper_mask.masked_fill_(causal_mask, float('-inf'))
                
                return proper_mask.to(torch.float)
            
            # Convert HuggingFace mask (1=attend, 0=mask) to PyTorch key_padding_mask (True=mask, False=attend)
            key_padding_mask = (attention_mask == 0).to(torch.bool)
            return key_padding_mask
        
        # If it's already in the right format, ensure it's bool or float
        return attention_mask.to(torch.bool) if attention_mask.dtype == torch.int64 or attention_mask.dtype == torch.int32 else attention_mask.to(torch.float)
    
    def _get_attn_kv(self, x):
        """
        Helper method to get key and value states for attention caching
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tuple of (key_states, value_states)
        """
        # Access internal implementations of MultiheadAttention
        if hasattr(self.attn, '_get_key_value'):
            # If the attention module has a helper method
            return self.attn._get_key_value(x)
        else:
            # Approximate implementation (this would need to be adapted for the specific MHA implementation)
            batch_size, seq_len, _ = x.shape
            device = x.device
            
            # Project input to get query, key, and value
            # Note: This is an approximation and may not match the exact implementation of nn.MultiheadAttention
            head_dim = self.attn.head_dim if hasattr(self.attn, 'head_dim') else self.attn.embed_dim // self.attn.num_heads
            
            q = self.attn.q_proj(x)
            k = self.attn.k_proj(x)
            v = self.attn.v_proj(x)
            
            return k, v
    
    def reset_cache(self):
        """
        Reset the KV cache for incremental inference
        """
        self.attn_kv_cache = None
        self.cache_length = 0
        self.mamba_block.reset_cache()
        
    def get_singular_values(self):
        """
        Get singular values from all SVF-capable layers
        
        Returns:
            Dictionary mapping layer names to singular values
        """
        return {
            "mamba_block": self.mamba_block.get_singular_values(),
            "fc1": self.fc1.get_singular_values(),
            "fc2": self.fc2.get_singular_values(),
        }
    
    def get_svd_components(self):
        """
        Get SVD components from all SVF-capable layers
        
        Returns:
            Dictionary mapping layer names to SVD components
        """
        return {
            "mamba_block": self.mamba_block.get_svd_components(),
            "fc1": self.fc1.get_svd_components(),
            "fc2": self.fc2.get_svd_components(),
        }
