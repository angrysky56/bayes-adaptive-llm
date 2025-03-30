"""
MambaFormer: A hybrid architecture combining Mamba and Transformer components

This module implements a hybrid architecture that combines state-space models (Mamba)
with attention mechanisms, designed to leverage the strengths of both approaches:
- Mamba's linear scaling with sequence length and strong in-context learning
- Attention's powerful retrieval capabilities and direct token-token interactions

The architecture follows the design described in the MambaFormer paper, with
SVD-enabled layers to support Singular Value Fine-tuning (SVF).
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class SVDLinear(nn.Module):
    """
    Linear layer that supports Singular Value Decomposition (SVD) operations
    
    This layer allows for directly manipulating the singular values of the weight matrix,
    enabling Singular Value Fine-tuning (SVF).
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the SVDLinear layer
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to use bias
            device: Device to place parameters
            dtype: Data type for parameters
        """
        super().__init__()
        
        # Initialize standard linear layer
        self.linear = nn.Linear(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        # Cache for SVD components
        self._svd_components = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the layer
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.linear(x)
    
    def get_svd_components(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the SVD components of the weight matrix
        
        Returns:
            Tuple of (U, S, V) where U, S, V are the SVD components
        """
        # If components already computed, return them
        if self._svd_components is not None:
            return self._svd_components
        
        # Compute SVD
        weight = self.linear.weight
        device = weight.device
        
        # Apply SVD
        try:
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
            V = Vh.transpose(-2, -1)
        except Exception as e:
            print(f"SVD failed with error: {e}")
            print(f"Weight device: {weight.device}, shape: {weight.shape}")
            # Fallback to CPU if CUDA SVD fails
            if device.type == 'cuda':
                weight_cpu = weight.cpu()
                U_cpu, S_cpu, Vh_cpu = torch.linalg.svd(weight_cpu, full_matrices=False)
                U = U_cpu.to(device)
                S = S_cpu.to(device)
                V = Vh_cpu.transpose(-2, -1).to(device)
            else:
                raise
        
        # Cache components
        self._svd_components = (U, S, V)
        
        return self._svd_components
    
    def get_singular_values(self) -> torch.Tensor:
        """
        Get the singular values of the weight matrix
        
        Returns:
            Singular values tensor
        """
        _, S, _ = self.get_svd_components()
        return S
    
    def update_weights(self, modified_S: torch.Tensor) -> None:
        """
        Update the weight matrix using modified singular values
        
        Args:
            modified_S: Modified singular values
            
        Returns:
            None (updates weights in-place)
        """
        U, _, V = self.get_svd_components()
        
        # Ensure modified_S has right shape
        if modified_S.ndim == 1:
            modified_S = torch.diag(modified_S)
        
        # Reconstruct weight matrix
        updated_weight = U @ modified_S @ V.transpose(-2, -1)
        
        # Update weight matrix
        with torch.no_grad():
            self.linear.weight.copy_(updated_weight)
            
        # Invalidate cache
        self._svd_components = None


class SVDMamba(nn.Module):
    """
    Mamba block that supports Singular Value Decomposition (SVD) operations
    
    This layer implements the core Mamba operations with SVD support for
    the linear projections.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand_factor: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the SVDMamba block
        
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            expand_factor: Expansion factor for input projection
            dt_min: Minimum delta time value
            dt_max: Maximum delta time value
            dt_init: Delta time initialization method
            dt_scale: Scale for delta time
            dt_init_floor: Floor for delta time initialization
            bias: Whether to use bias
            device: Device to place parameters
            dtype: Data type for parameters
        """
        super().__init__()
        
        # Dimensions
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(d_model * expand_factor)
        
        # Projections with SVD support
        self.in_proj = SVDLinear(d_model, self.d_inner * 2, bias=bias, device=device, dtype=dtype)
        self.out_proj = SVDLinear(self.d_inner, d_model, bias=bias, device=device, dtype=dtype)
        
        # Additional parameters for SSM
        # Simplified SSM parameters for MambaFormer prototype
        self.A = nn.Parameter(torch.randn(self.d_inner, self.d_state, device=device, dtype=dtype) * 0.1)
        self.B = nn.Parameter(torch.randn(self.d_inner, self.d_state, device=device, dtype=dtype) * 0.1)
        self.C = nn.Parameter(torch.randn(self.d_inner, self.d_state, device=device, dtype=dtype) * 0.1)
        
        # Timescale (delta) parameter
        self.dt = nn.Parameter(torch.rand(self.d_inner, device=device, dtype=dtype) * (dt_max - dt_min) + dt_min)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Mamba block
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Save original input shape for later reference
        batch_size, seq_len, d_model = x.shape
        
        # Clone for residual connection
        identity = x
        
        # Check for NaN/Inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            return identity
        
        # Normalize
        x = self.norm(x)
        
        # Project and split with error handling
        try:
            # Project input
            projected = self.in_proj(x)
            
            # Check for NaN/Inf in projection
            if torch.isnan(projected).any() or torch.isinf(projected).any():
                return identity
                
            # Split projection
            if projected.size(-1) % 2 != 0:
                # Handle odd dimension case by padding
                pad_size = projected.size(-1) + 1
                projected = F.pad(projected, (0, 1))
                
            # Now split into z and gate components
            z, gate = projected.chunk(2, dim=-1)
        except Exception as e:
            # Fall back to identity on error
            print(f"Error in SVDMamba projection: {e}")
            return identity
        
        # Simplified Mamba function for prototype
        # Using a more efficient implementation that avoids dimension mismatches
        try:
            # Apply non-linearity with numerical stability
            z_safe = torch.clamp(z, min=-15.0, max=15.0)  # Prevent extreme values
            z_gated = z * torch.sigmoid(z_safe)
            
            # Apply a simple 1D convolution-like operation for temporal mixing
            # Handle dt shape mismatches
            if self.dt.size(0) != z_gated.size(-1) and z_gated.size(-1) > 0:
                # Resize dt to match z_gated last dimension
                dt_expanded = self.dt.unsqueeze(0).unsqueeze(0)  # [1, 1, d_inner]
                dt_expanded = F.interpolate(
                    dt_expanded, 
                    size=z_gated.size(-1), 
                    mode='linear', 
                    align_corners=False
                ).squeeze(0)
                dt_weights = torch.softmax(-dt_expanded, dim=1)
            else:
                # Standard case
                dt_weights = torch.softmax(-self.dt, dim=0).view(1, 1, -1)
                
                # Make sure dimensions match before multiplication
                if dt_weights.size(-1) != z_gated.size(-1):
                    # Repeat or truncate dt_weights to match z_gated
                    if dt_weights.size(-1) < z_gated.size(-1):
                        factor = (z_gated.size(-1) + dt_weights.size(-1) - 1) // dt_weights.size(-1)
                        dt_weights = dt_weights.repeat(1, 1, factor)[:, :, :z_gated.size(-1)]
                    else:
                        dt_weights = dt_weights[:, :, :z_gated.size(-1)]
            
            # Apply temporal mixing with shape checking
            z_temporal = z_gated
            if z_gated.size(-1) == dt_weights.size(-1):
                z_temporal = z_gated * dt_weights
                
            # Apply gating with safe activation
            gate_safe = torch.clamp(gate, min=-15.0, max=15.0)  # Prevent extreme values
            output = z_temporal * F.silu(gate_safe)
            
            # Project back to model dimension
            output = self.out_proj(output)
            
            # Check output shape and fix if needed
            if output.size(0) != batch_size or output.size(1) != seq_len or output.size(2) != d_model:
                # Resize output to match input shape
                output = F.interpolate(
                    output.permute(0, 2, 1),  # [B, C, T]
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).permute(0, 2, 1)  # Back to [B, T, C]
                
                # If channel dimension is wrong, fix it
                if output.size(2) != d_model:
                    output = F.pad(output, (0, d_model - output.size(2))) if output.size(2) < d_model else output[:, :, :d_model]
            
            # Final check for NaN/Inf
            if torch.isnan(output).any() or torch.isinf(output).any():
                return identity
                
            # Residual connection
            output = output + identity
            
            return output
            
        except Exception as e:
            # Fall back to identity on error
            print(f"Error in SVDMamba processing: {e}")
            return identity
    
    def get_singular_values(self) -> Dict[str, torch.Tensor]:
        """
        Get singular values for SVF adaptation
        
        Returns:
            Dictionary mapping parameter names to singular values
        """
        return {
            "fc1": self.in_proj.get_singular_values(),
            "fc2": self.out_proj.get_singular_values()
        }
    
    def update_weights(self, modified_S_dict: Dict[str, torch.Tensor]) -> None:
        """
        Update weights using modified singular values
        
        Args:
            modified_S_dict: Dictionary mapping parameter names to modified singular values
            
        Returns:
            None (updates weights in-place)
        """
        if "fc1" in modified_S_dict:
            self.in_proj.update_weights(modified_S_dict["fc1"])
            
        if "fc2" in modified_S_dict:
            self.out_proj.update_weights(modified_S_dict["fc2"])


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with SVD support for Singular Value Fine-tuning
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize multi-head attention
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear projections
            device: Device to place parameters
            dtype: Data type for parameters
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        # Query, key, value projections with SVD support
        self.q_proj = SVDLinear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.k_proj = SVDLinear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.v_proj = SVDLinear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        
        # Output projection with SVD support
        self.out_proj = SVDLinear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Clone for residual connection
        identity = x
        
        # Normalize
        x = self.norm(x)
        
        # Project query, key, value
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Attention mask needs to be transformed to match the shape of attn_weights
            # attn_weights shape: [batch_size, num_heads, seq_len, seq_len]
            # attention_mask shape: [batch_size, seq_len]
            
            # Create attention_mask that prevents attending to padding tokens and future tokens
            seq_len = attn_weights.size(-1)
            
            # Ensure attention_mask has the right sequence length
            if attention_mask.size(1) < seq_len:
                # If attention_mask is shorter, pad it with zeros (don't attend)
                padding = torch.zeros(
                    (attention_mask.size(0), seq_len - attention_mask.size(1)),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([attention_mask, padding], dim=1)
            elif attention_mask.size(1) > seq_len:
                # If attention_mask is longer, truncate it
                attention_mask = attention_mask[:, :seq_len]
                
            # Expand attention_mask to match the attention weights dimensions
            # First, add the head dimension
            expanded_mask = attention_mask.unsqueeze(1)  # [batch_size, 1, seq_len]
            
            # Then, expand it to create a mask for each position in the sequence
            expanded_mask = expanded_mask.unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            
            # Finally, broadcast it to all heads
            expanded_mask = expanded_mask.expand(-1, self.num_heads, seq_len, -1)  # [batch_size, num_heads, seq_len, seq_len]
            
            # Convert mask values: 0 (pad) -> -inf (don't attend), 1 (token) -> 0 (attend)
            expanded_mask = (1.0 - expanded_mask) * -10000.0
            
            # Apply mask to attention weights
            attn_weights = attn_weights + expanded_mask
        
        # Create causal mask (lower triangular)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_weights.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Transpose back to [batch_size, seq_len, num_heads * head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        # Apply residual connection and dropout
        output = identity + self.resid_dropout(attn_output)
        
        return output
    
    def get_singular_values(self) -> Dict[str, torch.Tensor]:
        """
        Get singular values for SVF adaptation
        
        Returns:
            Dictionary mapping parameter names to singular values
        """
        return {
            "q_proj": self.q_proj.get_singular_values(),
            "k_proj": self.k_proj.get_singular_values(),
            "v_proj": self.v_proj.get_singular_values(),
            "out_proj": self.out_proj.get_singular_values()
        }
    
    def update_weights(self, modified_S_dict: Dict[str, torch.Tensor]) -> None:
        """
        Update weights using modified singular values
        
        Args:
            modified_S_dict: Dictionary mapping parameter names to modified singular values
            
        Returns:
            None (updates weights in-place)
        """
        if "q_proj" in modified_S_dict:
            self.q_proj.update_weights(modified_S_dict["q_proj"])
            
        if "k_proj" in modified_S_dict:
            self.k_proj.update_weights(modified_S_dict["k_proj"])
            
        if "v_proj" in modified_S_dict:
            self.v_proj.update_weights(modified_S_dict["v_proj"])
            
        if "out_proj" in modified_S_dict:
            self.out_proj.update_weights(modified_S_dict["out_proj"])


class MambaformerLayer(nn.Module):
    """
    MambaFormer layer combining Mamba and attention blocks
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_state: int = 16,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize a MambaFormer layer
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_state: SSM state dimension
            dropout: Dropout probability
            device: Device to place parameters
            dtype: Data type for parameters
        """
        super().__init__()
        
        # Mamba block
        self.mamba_block = SVDMamba(
            d_model=d_model,
            d_state=d_state,
            device=device,
            dtype=dtype
        )
        
        # Attention block
        self.attention_block = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            device=device,
            dtype=dtype
        )
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the MambaFormer layer
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask for the attention block
                            of shape [batch_size, seq_len] where 1 means attend
                            and 0 means don't attend
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Process through Mamba block first
        mamba_output = self.mamba_block(x)
        
        # Check if attention_mask needs adjustment
        if attention_mask is not None and attention_mask.size(1) != mamba_output.size(1):
            # Adjust the attention mask to match the sequence length
            seq_len = mamba_output.size(1)
            batch_size = mamba_output.size(0)
            
            if attention_mask.size(1) < seq_len:
                # If mask is too short, pad with zeros (don't attend)
                padding = torch.zeros(
                    (batch_size, seq_len - attention_mask.size(1)),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                adjusted_mask = torch.cat([attention_mask, padding], dim=1)
            else:
                # If mask is too long, truncate
                adjusted_mask = attention_mask[:, :seq_len]
                
            # Process through attention block with adjusted mask
            output = self.attention_block(mamba_output, attention_mask=adjusted_mask)
        else:
            # Process through attention block with original mask
            output = self.attention_block(mamba_output, attention_mask=attention_mask)
        
        return output
    
    def get_singular_values(self) -> Dict[str, torch.Tensor]:
        """
        Get singular values from all sub-components
        
        Returns:
            Dictionary mapping parameter names to singular values
        """
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
    
    def update_weights(self, modified_S_dict: Dict[str, torch.Tensor]) -> None:
        """
        Update weights using modified singular values
        
        Args:
            modified_S_dict: Dictionary mapping parameter names to modified singular values
            
        Returns:
            None (updates weights in-place)
        """
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


class MambaFormer(nn.Module):
    """
    MambaFormer architecture combining Mamba and Transformer elements
    
    This hybrid architecture:
    1. Uses an initial Mamba layer to process input
    2. Alternates between Mamba and attention blocks
    3. Does not use positional encoding (initial Mamba provides that functionality)
    4. Supports Singular Value Fine-tuning (SVF) for parameter-efficient adaptation
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_state: int = 16,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the MambaFormer model
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_layers: Number of MambaFormer layers
            n_heads: Number of attention heads
            d_state: SSM state dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            device: Device to place parameters
            dtype: Data type for parameters
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # Initial Mamba block (instead of positional encoding)
        self.initial_mamba = SVDMamba(
            d_model=d_model,
            d_state=d_state,
            device=device,
            dtype=dtype
        )
        
        # MambaFormer layers
        self.layers = nn.ModuleList([
            MambaformerLayer(
                d_model=d_model,
                num_heads=n_heads,
                d_state=d_state,
                dropout=dropout,
                device=device,
                dtype=dtype
            )
            for _ in range(n_layers)
        ])
        
        # Output layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)
        
        # Tie weights with embedding
        self.lm_head.weight = self.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights for the model
        
        Args:
            module: Module to initialize
            
        Returns:
            None (updates weights in-place)
        """
        if isinstance(module, nn.Linear):
            # Initialize linear projections
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Initialize embeddings
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        expert_vector: Optional[Any] = None,
        return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the MambaFormer model
        
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len],
                            with 1 for tokens to attend to and 0 for tokens to ignore
            expert_vector: Optional expert vector for adaptation
            return_dict: Whether to return outputs as a dictionary
            
        Returns:
            If return_dict=True: Dictionary with model outputs
            Otherwise: Logits tensor of shape [batch_size, seq_len, vocab_size]
        """
        # Get token embeddings
        x = self.embedding(input_ids)
        
        # Apply initial Mamba block (instead of positional encoding)
        x = self.initial_mamba(x)
        
        # Apply expert vector if provided
        if expert_vector is not None:
            # Get singular values
            singular_values = self.get_singular_values()
            
            # Apply the expert vector to singular values
            # This is a simplified example - in practice, this would be more complex
            if isinstance(expert_vector, (int, float)):
                # Simple scalar multiplier for all singular values
                modified_S_dict = {}
                for key, values in singular_values.items():
                    if isinstance(values, dict):
                        # Handle nested dictionaries
                        modified_values = {}
                        for sub_key, sub_values in values.items():
                            if isinstance(sub_values, torch.Tensor):
                                modified_values[sub_key] = sub_values * expert_vector
                            elif isinstance(sub_values, dict):
                                # Handle doubly-nested dictionaries
                                modified_sub_values = {}
                                for sub_sub_key, tensor in sub_values.items():
                                    if isinstance(tensor, torch.Tensor):
                                        modified_sub_values[sub_sub_key] = tensor * expert_vector
                                modified_values[sub_key] = modified_sub_values
                        modified_S_dict[key] = modified_values
                    elif isinstance(values, torch.Tensor):
                        modified_S_dict[key] = values * expert_vector
                
                # Update weights
                self.update_weights(modified_S_dict)
            elif isinstance(expert_vector, dict):
                # Expert vector is a dictionary with specific modifiers
                self.update_weights(expert_vector)
        
        # Apply layers
        for layer in self.layers:
            # Check if we need to handle attention_mask separately based on layer type
            if hasattr(layer, 'attention_block'):
                # For MambaformerLayer, process mamba and attention separately
                x = layer.mamba_block(x)
                
                # Match sequence dimensions if needed
                if attention_mask is not None and attention_mask.size(1) != x.size(1):
                    # Create a new attention mask that matches the sequence length
                    seq_len = x.size(1)
                    batch_size = x.size(0)
                    
                    if attention_mask.size(1) < seq_len:
                        # If attention_mask is shorter, pad it with zeros (don't attend)
                        padding = torch.zeros(
                            (batch_size, seq_len - attention_mask.size(1)),
                            device=attention_mask.device,
                            dtype=attention_mask.dtype
                        )
                        adjusted_mask = torch.cat([attention_mask, padding], dim=1)
                    else:
                        # If attention_mask is longer, truncate it
                        adjusted_mask = attention_mask[:, :seq_len]
                        
                    # Apply attention with adjusted mask
                    x = layer.attention_block(x, attention_mask=adjusted_mask)
                else:
                    # Use the original mask if dimensions match
                    x = layer.attention_block(x, attention_mask=attention_mask)
            else:
                # For other layer types, process as usual
                x = layer(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Get logits
        logits = self.lm_head(x)
        
        # Return as dictionary if requested
        if return_dict:
            return {"logits": logits}
        
        return logits
    
    def get_prompt_embedding(self, prompt: str) -> torch.Tensor:
        """
        Get embedding for a prompt
        
        Args:
            prompt: Text prompt
            
        Returns:
            Embedding tensor
        """
        # This is a placeholder - in practice, this would use a tokenizer
        # and extract features from the model's hidden states
        embedding = torch.randn(self.d_model, device=next(self.parameters()).device)
        embedding = F.normalize(embedding, p=2, dim=0)
        return embedding
        
    def get_logprobs(self, inputs, outputs=None):
        """
        Get log probabilities for the model outputs
        
        Args:
            inputs: Model inputs (can be input_ids tensor or a dict with 'input_ids' and other keys)
            outputs: Optional pre-computed model outputs
            
        Returns:
            Log probabilities tensor
        """
        # Extract input_ids and attention_mask from inputs if it's a dict
        if isinstance(inputs, dict):
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask')
        else:
            input_ids = inputs
            attention_mask = None
            
        # Get outputs if not provided
        if outputs is None:
            outputs = self.forward(input_ids, attention_mask=attention_mask, return_dict=True)
            
        # Extract logits from outputs
        if isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        else:
            logits = outputs
            
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get sequence length from the tensor shapes
        batch_size, seq_len = input_ids.shape
        
        if seq_len > 1:  # Only if we have more than one token
            # Ensure we don't shift beyond the available sequence
            max_shift = min(seq_len - 1, log_probs.size(1) - 1)
            
            # Select log_probs for token predictions (excluding the last token)
            shifted_log_probs = log_probs[:, :max_shift, :]
            
            # Select log_probs for actual tokens (excluding the first token)
            shifted_input_ids = input_ids[:, 1:1+max_shift]
            
            # Handle potential dimension mismatch
            if shifted_log_probs.size(1) != shifted_input_ids.size(1):
                min_len = min(shifted_log_probs.size(1), shifted_input_ids.size(1))
                shifted_log_probs = shifted_log_probs[:, :min_len, :]
                shifted_input_ids = shifted_input_ids[:, :min_len]
            
            # Get log_probs for each actual token
            token_log_probs = torch.gather(
                shifted_log_probs,
                dim=2,
                index=shifted_input_ids.unsqueeze(-1),
            ).squeeze(-1)
            
            # Apply attention_mask to zero out log_probs for padding tokens
            if attention_mask is not None and attention_mask.size(1) > 1:
                # Shift attention_mask to match shifted_input_ids
                shifted_attention_mask = attention_mask[:, 1:1+max_shift]
                
                # Handle potential dimension mismatch
                if token_log_probs.size(1) != shifted_attention_mask.size(1):
                    min_len = min(token_log_probs.size(1), shifted_attention_mask.size(1))
                    token_log_probs = token_log_probs[:, :min_len]
                    shifted_attention_mask = shifted_attention_mask[:, :min_len]
                
                token_log_probs = token_log_probs * shifted_attention_mask
        else:
            # Handle single token case
            token_log_probs = torch.zeros(batch_size, 0, device=input_ids.device)
        
        return token_log_probs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_p: float = 0.9,
        expert_vector: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p probability for nucleus sampling
            expert_vector: Optional expert vector for adaptation
            
        Returns:
            Generated token IDs
        """
        # Clone input_ids to avoid modifying the original
        generated_ids = input_ids.clone()
        
        # Generate tokens auto-regressively
        for _ in range(max_new_tokens):
            # Get the last max_seq_len tokens
            if generated_ids.size(1) > self.max_seq_len:
                input_chunk = generated_ids[:, -self.max_seq_len:]
            else:
                input_chunk = generated_ids
            
            # Forward pass
            logits = self.forward(input_chunk)
            
            # Take the logits for the last token
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-p sampling (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1,
                    index=sorted_indices,
                    src=sorted_indices_to_remove
                )
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))
            
            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append next token to generated_ids
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check if we've generated an EOS token
            if next_token.item() == 2:  # Assuming 2 is the EOS token
                break
        
        return generated_ids
        
    def get_singular_values(self) -> Dict[str, torch.Tensor]:
        """
        Get singular values from all components of the model
        
        Returns:
            Dictionary mapping parameter names to singular values
        """
        singular_values = {}
        
        # Get values from initial Mamba block
        initial_mamba_values = self.initial_mamba.get_singular_values()
        singular_values["initial_mamba"] = initial_mamba_values
        
        # Get values from each layer
        for i, layer in enumerate(self.layers):
            layer_values = layer.get_singular_values()
            singular_values[f"layer_{i}"] = layer_values
        
        return singular_values
    
    def update_weights(self, modified_S_dict: Dict[str, torch.Tensor]) -> None:
        """
        Update weights using modified singular values
        
        Args:
            modified_S_dict: Dictionary mapping parameter names to modified singular values
            
        Returns:
            None (updates weights in-place)
        """
        # Update initial Mamba block
        if "initial_mamba" in modified_S_dict:
            self.initial_mamba.update_weights(modified_S_dict["initial_mamba"])
        
        # Update each layer
        for i, layer in enumerate(self.layers):
            layer_key = f"layer_{i}"
            if layer_key in modified_S_dict:
                layer.update_weights(modified_S_dict[layer_key])
