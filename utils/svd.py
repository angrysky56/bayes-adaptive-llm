"""
SVD Utility Functions

This module provides utility functions for Singular Value Decomposition (SVD)
operations used in Singular Value Fine-tuning (SVF).
"""

import torch
import torch.nn as nn


def apply_svd(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply SVD to a weight matrix
    
    Args:
        weight: Weight matrix
        
    Returns:
        Tuple of (U, S, V) where weight ≈ U @ diag(S) @ V.T
    """
    # Make sure we're working with a tensor that's on the right device
    device = weight.device
    
    # Handle different dimensionality of weight matrices
    orig_shape = weight.shape
    if len(orig_shape) > 2:
        # Reshape to 2D for SVD
        weight = weight.reshape(orig_shape[0], -1)
    
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
    
    return U, S, V


def update_with_svf(
    weight: torch.Tensor, 
    U: torch.Tensor, 
    S: torch.Tensor, 
    V: torch.Tensor,
    modified_S: torch.Tensor
) -> torch.Tensor:
    """
    Update a weight matrix using modified singular values
    
    Args:
        weight: Original weight matrix
        U: Left singular vectors
        S: Original singular values
        V: Right singular vectors
        modified_S: Modified singular values
        
    Returns:
        Updated weight matrix
    """
    # Make sure everything is on the same device
    device = weight.device
    U = U.to(device)
    V = V.to(device)
    modified_S = modified_S.to(device)
    
    # Ensure the singular values have the right shape
    if modified_S.ndim == 1:
        modified_S = torch.diag(modified_S)
    
    # Reconstruct the weight matrix
    orig_shape = weight.shape
    updated_weight = U @ modified_S @ V.transpose(-2, -1)
    
    # Restore original shape if needed
    if len(orig_shape) > 2:
        updated_weight = updated_weight.reshape(orig_shape)
        
    return updated_weight
