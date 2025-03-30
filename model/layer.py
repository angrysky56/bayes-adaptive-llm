"""
SVD-Capable Layers

This module provides layer implementations with SVD capabilities
for use with Singular Value Fine-tuning (SVF).
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Changed from relative to absolute import
from utils.svd import apply_svd, update_with_svf


class SVDLinear(nn.Linear):
    """
    Linear layer with SVD capabilities for SVF
    
    This layer extends the standard PyTorch Linear layer with methods
    to support Singular Value Fine-tuning.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None
    ):
        """
        Initialize a SVD-capable linear layer
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to use a bias term
            device: Device to use
            dtype: Data type to use
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        # Cache for SVD components to avoid recomputation
        self._svd_cache = None
    
    def get_singular_values(self) -> torch.Tensor:
        """
        Get singular values of the weight matrix
        
        Returns:
            Singular values tensor
        """
        _, S, _ = self.get_svd_components()
        return S
    
    def get_svd_components(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get SVD components of the weight matrix
        
        Returns:
            Tuple of (U, S, V) where weight ≈ U @ diag(S) @ V.T
        """
        if self._svd_cache is None:
            U, S, V = apply_svd(self.weight)
            self._svd_cache = (U, S, V)
        
        return self._svd_cache
    
    def update_weights(self, modified_S: torch.Tensor) -> None:
        """
        Update weights using modified singular values
        
        Args:
            modified_S: Modified singular values
        """
        U, S, V = self.get_svd_components()
        self.weight.data = update_with_svf(self.weight, U, S, V, modified_S)
        
        # Invalidate cache
        self._svd_cache = None
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input: Input tensor
            
        Returns:
            Output tensor
        """
        return F.linear(input, self.weight, self.bias)


class SVDConv1d(nn.Conv1d):
    """
    1D convolutional layer with SVD capabilities for SVF
    
    This layer extends the standard PyTorch Conv1d layer with methods
    to support Singular Value Fine-tuning.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ):
        """
        Initialize a SVD-capable 1D convolutional layer
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernel
            stride: Stride of the convolution
            padding: Padding added to input
            dilation: Spacing between kernel elements
            groups: Number of blocked connections
            bias: Whether to use a bias term
            padding_mode: Padding mode
            device: Device to use
            dtype: Data type to use
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )
        
        # Cache for SVD components to avoid recomputation
        self._svd_cache = None
    
    def get_singular_values(self) -> torch.Tensor:
        """
        Get singular values of the weight matrix
        
        Returns:
            Singular values tensor
        """
        _, S, _ = self.get_svd_components()
        return S
    
    def get_svd_components(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get SVD components of the weight matrix
        
        Returns:
            Tuple of (U, S, V) where weight ≈ U @ diag(S) @ V.T
        """
        if self._svd_cache is None:
            # Reshape to 2D for SVD
            weight_2d = self.weight.reshape(self.out_channels, -1)
            U, S, V = apply_svd(weight_2d)
            self._svd_cache = (U, S, V)
        
        return self._svd_cache
    
    def update_weights(self, modified_S: torch.Tensor) -> None:
        """
        Update weights using modified singular values
        
        Args:
            modified_S: Modified singular values
        """
        U, S, V = self.get_svd_components()
        
        # Reshape to 2D for update
        weight_2d = self.weight.reshape(self.out_channels, -1)
        updated_weight_2d = update_with_svf(weight_2d, U, S, V, modified_S)
        
        # Reshape back to original shape
        self.weight.data = updated_weight_2d.reshape(self.weight.shape)
        
        # Invalidate cache
        self._svd_cache = None
