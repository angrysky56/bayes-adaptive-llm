"""
Singular Value Fine-tuning (SVF) Implementation

This module implements the SVF technique which allows for parameter-efficient
fine-tuning by modifying only the singular values of weight matrices.

References:
- Transformer²: Self-adaptive LLMs (https://arxiv.org/abs/2501.06252)
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Changed from relative to absolute import
from utils.svd import apply_svd, update_with_svf

def apply_expert_vector(model: nn.Module, expert_vector: Dict[str, Any], scale: float = 1.0) -> None:
    """
    Apply an expert vector to a model's weights directly
    
    This function is a convenient wrapper around SVF functionality for direct application
    of expert vectors to a model without instantiating an SVF object.
    
    Args:
        model: Model to apply the expert vector to
        expert_vector: Dictionary mapping layer names to singular value scale factors
                     (either tensors or nested dictionaries)
        scale: Optional scaling factor to apply to the expert vector
        
    Returns:
        None: The model is modified in-place
    """
    # Get the device from the model
    device = next(model.parameters()).device
    
    for name, module in model.named_modules():
        if name in expert_vector and hasattr(module, 'get_singular_values'):
            # Get the expert vector for this layer
            layer_expert = expert_vector[name]
            
            if isinstance(layer_expert, dict) and hasattr(module, 'get_singular_values'):
                # For layers with multiple parameters (like SVDMamba)
                modified_S_dict = {}
                singular_values_dict = module.get_singular_values()
                
                for param_name, singular_values in singular_values_dict.items():
                    if param_name in layer_expert:
                        # Move expert vector to the same device as singular values
                        layer_expert_tensor = layer_expert[param_name].to(device)
                        modified_S_dict[param_name] = singular_values * (layer_expert_tensor * scale)
                
                if modified_S_dict:
                    module.update_weights(modified_S_dict)
            elif hasattr(module, 'get_singular_values'):
                # For layers with a single weight matrix (like SVDLinear)
                singular_values = module.get_singular_values()
                
                if not isinstance(singular_values, dict):
                    # Move expert vector to the same device as singular values
                    if isinstance(layer_expert, torch.Tensor):
                        layer_expert = layer_expert.to(device)
                    
                    # Apply expert vector to singular values
                    modified_S = singular_values * (layer_expert * scale)
                    
                    # Update weight matrix
                    module.update_weights(modified_S)

class SVF:
    """
    Main interface for Singular Value Fine-tuning (SVF).
    
    This class provides a simple interface for applying SVF to a model,
    managing expert vectors, and performing other related operations.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        rank: Optional[int] = None,
        layers: Optional[List[str]] = None
    ):
        """
        Initialize SVF.
        
        Args:
            model: Model to apply SVF to
            rank: Rank of SVF adaptation (if None, use full rank)
            layers: List of layer names to apply SVF to (if None, apply to all compatible layers)
        """
        self.model = model
        self.rank = rank
        self.layer_names = layers
        
        # Initialize SVF fine-tuner
        self.finetuner = SVFFinetuner(model)
        
        # Create default expert
        self.current_expert = None
    
    def create_expert(self, name: str) -> Dict[str, Any]:
        """
        Create a new expert vector.
        
        Args:
            name: Name of the expert
            
        Returns:
            Dictionary of expert adapters
        """
        return self.finetuner.create_expert(name)
    
    def apply_expert_vector(self, expert_vector: Any) -> None:
        """
        Apply an expert vector to the model.
        
        Args:
            expert_vector: Expert vector to apply. Can be:
                - A dictionary mapping layer names to tensor modifiers
                - A dictionary with nested structure for complex models
                - A scalar value to multiply all singular values uniformly
                - A dictionary mapping expert names to weights for blending
        """
        # Handle scalar inputs (simple multiplier for all singular values)
        if isinstance(expert_vector, (int, float)):
            # Create a uniform scaling for all layers
            uniform_expert = {}
            for name, layer in self.finetuner.svf_layers:
                singular_values = layer.get_singular_values()
                if isinstance(singular_values, dict):
                    # For nested structures
                    uniform_expert[name] = {
                        param_name: torch.ones_like(sv) * expert_vector
                        for param_name, sv in singular_values.items()
                    }
                else:
                    # For simple tensors
                    uniform_expert[name] = torch.ones_like(singular_values) * expert_vector
            
            # Apply the uniform expert
            self.finetuner.apply_to_model(expert_dict=uniform_expert)
            return
                
        # Convert expert vector format if needed
        if isinstance(expert_vector, dict):
            # Check if it's a layer-to-modifier mapping or expert-to-weight mapping
            has_tensor_values = any(isinstance(v, torch.Tensor) for v in expert_vector.values())
            has_dict_values = any(isinstance(v, dict) for v in expert_vector.values())
            
            if has_tensor_values or has_dict_values:
                # Expert vector is in the right format, apply it directly
                self.finetuner.apply_to_model(expert_dict=expert_vector)
            else:
                # Assume expert_vector is a dictionary mapping expert names to weights
                self.finetuner.apply_to_model(expert_weights=expert_vector)
    
    def parameters(self) -> List[nn.Parameter]:
        """
        Get all trainable parameters of the currently selected expert.
        
        Returns:
            List of trainable parameters
        """
        if self.current_expert is None:
            raise ValueError("No expert selected. Call create_expert() first.")
        
        return self.finetuner.get_trainable_parameters(self.current_expert)
    
    def save_expert_vector(self, path: str, expert_name: Optional[str] = None) -> None:
        """
        Save expert vector to disk.
        
        Args:
            path: Path to save the expert vector
            expert_name: Name of the expert to save (if None, use current expert)
        """
        if expert_name is None:
            if self.current_expert is None:
                raise ValueError("No expert selected. Provide expert_name or call create_expert() first.")
            expert_name = self.current_expert
        
        self.finetuner.save_expert(expert_name, path)
    
    def load_expert_vector(self, path: str, expert_name: Optional[str] = None) -> None:
        """
        Load expert vector from disk.
        
        Args:
            path: Path to the saved expert vector
            expert_name: Name to assign to the loaded expert (if None, use current expert)
        """
        if expert_name is None:
            if self.current_expert is None:
                # Generate a default name
                expert_name = f"expert_{len(self.finetuner.adapters)}"
            else:
                expert_name = self.current_expert
                
        self.finetuner.load_expert(expert_name, path)
        self.current_expert = expert_name
    
    def blend_experts(self, expert_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """
        Create a blended expert by combining multiple experts.
        
        Args:
            expert_weights: Dictionary mapping expert names to their weights
            
        Returns:
            Dictionary mapping layer names to blended expert vectors
        """
        return self.finetuner.blend_experts(expert_weights)


class SVFAdapter(nn.Module):
    """
    Adapter for Singular Value Fine-tuning (SVF)
    
    This adapter modifies the singular values of a weight matrix
    while keeping the singular vectors fixed.
    """
    
    def __init__(self, singular_values: torch.Tensor):
        """
        Initialize an SVF adapter
        
        Args:
            singular_values: Original singular values to be modified
        """
        super().__init__()
        
        if not isinstance(singular_values, torch.Tensor):
            raise ValueError(f"Expected tensor, got {type(singular_values)}")
            
        # Store the original singular values shape
        self.shape = singular_values.shape
        
        # Initialize the expert vector to ones (identity transformation)
        self.expert_vector = nn.Parameter(torch.ones_like(singular_values))
        
    def forward(self, singular_values: torch.Tensor) -> torch.Tensor:
        """
        Apply the expert vector to modify singular values
        
        Args:
            singular_values: Original singular values
            
        Returns:
            Modified singular values
        """
        return singular_values * self.expert_vector
    

class SVFFinetuner:
    """
    Manager for Singular Value Fine-tuning
    
    This class handles the SVF process for an entire model.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize an SVF fine-tuner
        
        Args:
            model: The model to fine-tune
        """
        self.model = model
        self.adapters = {}
        
        # Get all layers that support SVF (have get_singular_values method)
        self.svf_layers = []
        for name, module in model.named_modules():
            if hasattr(module, 'get_singular_values'):
                self.svf_layers.append((name, module))
    
    def create_expert(self, expert_name: str) -> Dict[str, SVFAdapter]:
        """
        Create a new expert by initializing adapters for all SVF-supported layers
        
        Args:
            expert_name: Name of the expert
            
        Returns:
            Dictionary mapping layer names to their SVF adapters
        """
        expert_adapters = {}
        
        for name, layer in self.svf_layers:
            # Get original singular values
            singular_values_dict = layer.get_singular_values()
            
            # Handle different layer types
            if isinstance(singular_values_dict, dict):
                # For layers like SVDMamba that return dictionaries of singular values
                layer_adapters = {}
                for param_name, sing_values in singular_values_dict.items():
                    # Ensure sing_values is a tensor before creating adapter
                    if isinstance(sing_values, torch.Tensor):
                        adapter = SVFAdapter(sing_values)
                        layer_adapters[param_name] = adapter
                
                if layer_adapters:
                    expert_adapters[name] = layer_adapters
            else:
                # For layers like SVDLinear that return tensors directly
                # Ensure singular_values_dict is a tensor
                if isinstance(singular_values_dict, torch.Tensor):
                    adapter = SVFAdapter(singular_values_dict)
                    expert_adapters[name] = adapter
        
        # Store expert in manager
        self.adapters[expert_name] = expert_adapters
        
        return expert_adapters
    
    def apply_expert(
        self, 
        expert_name: str, 
        layer_name: str, 
        singular_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        param_name: Optional[str] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply an expert's adapter to a layer's singular values
        
        Args:
            expert_name: Name of the expert
            layer_name: Name of the layer
            singular_values: Original singular values (tensor or dict)
            param_name: Optional parameter name for nested adapters
            
        Returns:
            Modified singular values (tensor or dict)
        """
        if expert_name not in self.adapters:
            raise ValueError(f"Expert '{expert_name}' not found")
        
        if layer_name not in self.adapters[expert_name]:
            raise ValueError(f"Layer '{layer_name}' not found in expert '{expert_name}'")
        
        # Get adapter
        layer_adapter = self.adapters[expert_name][layer_name]
        
        # Handle nested adapters (for layers like SVDMamba)
        if isinstance(layer_adapter, dict) and isinstance(singular_values, dict):
            result = {}
            for key, value in singular_values.items():
                if key in layer_adapter:
                    result[key] = layer_adapter[key](value)
                else:
                    result[key] = value
            return result
        elif isinstance(layer_adapter, dict) and param_name is not None:
            if param_name in layer_adapter:
                return layer_adapter[param_name](singular_values)
            return singular_values
        else:
            # Apply adapter directly
            return layer_adapter(singular_values)
    
    def get_trainable_parameters(self, expert_name: str) -> List[nn.Parameter]:
        """
        Get all trainable parameters of an expert
        
        Args:
            expert_name: Name of the expert
            
        Returns:
            List of trainable parameters
        """
        if expert_name not in self.adapters:
            raise ValueError(f"Expert '{expert_name}' not found")
        
        params = []
        for adapter_or_dict in self.adapters[expert_name].values():
            if isinstance(adapter_or_dict, dict):
                # For nested adapters (like SVDMamba)
                for adapter in adapter_or_dict.values():
                    params.append(adapter.expert_vector)
            else:
                # For direct adapters (like SVDLinear)
                params.append(adapter_or_dict.expert_vector)
        
        return params
    
    def blend_experts(
        self,
        expert_weights: Dict[str, float]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Create a blended expert by combining multiple experts
        
        Args:
            expert_weights: Dictionary mapping expert names to their weights
            
        Returns:
            Dictionary mapping layer names to blended expert vectors (or nested dictionaries)
        """
        if not expert_weights:
            return {}
        
        # Normalize weights to sum to 1
        total_weight = sum(expert_weights.values())
        normalized_weights = {
            name: weight / total_weight 
            for name, weight in expert_weights.items()
        }
        
        blended_expert = {}
        
        # Get all expert names that exist in the manager
        valid_experts = [name for name in normalized_weights if name in self.adapters]
        
        # If no valid experts, return empty dict
        if not valid_experts:
            return {}
        
        # Get all layer names from the first valid expert
        first_expert = valid_experts[0]
        for layer_name, adapter_or_dict in self.adapters[first_expert].items():
            if isinstance(adapter_or_dict, dict):
                # For nested adapters (like SVDMamba)
                blended_param_dict = {}
                
                # Initialize with parameter names from the first expert
                for param_name, adapter in adapter_or_dict.items():
                    # Get device from first expert
                    device = adapter.expert_vector.device
                    
                    # Initialize blended vector
                    blended_vector = torch.zeros_like(adapter.expert_vector)
                    
                    # Blend expert vectors for this parameter
                    for expert_name, weight in normalized_weights.items():
                        if (expert_name in self.adapters and 
                            layer_name in self.adapters[expert_name] and
                            isinstance(self.adapters[expert_name][layer_name], dict) and
                            param_name in self.adapters[expert_name][layer_name]):
                            
                            blended_vector += weight * self.adapters[expert_name][layer_name][param_name].expert_vector
                    
                    blended_param_dict[param_name] = blended_vector
                
                if blended_param_dict:
                    blended_expert[layer_name] = blended_param_dict
            else:
                # For direct adapters (like SVDLinear)
                # Initialize with zeros
                original_shape = adapter_or_dict.shape
                device = adapter_or_dict.expert_vector.device
                blended_vector = torch.zeros(original_shape, device=device)
                
                # Blend expert vectors
                for expert_name, weight in normalized_weights.items():
                    if (expert_name in self.adapters and 
                        layer_name in self.adapters[expert_name] and
                        not isinstance(self.adapters[expert_name][layer_name], dict)):
                        
                        blended_vector += weight * self.adapters[expert_name][layer_name].expert_vector
                
                blended_expert[layer_name] = blended_vector
        
        return blended_expert
    
    def apply_to_model(
        self,
        expert_weights: Optional[Dict[str, float]] = None,
        expert_name: Optional[str] = None,
        expert_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Apply expert adaptations to the model
        
        One of expert_weights, expert_name, or expert_dict must be provided.
        
        Args:
            expert_weights: Dictionary mapping expert names to their weights
            expert_name: Name of a single expert to apply
            expert_dict: Direct dictionary of expert vectors to apply to model layers
            
        Raises:
            ValueError: If parameters are not provided correctly
        """
        # Check parameters
        provided_params = sum(p is not None for p in [expert_weights, expert_name, expert_dict])
        if provided_params != 1:
            raise ValueError("Exactly one of expert_weights, expert_name, or expert_dict must be provided")
        
        # Get expert vectors to apply
        if expert_dict is not None:
            # Use expert_dict directly
            pass
        elif expert_name is not None:
            if expert_name not in self.adapters:
                raise ValueError(f"Expert '{expert_name}' not found")
            
            # Prepare expert_dict structure depending on the layer type
            expert_dict = {}
            for layer_name, adapter_or_dict in self.adapters[expert_name].items():
                if isinstance(adapter_or_dict, dict):
                    # For nested adapters (like SVDMamba)
                    expert_dict[layer_name] = {
                        param_name: adapter.expert_vector
                        for param_name, adapter in adapter_or_dict.items()
                    }
                else:
                    # For direct adapters (like SVDLinear)
                    expert_dict[layer_name] = adapter_or_dict.expert_vector
        else:
            expert_dict = self.blend_experts(expert_weights)
        
        # Apply expert vectors to model
        for name, layer in self.svf_layers:
            if name in expert_dict:
                expert_vector = expert_dict[name]
                
                if isinstance(expert_vector, dict):
                    # For layers with multiple parameters (like SVDMamba)
                    modified_S_dict = {}
                    singular_values_dict = layer.get_singular_values()
                    
                    for param_name, singular_values in singular_values_dict.items():
                        if param_name in expert_vector:
                            # Move expert vector to same device as singular values
                            expert_value = expert_vector[param_name]
                            if isinstance(expert_value, torch.Tensor):
                                expert_value = expert_value.to(singular_values.device)
                            modified_S_dict[param_name] = singular_values * expert_value
                    
                    if modified_S_dict:
                        layer.update_weights(modified_S_dict)
                else:
                    # For layers with a single weight matrix (like SVDLinear)
                    singular_values = layer.get_singular_values()
                    
                    if not isinstance(singular_values, dict):
                        # Apply expert vector to singular values
                        # Move expert vector to same device as singular values
                        if isinstance(expert_vector, torch.Tensor):
                            expert_vector = expert_vector.to(singular_values.device)
                        modified_S = singular_values * expert_vector
                        
                        # Update weight matrix
                        layer.update_weights(modified_S)
    
    def save_expert(self, expert_name: str, path: str) -> None:
        """
        Save an expert to disk
        
        Args:
            expert_name: Name of the expert
            path: Path to save the expert
            
        Raises:
            ValueError: If the expert does not exist
        """
        if expert_name not in self.adapters:
            raise ValueError(f"Expert '{expert_name}' not found")
        
        # Collect expert state dict
        expert_state = {}
        for layer_name, adapter_or_dict in self.adapters[expert_name].items():
            if isinstance(adapter_or_dict, dict):
                # For nested adapters (like SVDMamba)
                param_dict = {}
                for param_name, adapter in adapter_or_dict.items():
                    param_dict[param_name] = adapter.expert_vector.data.cpu()
                expert_state[layer_name] = param_dict
            else:
                # For direct adapters (like SVDLinear)
                expert_state[layer_name] = adapter_or_dict.expert_vector.data.cpu()
        
        torch.save(expert_state, path)
    
    def load_expert(self, expert_name: str, path: str) -> None:
        """
        Load an expert from disk
        
        Args:
            expert_name: Name to assign to the loaded expert
            path: Path to the saved expert
            
        Raises:
            FileNotFoundError: If the expert file does not exist
        """
        expert_state = torch.load(path)
        
        # Create a new expert
        expert_adapters = self.create_expert(expert_name)
        
        # Load expert state
        for layer_name, vector_or_dict in expert_state.items():
            if layer_name in expert_adapters:
                if isinstance(vector_or_dict, dict) and isinstance(expert_adapters[layer_name], dict):
                    # For nested adapters (like SVDMamba)
                    for param_name, vector in vector_or_dict.items():
                        if param_name in expert_adapters[layer_name]:
                            expert_adapters[layer_name][param_name].expert_vector.data.copy_(vector)
                else:
                    # For direct adapters (like SVDLinear)
                    expert_adapters[layer_name].expert_vector.data.copy_(vector_or_dict)
