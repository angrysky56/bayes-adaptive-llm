"""
Expert vectors for Singular Value Fine-tuning (SVF).

This module provides functionality for creating, loading, and managing expert vectors
for use with SVF.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any


class ExpertVector:
    """
    Expert vector for use with SVF.
    
    This class provides a container for expert vectors and methods for
    loading, saving, and manipulating them.
    """
    
    def __init__(
        self,
        name: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an expert vector.
        
        Args:
            name: Name of the expert
            data: Optional data dictionary
        """
        self.name = name
        self.data = data or {}
        
    def load_from_file(self, path: str) -> None:
        """
        Load expert vector from a file.
        
        Args:
            path: Path to the file
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert vector file {path} not found")
        
        try:
            self.data = torch.load(path)
        except Exception as e:
            raise ValueError(f"Error loading expert vector from {path}: {e}")
        
    def save_to_file(self, path: str) -> None:
        """
        Save expert vector to a file.
        
        Args:
            path: Path to save the file
        """
        if not self.data:
            raise ValueError("Expert vector is empty")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            torch.save(self.data, path)
        except Exception as e:
            raise ValueError(f"Error saving expert vector to {path}: {e}")
        
    def __getitem__(self, key: str) -> Any:
        """
        Get item from expert vector.
        
        Args:
            key: Key to look up
            
        Returns:
            Value associated with the key
        """
        return self.data[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set item in expert vector.
        
        Args:
            key: Key to set
            value: Value to set
        """
        self.data[key] = value
        
    def __contains__(self, key: str) -> bool:
        """
        Check if key is in expert vector.
        
        Args:
            key: Key to check
            
        Returns:
            True if key is in expert vector, False otherwise
        """
        return key in self.data
    
    def keys(self) -> List[str]:
        """
        Get keys in expert vector.
        
        Returns:
            List of keys
        """
        return list(self.data.keys())
    
    def values(self) -> List[Any]:
        """
        Get values in expert vector.
        
        Returns:
            List of values
        """
        return list(self.data.values())
    
    def items(self) -> List[Tuple[str, Any]]:
        """
        Get items in expert vector.
        
        Returns:
            List of (key, value) tuples
        """
        return list(self.data.items())
    
    def __repr__(self) -> str:
        """
        Get string representation of expert vector.
        
        Returns:
            String representation
        """
        return f"ExpertVector(name={self.name}, keys={list(self.data.keys())})"


class ExpertManager:
    """
    Manager for expert vectors.
    
    This class provides methods for loading, saving, and managing
    multiple expert vectors.
    """
    
    def __init__(self, expert_dir: str):
        """
        Initialize an expert manager.
        
        Args:
            expert_dir: Directory containing expert vectors
        """
        self.expert_dir = expert_dir
        self.experts = {}
        
        # Create directory if it doesn't exist
        os.makedirs(expert_dir, exist_ok=True)
        
    def load_experts(self) -> None:
        """
        Load all expert vectors from the expert directory.
        """
        # Get all .pt files in the expert directory
        expert_files = [f for f in os.listdir(self.expert_dir) if f.endswith('.pt')]
        
        # Load each expert vector
        for file_name in expert_files:
            expert_name = os.path.splitext(file_name)[0]
            expert_path = os.path.join(self.expert_dir, file_name)
            
            try:
                expert = ExpertVector(expert_name)
                expert.load_from_file(expert_path)
                self.experts[expert_name] = expert
                print(f"Loaded expert vector: {expert_name}")
            except Exception as e:
                print(f"Error loading expert vector {expert_name}: {e}")
                
    def save_expert(self, expert: ExpertVector) -> None:
        """
        Save an expert vector to the expert directory.
        
        Args:
            expert: Expert vector to save
        """
        expert_path = os.path.join(self.expert_dir, f"{expert.name}.pt")
        expert.save_to_file(expert_path)
        print(f"Saved expert vector: {expert.name}")
        
    def create_expert(self, name: str, data: Optional[Dict[str, Any]] = None) -> ExpertVector:
        """
        Create a new expert vector.
        
        Args:
            name: Name of the expert
            data: Optional data dictionary
            
        Returns:
            New expert vector
        """
        expert = ExpertVector(name, data)
        self.experts[name] = expert
        return expert
    
    def get_expert(self, name: str) -> ExpertVector:
        """
        Get an expert vector by name.
        
        Args:
            name: Name of the expert
            
        Returns:
            Expert vector
            
        Raises:
            KeyError: If expert vector does not exist
        """
        if name not in self.experts:
            raise KeyError(f"Expert vector {name} not found")
        
        return self.experts[name]
    
    def delete_expert(self, name: str) -> None:
        """
        Delete an expert vector.
        
        Args:
            name: Name of the expert
            
        Raises:
            KeyError: If expert vector does not exist
        """
        if name not in self.experts:
            raise KeyError(f"Expert vector {name} not found")
        
        # Delete from memory
        del self.experts[name]
        
        # Delete file
        expert_path = os.path.join(self.expert_dir, f"{name}.pt")
        if os.path.exists(expert_path):
            os.remove(expert_path)
            print(f"Deleted expert vector: {name}")
            
    def list_experts(self) -> List[str]:
        """
        List all expert vectors.
        
        Returns:
            List of expert vector names
        """
        return list(self.experts.keys())
    
    def blend_experts(
        self,
        weights: Dict[str, float]
    ) -> ExpertVector:
        """
        Create a blended expert by combining multiple experts.
        
        Args:
            weights: Dictionary mapping expert names to weights
            
        Returns:
            Blended expert vector
            
        Raises:
            ValueError: If no experts specified or weight sum is zero
        """
        if not weights:
            raise ValueError("No experts specified for blending")
        
        # Filter out experts that don't exist
        valid_weights = {
            name: weight for name, weight in weights.items()
            if name in self.experts
        }
        
        if not valid_weights:
            raise ValueError("No valid experts specified for blending")
        
        # Normalize weights
        total_weight = sum(valid_weights.values())
        if total_weight == 0:
            raise ValueError("Sum of weights is zero")
        
        normalized_weights = {
            name: weight / total_weight
            for name, weight in valid_weights.items()
        }
        
        # Create blended expert data
        blended_data = {}
        
        # Get the first expert to determine the structure
        first_expert = next(iter(valid_weights.keys()))
        for key, value in self.experts[first_expert].items():
            if isinstance(value, torch.Tensor):
                # Initialize with zeros
                blended_data[key] = torch.zeros_like(value)
                
                # Add weighted contributions
                for name, weight in normalized_weights.items():
                    if key in self.experts[name]:
                        blended_data[key] += weight * self.experts[name][key]
            elif isinstance(value, dict):
                # Handle nested dictionaries
                blended_data[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        blended_data[key][subkey] = torch.zeros_like(subvalue)
                        
                        # Add weighted contributions
                        for name, weight in normalized_weights.items():
                            if (key in self.experts[name] and 
                                subkey in self.experts[name][key]):
                                blended_data[key][subkey] += weight * self.experts[name][key][subkey]
        
        # Create blended expert
        blended_name = "+".join(
            f"{name}_{weight:.2f}"
            for name, weight in normalized_weights.items()
        )
        
        return ExpertVector(blended_name, blended_data)
