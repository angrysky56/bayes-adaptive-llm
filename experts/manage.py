"""
Expert Management

This module provides functions for managing SVF experts,
including loading, saving, and evaluating experts.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..svf.svf import SVFFinetuner


class ExpertManager:
    """
    Manager for SVF experts
    
    This class provides functionality for managing experts,
    including loading, saving, and metadata.
    """
    
    def __init__(
        self, 
        experts_dir: str,
        finetuner: SVFFinetuner
    ):
        """
        Initialize an expert manager
        
        Args:
            experts_dir: Directory to store experts
            finetuner: SVF finetuner managing the experts
        """
        self.experts_dir = experts_dir
        self.finetuner = finetuner
        
        # Create experts directory if it doesn't exist
        os.makedirs(experts_dir, exist_ok=True)
        
        # Dictionary to store expert metadata
        self.expert_metadata = {}
        
        # Load metadata for existing experts
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load metadata for all experts in the experts directory"""
        self.expert_metadata = {}
        
        for filename in os.listdir(self.experts_dir):
            if filename.endswith('.json'):
                expert_name = filename[:-5]  # Remove .json extension
                metadata_path = os.path.join(self.experts_dir, filename)
                
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    self.expert_metadata[expert_name] = metadata
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Could not load metadata for expert '{expert_name}': {e}")
    
    def save_expert(
        self, 
        expert_name: str, 
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save an expert to disk
        
        Args:
            expert_name: Name of the expert
            metadata: Optional metadata to store with the expert
            
        Raises:
            ValueError: If the expert does not exist
        """
        # Save expert weights
        expert_path = os.path.join(self.experts_dir, f"{expert_name}.pt")
        self.finetuner.save_expert(expert_name, expert_path)
        
        # Save metadata
        metadata_path = os.path.join(self.experts_dir, f"{expert_name}.json")
        expert_meta = metadata or {}
        
        with open(metadata_path, 'w') as f:
            json.dump(expert_meta, f, indent=2)
        
        # Update metadata cache
        self.expert_metadata[expert_name] = expert_meta
    
    def load_expert(
        self, 
        expert_name: str
    ) -> Dict:
        """
        Load an expert from disk
        
        Args:
            expert_name: Name of the expert
            
        Returns:
            Expert metadata
            
        Raises:
            FileNotFoundError: If the expert does not exist
        """
        expert_path = os.path.join(self.experts_dir, f"{expert_name}.pt")
        
        if not os.path.exists(expert_path):
            raise FileNotFoundError(f"Expert '{expert_name}' not found")
        
        # Load expert weights
        self.finetuner.load_expert(expert_name, expert_path)
        
        # Return metadata
        return self.get_metadata(expert_name)
    
    def get_metadata(self, expert_name: str) -> Dict:
        """
        Get metadata for an expert
        
        Args:
            expert_name: Name of the expert
            
        Returns:
            Expert metadata
            
        Raises:
            KeyError: If the expert does not exist
        """
        if expert_name not in self.expert_metadata:
            # Try to load metadata
            metadata_path = os.path.join(self.experts_dir, f"{expert_name}.json")
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    self.expert_metadata[expert_name] = metadata
                    return metadata
                except (json.JSONDecodeError, IOError) as e:
                    raise KeyError(f"Could not load metadata for expert '{expert_name}': {e}")
            
            raise KeyError(f"Expert '{expert_name}' not found")
        
        return self.expert_metadata[expert_name]
    
    def update_metadata(
        self, 
        expert_name: str, 
        metadata: Dict
    ) -> None:
        """
        Update metadata for an expert
        
        Args:
            expert_name: Name of the expert
            metadata: New metadata
            
        Raises:
            KeyError: If the expert does not exist
        """
        if expert_name not in self.expert_metadata:
            # Check if expert exists
            expert_path = os.path.join(self.experts_dir, f"{expert_name}.pt")
            
            if not os.path.exists(expert_path):
                raise KeyError(f"Expert '{expert_name}' not found")
        
        # Update metadata
        metadata_path = os.path.join(self.experts_dir, f"{expert_name}.json")
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update metadata cache
        self.expert_metadata[expert_name] = metadata
    
    def list_experts(self) -> List[str]:
        """
        List all available experts
        
        Returns:
            List of expert names
        """
        return list(self.expert_metadata.keys())
    
    def delete_expert(self, expert_name: str) -> None:
        """
        Delete an expert
        
        Args:
            expert_name: Name of the expert
            
        Raises:
            KeyError: If the expert does not exist
        """
        expert_path = os.path.join(self.experts_dir, f"{expert_name}.pt")
        metadata_path = os.path.join(self.experts_dir, f"{expert_name}.json")
        
        # Check if expert exists
        if not os.path.exists(expert_path):
            raise KeyError(f"Expert '{expert_name}' not found")
        
        # Delete expert files
        os.remove(expert_path)
        
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        # Remove from metadata cache
        self.expert_metadata.pop(expert_name, None)
