"""
Bayesian Controller for Expert Adaptation

This module implements a Bayesian controller for dispatching between
expert vectors based on input prompts. It maintains priors for each
expert, computes likelihoods based on input features, and updates
posteriors using Bayes' rule.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any


class BayesianController:
    """
    A Bayesian controller for expert adaptation that dynamically selects
    and blends expert vectors based on input prompts.
    
    This controller:
    1. Maintains priors for each expert (skill)
    2. Computes likelihoods based on input features
    3. Updates posteriors using Bayes' rule
    4. Blends expert vectors based on posteriors
    """
    
    def __init__(
        self,
        expert_names: List[str],
        initial_priors: Optional[Dict[str, float]] = None,
        min_probability: float = 1e-5,
        max_probability: float = 0.999,
        temperature: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize a Bayesian Controller
        
        Args:
            expert_names: List of expert names
            initial_priors: Optional dictionary mapping expert names to prior probabilities
            min_probability: Minimum probability to prevent zero probabilities
            max_probability: Maximum probability to prevent probabilities of 1.0
            temperature: Temperature for scaling likelihood values
            device: Device to use for tensor operations
        """
        self.expert_names = expert_names
        self.num_experts = len(expert_names)
        self.min_probability = min_probability
        self.max_probability = max_probability
        self.temperature = temperature
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize priors
        if initial_priors is None:
            # If no initial priors provided, use uniform distribution
            initial_priors = {name: 1.0 / self.num_experts for name in expert_names}
        
        # Ensure all expert names have a prior
        for name in expert_names:
            if name not in initial_priors:
                initial_priors[name] = self.min_probability
        
        # Normalize priors
        total = sum(initial_priors.values())
        self.priors = {name: initial_priors[name] / total for name in expert_names}
        
        # Initialize posteriors to match priors
        self.posteriors = self.priors.copy()
        
        # Cache for expert embeddings and representations
        self.expert_embeddings = {}
        
        # Track history for analysis
        self.history = []
        
    def update_priors(self, new_priors: Dict[str, float]) -> None:
        """
        Update the priors for each expert
        
        Args:
            new_priors: Dictionary mapping expert names to new prior probabilities
        """
        # Ensure all expert names have a prior
        for name in self.expert_names:
            if name not in new_priors:
                new_priors[name] = self.min_probability
                
        # Normalize priors
        total = sum(new_priors.values())
        self.priors = {name: new_priors[name] / total for name in self.expert_names}
        
        # Update posteriors to match new priors
        self.posteriors = self.priors.copy()
        
    def compute_likelihood(
        self, 
        input_features: torch.Tensor,
        expert_name: str,
        method: str = 'embedding_similarity'
    ) -> float:
        """
        Compute the likelihood of the input features given the expert
        
        Args:
            input_features: Input features tensor
            expert_name: Name of the expert
            method: Method for computing likelihood ('embedding_similarity', 'classifier', etc.)
            
        Returns:
            Likelihood value
        """
        # Check if expert embedding exists in cache
        if expert_name not in self.expert_embeddings:
            # Create a default expert embedding
            if method == 'embedding_similarity':
                # Get the index of this expert
                expert_idx = self.expert_names.index(expert_name)
                
                # Always create a one-hot encoding for this expert
                # This will create a vector of length num_experts with 1.0 at the expert's index
                projection = torch.zeros(len(self.expert_names), device=self.device)
                projection[expert_idx] = 1.0  # One-hot encoding
                self.expert_embeddings[expert_name] = projection
                
        # Compute likelihood using the cached embedding
        if method == 'embedding_similarity':
            # Get the expert embedding from the cache
            expert_embedding = self.expert_embeddings[expert_name]
            
            # If input features are high-dimensional (like 768) and expert embedding is low-dimensional (like 4),
            # we need to project the input features to match the expert embedding dimension
            if input_features.dim() == 1 and expert_embedding.dim() == 1 and input_features.size(0) > expert_embedding.size(0):
                # Normalize the input features
                input_features_norm = F.normalize(input_features, p=2, dim=0)
                
                # Project input features to the same dimension as expert embeddings
                chunk_size = input_features.size(0) // expert_embedding.size(0)
                projected_input = torch.zeros_like(expert_embedding, device=self.device)
                for i in range(expert_embedding.size(0)):
                    start_idx = i * chunk_size
                    end_idx = start_idx + chunk_size if i < expert_embedding.size(0) - 1 else input_features.size(0)
                    projected_input[i] = torch.mean(input_features_norm[start_idx:end_idx])
                
                # Normalize the projected input
                projected_input = F.normalize(projected_input, p=2, dim=0)
                
                # Now compute similarity with expert embedding
                expert_embedding_norm = F.normalize(expert_embedding, p=2, dim=0)
                similarity = torch.sum(projected_input * expert_embedding_norm).item()
            else:
                # For all other cases (shapes match or other forms of mismatch),
                # we'll handle by flattening and resizing
                input_features_flat = input_features.flatten()
                expert_embedding_flat = expert_embedding.flatten()
                
                # Ensure both tensors have the same size
                if input_features_flat.size(0) != expert_embedding_flat.size(0):
                    if input_features_flat.size(0) < expert_embedding_flat.size(0):
                        # Repeat input features to match expert embedding size
                        repeat_factor = (expert_embedding_flat.size(0) + input_features_flat.size(0) - 1) // input_features_flat.size(0)
                        input_features_flat = input_features_flat.repeat(repeat_factor)[:expert_embedding_flat.size(0)]
                    else:
                        # Repeat expert embedding to match input features size
                        repeat_factor = (input_features_flat.size(0) + expert_embedding_flat.size(0) - 1) // expert_embedding_flat.size(0)
                        expert_embedding_flat = expert_embedding_flat.repeat(repeat_factor)[:input_features_flat.size(0)]
                
                # Normalize both tensors
                input_features_norm = F.normalize(input_features_flat, p=2, dim=0)
                expert_embedding_norm = F.normalize(expert_embedding_flat, p=2, dim=0)
                
                # Compute cosine similarity
                similarity = torch.sum(input_features_norm * expert_embedding_norm).item()
            
            # Convert to probability using softmax-like transformation
            likelihood = (similarity + 1) / 2  # Scale from [-1, 1] to [0, 1]
            
            # Apply temperature
            if self.temperature != 1.0:
                likelihood = likelihood ** (1 / self.temperature)
                
            # Ensure likelihood is within bounds
            likelihood = max(min(likelihood, self.max_probability), self.min_probability)
            
            return likelihood
            
        # For other methods, you could implement different likelihood functions
        
        # Default: return a small fixed likelihood
        return 0.1
            
    def update_beliefs(self, input_features: torch.Tensor) -> Dict[str, float]:
        """
        Update beliefs about expert probabilities using Bayes' rule
        
        Args:
            input_features: Input features tensor
            
        Returns:
            Dictionary mapping expert names to posterior probabilities
        """
        # Compute likelihoods for each expert
        likelihoods = {}
        for name in self.expert_names:
            likelihoods[name] = self.compute_likelihood(input_features, name)
            
        # Apply Bayes' rule
        # P(expert|input) ∝ P(input|expert) * P(expert)
        unnormalized_posteriors = {}
        for name in self.expert_names:
            unnormalized_posteriors[name] = likelihoods[name] * self.priors[name]
            
        # Normalize posteriors
        total = sum(unnormalized_posteriors.values())
        if total > 0:
            self.posteriors = {name: prob / total for name, prob in unnormalized_posteriors.items()}
        else:
            # If all posteriors are zero, revert to priors
            self.posteriors = self.priors.copy()
            
        # Record history
        self.history.append({
            'priors': self.priors.copy(),
            'likelihoods': likelihoods.copy(),
            'posteriors': self.posteriors.copy()
        })
            
        return self.posteriors
    
    def blend_expert_vectors(
        self, 
        expert_vectors: Dict[str, Any],
        mode: str = 'weighted_average',
        top_k: Optional[int] = None
    ) -> Any:
        """
        Blend expert vectors based on posterior probabilities
        
        Args:
            expert_vectors: Dictionary mapping expert names to expert vectors
            mode: Blending mode ('weighted_average', 'top_k', 'max')
            top_k: Number of top experts to consider (for 'top_k' mode)
            
        Returns:
            Blended expert vector
        """
        if mode == 'max':
            # Select the expert with the highest posterior
            top_expert = max(self.posteriors.items(), key=lambda x: x[1])[0]
            return expert_vectors[top_expert]
            
        elif mode == 'top_k':
            # Select top-k experts and renormalize
            if top_k is None or top_k >= len(self.expert_names):
                # If top_k is not specified or too large, use all experts
                return self.blend_expert_vectors(expert_vectors, mode='weighted_average')
                
            # Sort experts by posterior probability
            sorted_posteriors = sorted(
                self.posteriors.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Select top-k experts
            top_k_experts = sorted_posteriors[:top_k]
            
            # Renormalize posteriors
            total = sum(p for _, p in top_k_experts)
            normalized_posteriors = {name: p / total for name, p in top_k_experts}
            
            # Create a new dictionary with only the top-k experts
            modified_posteriors = self.posteriors.copy()
            for name in self.expert_names:
                if name not in [n for n, _ in top_k_experts]:
                    modified_posteriors[name] = 0.0
                else:
                    modified_posteriors[name] = normalized_posteriors[name]
                    
            # Blend using the modified posteriors
            temp_posteriors = self.posteriors
            self.posteriors = modified_posteriors
            result = self.blend_expert_vectors(expert_vectors, mode='weighted_average')
            self.posteriors = temp_posteriors
            return result
            
        else:  # weighted_average
            # If the expert vectors are simple tensors or scalars
            if all(isinstance(v, (torch.Tensor, float, int)) for v in expert_vectors.values()):
                # Convert scalars to tensors if needed
                converted_vectors = {}
                for name, vector in expert_vectors.items():
                    if isinstance(vector, (float, int)):
                        converted_vectors[name] = torch.tensor(vector, device=self.device)
                    else:
                        converted_vectors[name] = vector
                        
                # Compute weighted average
                result = None
                for name, vector in converted_vectors.items():
                    if name in self.posteriors:
                        weight = self.posteriors[name]
                        if result is None:
                            result = weight * vector
                        else:
                            result = result + weight * vector
                            
                return result
                
            # If the expert vectors are dictionaries (e.g., SVF expert vectors)
            elif all(isinstance(v, dict) for v in expert_vectors.values()):
                # Initialize result dictionary
                result = {}
                
                # Get all keys across all expert vectors
                all_keys = set()
                for vector in expert_vectors.values():
                    all_keys.update(vector.keys())
                    
                # Blend each key separately
                for key in all_keys:
                    key_vectors = {}
                    for name, vector in expert_vectors.items():
                        if key in vector:
                            key_vectors[name] = vector[key]
                    
                    # Recursively blend the key-specific vectors
                    result[key] = self.blend_expert_vectors(key_vectors, mode='weighted_average')
                    
                return result
                
            # If the expert vectors are nested dictionaries
            elif all(isinstance(v, dict) and all(isinstance(vv, dict) for vv in v.values()) for v in expert_vectors.values()):
                # Initialize result dictionary
                result = {}
                
                # Get all outer keys across all expert vectors
                all_outer_keys = set()
                for vector in expert_vectors.values():
                    all_outer_keys.update(vector.keys())
                    
                # Blend each outer key separately
                for outer_key in all_outer_keys:
                    outer_key_vectors = {}
                    for name, vector in expert_vectors.items():
                        if outer_key in vector:
                            outer_key_vectors[name] = vector[outer_key]
                    
                    # Recursively blend the outer-key-specific vectors
                    result[outer_key] = self.blend_expert_vectors(outer_key_vectors, mode='weighted_average')
                    
                return result
                
            # Fallback for other types
            else:
                print(f"Warning: Unsupported expert vector type for blending. Using max mode instead.")
                return self.blend_expert_vectors(expert_vectors, mode='max')
            
    def get_classification(self, threshold: float = 0.5) -> Optional[str]:
        """
        Get the most likely expert classification if its probability exceeds the threshold
        
        Args:
            threshold: Probability threshold for classification
            
        Returns:
            Name of the most likely expert, or None if no expert exceeds the threshold
        """
        top_expert = max(self.posteriors.items(), key=lambda x: x[1])
        if top_expert[1] >= threshold:
            return top_expert[0]
        return None
    
    def process_prompt(
        self,
        prompt: str,
        embedder: Any,
        expert_vectors: Dict[str, Any],
        mode: str = 'weighted_average',
        top_k: Optional[int] = None
    ) -> Any:
        """
        Process a prompt to get a blended expert vector
        
        Args:
            prompt: Input prompt
            embedder: Function to convert prompt to embeddings
            expert_vectors: Dictionary mapping expert names to expert vectors
            mode: Blending mode
            top_k: Number of top experts to consider
            
        Returns:
            Blended expert vector
        """
        # Get embeddings for the prompt
        embeddings = embedder(prompt)
        
        # Update beliefs
        self.update_beliefs(embeddings)
        
        # Blend expert vectors
        return self.blend_expert_vectors(expert_vectors, mode=mode, top_k=top_k)
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information about the current state
        
        Returns:
            Dictionary with debug information
        """
        return {
            'priors': self.priors,
            'posteriors': self.posteriors,
            'history': self.history
        }
