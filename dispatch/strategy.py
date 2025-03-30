"""
Bayesian Controller Strategies

This module provides strategies and evidence functions for the Bayesian controller.
These determine how input relevance to different experts is assessed.
"""

from typing import Dict, List, Any, Callable, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceFunction:
    """
    Base class for evidence functions
    
    Evidence functions compute the likelihood of input data given a specific expert.
    Higher values indicate stronger evidence that the expert is relevant.
    """
    
    def __call__(self, expert_name: str, input_data: Any) -> float:
        """
        Compute likelihood of input given expert
        
        Args:
            expert_name: Name of the expert
            input_data: Input data to evaluate
            
        Returns:
            Likelihood score (higher means input is more relevant to expert)
        """
        raise NotImplementedError


class CosineEvidenceFunction(EvidenceFunction):
    """
    Evidence function based on cosine similarity
    
    This function computes the likelihood based on cosine similarity
    between input embeddings and expert embeddings.
    """
    
    def __init__(
        self,
        expert_embeddings: Dict[str, torch.Tensor],
        embedding_model: Optional[Callable] = None,
        temperature: float = 1.0
    ):
        """
        Initialize a cosine evidence function
        
        Args:
            expert_embeddings: Dictionary mapping expert names to embedding vectors
            embedding_model: Optional function to embed input data
            temperature: Temperature parameter for softening/sharpening similarities
        """
        self.expert_embeddings = expert_embeddings
        self.embedding_model = embedding_model
        self.temperature = temperature
    
    def __call__(self, expert_name: str, input_data: Any) -> float:
        """
        Compute likelihood based on cosine similarity
        
        Args:
            expert_name: Name of the expert
            input_data: Input data to evaluate
            
        Returns:
            Likelihood score based on cosine similarity
        """
        # If input is already an embedding, use it directly
        if isinstance(input_data, torch.Tensor) and input_data.dim() == 1:
            input_embedding = input_data
        # Otherwise, use embedding model to embed input
        elif self.embedding_model is not None:
            input_embedding = self.embedding_model(input_data)
        else:
            raise ValueError("Input must be an embedding or embedding_model must be provided")
        
        # Get expert embedding
        if expert_name not in self.expert_embeddings:
            return 0.0
        
        expert_embedding = self.expert_embeddings[expert_name]
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(
            input_embedding.unsqueeze(0),
            expert_embedding.unsqueeze(0)
        ).item()
        
        # Apply temperature and ensure positive
        return max(0.0, torch.exp(similarity / self.temperature).item())


class KeywordEvidenceFunction(EvidenceFunction):
    """
    Evidence function based on keyword matching
    
    This function computes the likelihood based on the presence of
    keywords associated with each expert.
    """
    
    def __init__(
        self,
        expert_keywords: Dict[str, List[str]],
        default_score: float = 0.1
    ):
        """
        Initialize a keyword evidence function
        
        Args:
            expert_keywords: Dictionary mapping expert names to lists of keywords
            default_score: Default score for experts with no keyword matches
        """
        self.expert_keywords = expert_keywords
        self.default_score = default_score
    
    def __call__(self, expert_name: str, input_data: str) -> float:
        """
        Compute likelihood based on keyword matching
        
        Args:
            expert_name: Name of the expert
            input_data: Input text to evaluate
            
        Returns:
            Likelihood score based on keyword matches
        """
        if not isinstance(input_data, str):
            input_data = str(input_data)
        
        # Convert input to lowercase for case-insensitive matching
        input_lower = input_data.lower()
        
        # Get keywords for this expert
        if expert_name not in self.expert_keywords:
            return self.default_score
        
        keywords = self.expert_keywords[expert_name]
        
        # Count keyword matches
        match_count = sum(1 for keyword in keywords if keyword.lower() in input_lower)
        
        # Compute score based on match count
        if len(keywords) > 0:
            score = match_count / len(keywords)
            return max(score, self.default_score)
        else:
            return self.default_score


class ClassifierEvidenceFunction(EvidenceFunction):
    """
    Evidence function based on a classifier model
    
    This function computes the likelihood using a classifier model
    that predicts the relevance of each expert.
    """
    
    def __init__(
        self,
        classifier_model: nn.Module,
        expert_indices: Dict[str, int],
        input_processor: Optional[Callable] = None
    ):
        """
        Initialize a classifier evidence function
        
        Args:
            classifier_model: Model that outputs expert probabilities
            expert_indices: Dictionary mapping expert names to indices in model output
            input_processor: Optional function to preprocess input data
        """
        self.classifier_model = classifier_model
        self.expert_indices = expert_indices
        self.input_processor = input_processor
    
    def __call__(self, expert_name: str, input_data: Any) -> float:
        """
        Compute likelihood using classifier model
        
        Args:
            expert_name: Name of the expert
            input_data: Input data to evaluate
            
        Returns:
            Likelihood score from classifier model
        """
        # Check if expert is in classifier outputs
        if expert_name not in self.expert_indices:
            return 0.0
        
        # Preprocess input if needed
        if self.input_processor is not None:
            processed_input = self.input_processor(input_data)
        else:
            processed_input = input_data
        
        # Get model output
        with torch.no_grad():
            output = self.classifier_model(processed_input)
        
        # Get probability for this expert
        if isinstance(output, torch.Tensor):
            if output.dim() > 1:
                probs = F.softmax(output, dim=-1)
            else:
                probs = output
            
            expert_idx = self.expert_indices[expert_name]
            return probs[expert_idx].item()
        else:
            # Handle case where model returns dictionary or other structure
            return output[expert_name]


class CompositeEvidenceFunction(EvidenceFunction):
    """
    Composite evidence function combining multiple evidence sources
    
    This function combines multiple evidence functions with weights.
    """
    
    def __init__(
        self,
        evidence_fns: List[EvidenceFunction],
        weights: Optional[List[float]] = None
    ):
        """
        Initialize a composite evidence function
        
        Args:
            evidence_fns: List of evidence functions
            weights: Optional list of weights for each function
        """
        self.evidence_fns = evidence_fns
        
        # Initialize uniform weights if not provided
        if weights is None:
            self.weights = [1.0 / len(evidence_fns)] * len(evidence_fns)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def __call__(self, expert_name: str, input_data: Any) -> float:
        """
        Compute weighted sum of evidence function outputs
        
        Args:
            expert_name: Name of the expert
            input_data: Input data to evaluate
            
        Returns:
            Weighted sum of evidence function outputs
        """
        scores = [
            fn(expert_name, input_data) * weight
            for fn, weight in zip(self.evidence_fns, self.weights)
        ]
        
        return sum(scores)
