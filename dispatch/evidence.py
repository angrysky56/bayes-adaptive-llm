"""
Evidence functions for Bayesian controller.

This module implements various evidence functions that compute likelihoods
for expert selection in the Bayesian controller.
"""

import re
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable
from abc import ABC, abstractmethod


class EvidenceFunction(ABC):
    """Base class for evidence functions."""
    
    @abstractmethod
    def __call__(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Compute likelihoods for each expert based on input text.
        
        Args:
            input_text: Input text
            context: Additional context (optional)
            
        Returns:
            Dictionary mapping expert names to likelihoods
        """
        pass
    
    def normalize(self, likelihoods: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize likelihoods to sum to 1.
        
        Args:
            likelihoods: Dictionary mapping expert names to likelihoods
            
        Returns:
            Normalized likelihoods
        """
        # Get sum of likelihoods
        total = sum(likelihoods.values())
        
        # If total is 0, return uniform distribution
        if total == 0:
            return {expert: 1.0 / len(likelihoods) for expert in likelihoods}
        
        # Normalize likelihoods
        return {expert: likelihood / total for expert, likelihood in likelihoods.items()}


class KeywordEvidenceFunction(EvidenceFunction):
    """
    Evidence function based on keyword matching.
    
    This function computes likelihoods based on the presence of keywords
    in the input text. The keywords are specified for each expert in the
    configuration.
    """
    
    def __init__(self, config: Dict[str, List[str]]):
        """
        Initialize the keyword evidence function.
        
        Args:
            config: Dictionary mapping expert names to lists of keywords
        """
        self.keyword_sets = config
    
    def __call__(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Compute likelihoods based on keyword matching.
        
        Args:
            input_text: Input text
            context: Additional context (optional)
            
        Returns:
            Dictionary mapping expert names to likelihoods
        """
        # Compute likelihoods
        likelihoods = {}
        
        for expert, keywords in self.keyword_sets.items():
            # Count the number of matching keywords
            count = 0
            for keyword in keywords:
                # Use word boundary regex for more accurate matching
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                count += len(re.findall(pattern, input_text.lower()))
            
            # Compute likelihood
            likelihood = min(1.0, count / 10.0)  # Cap at 1.0
            likelihoods[expert] = likelihood
        
        # Normalize likelihoods
        return self.normalize(likelihoods)


class EmbeddingEvidenceFunction(EvidenceFunction):
    """
    Evidence function based on embedding similarity.
    
    This function computes likelihoods based on the similarity between
    the input text embedding and reference embeddings for each expert.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the embedding evidence function.
        
        Args:
            config: Configuration for the embedding evidence function
        """
        self.model_name = config.get("model", "all-MiniLM-L6-v2")
        self.threshold = config.get("threshold", 0.75)
        
        # Try to import sentence_transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            print("Warning: sentence_transformers not installed. Using dummy embeddings.")
            self.model = None
        
        # Load reference embeddings
        self.reference_embeddings = self._load_reference_embeddings()
    
    def _load_reference_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load reference embeddings for each expert.
        
        Returns:
            Dictionary mapping expert names to reference embeddings
        """
        # In a real implementation, you would load pre-computed embeddings
        # For now, let's return dummy embeddings
        return {
            "math": np.random.randn(384),
            "code": np.random.randn(384),
            "reasoning": np.random.randn(384),
            "alignment": np.random.randn(384),
        }
    
    def __call__(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Compute likelihoods based on embedding similarity.
        
        Args:
            input_text: Input text
            context: Additional context (optional)
            
        Returns:
            Dictionary mapping expert names to likelihoods
        """
        # Compute input embedding
        if self.model is not None:
            input_embedding = self.model.encode(input_text)
        else:
            # Use dummy embedding if model is not available
            input_embedding = np.random.randn(384)
        
        # Compute similarities
        likelihoods = {}
        
        for expert, reference_embedding in self.reference_embeddings.items():
            # Compute cosine similarity
            similarity = np.dot(input_embedding, reference_embedding) / (
                np.linalg.norm(input_embedding) * np.linalg.norm(reference_embedding)
            )
            
            # Apply threshold
            likelihood = max(0.0, similarity - self.threshold) / (1.0 - self.threshold)
            likelihoods[expert] = likelihood
        
        # Normalize likelihoods
        return self.normalize(likelihoods)


class ClassifierEvidenceFunction(EvidenceFunction):
    """
    Evidence function based on a trained classifier.
    
    This function computes likelihoods using a trained classifier
    that predicts the probability of each expert being relevant.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the classifier evidence function.
        
        Args:
            config: Configuration for the classifier evidence function
        """
        self.model_path = config.get("model_path")
        
        # Load classifier model
        self.model = self._load_classifier_model()
    
    def _load_classifier_model(self):
        """
        Load classifier model.
        
        Returns:
            Classifier model
        """
        # In a real implementation, you would load a trained classifier
        # For now, let's return a dummy model
        return DummyClassifier()
    
    def __call__(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Compute likelihoods using a trained classifier.
        
        Args:
            input_text: Input text
            context: Additional context (optional)
            
        Returns:
            Dictionary mapping expert names to likelihoods
        """
        # Use classifier to predict probabilities
        probabilities = self.model.predict_proba(input_text)
        
        # Return probabilities as likelihoods
        return probabilities


class CompositeEvidenceFunction(EvidenceFunction):
    """
    Composite evidence function that combines multiple evidence functions.
    
    This function combines the likelihoods from multiple evidence functions
    using a weighted average.
    """
    
    def __init__(
        self,
        evidence_functions: List[EvidenceFunction],
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize the composite evidence function.
        
        Args:
            evidence_functions: List of evidence functions
            weights: List of weights for each evidence function (optional)
        """
        self.evidence_functions = evidence_functions
        
        # Set weights
        if weights is None:
            # Use uniform weights if not provided
            self.weights = [1.0 / len(evidence_functions)] * len(evidence_functions)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [weight / total for weight in weights]
    
    def __call__(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Compute likelihoods using multiple evidence functions.
        
        Args:
            input_text: Input text
            context: Additional context (optional)
            
        Returns:
            Dictionary mapping expert names to likelihoods
        """
        # Compute likelihoods from each evidence function
        all_likelihoods = []
        
        for evidence_fn in self.evidence_functions:
            all_likelihoods.append(evidence_fn(input_text, context))
        
        # Combine likelihoods using weighted average
        combined_likelihoods = {}
        
        # Get all expert names
        expert_names = set()
        for likelihoods in all_likelihoods:
            expert_names.update(likelihoods.keys())
        
        # Compute combined likelihoods
        for expert in expert_names:
            # Compute weighted average
            combined_likelihood = 0.0
            
            for i, likelihoods in enumerate(all_likelihoods):
                # Use 0.0 if expert is not in likelihoods
                likelihood = likelihoods.get(expert, 0.0)
                combined_likelihood += self.weights[i] * likelihood
            
            combined_likelihoods[expert] = combined_likelihood
        
        # Normalize likelihoods
        return self.normalize(combined_likelihoods)


# Dummy classifier for testing
class DummyClassifier:
    """Dummy classifier for testing."""
    
    def predict_proba(self, input_text: str) -> Dict[str, float]:
        """
        Predict probabilities for each expert.
        
        Args:
            input_text: Input text
            
        Returns:
            Dictionary mapping expert names to probabilities
        """
        # Return dummy probabilities
        return {
            "math": 0.25,
            "code": 0.25,
            "reasoning": 0.25,
            "alignment": 0.25,
        }


def create_evidence_function(func_type: str, config: Dict[str, Any]) -> EvidenceFunction:
    """
    Create an evidence function by type.
    
    Args:
        func_type: Type of evidence function
        config: Configuration for the evidence function
        
    Returns:
        Evidence function
    """
    if func_type == "KeywordEvidenceFunction":
        return KeywordEvidenceFunction(config)
    elif func_type == "EmbeddingEvidenceFunction":
        return EmbeddingEvidenceFunction(config)
    elif func_type == "ClassifierEvidenceFunction":
        return ClassifierEvidenceFunction(config)
    else:
        raise ValueError(f"Unknown evidence function type: {func_type}")
