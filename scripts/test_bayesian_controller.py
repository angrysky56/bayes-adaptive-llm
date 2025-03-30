#!/usr/bin/env python
"""
Test script for the BayesianController

This script tests the functionality of the BayesianController for expert adaptation.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from dispatch.bayesian_controller import BayesianController


def test_basic_functionality():
    """
    Test basic BayesianController functionality
    """
    print("Testing basic BayesianController functionality...")
    
    # Set the device to CPU for testing
    device = torch.device("cpu")
    
    # Create a controller with some experts
    experts = ["math", "code", "reasoning", "ethics"]
    controller = BayesianController(expert_names=experts, device=device)
    
    # Check initial priors
    print(f"Initial priors: {controller.priors}")
    assert len(controller.priors) == len(experts), "Priors should include all experts"
    assert abs(sum(controller.priors.values()) - 1.0) < 1e-6, "Priors should sum to 1.0"
    
    # Update beliefs with a mock embedding
    embedding = torch.randn(768, device=device)  # Mock embedding
    posteriors = controller.update_beliefs(embedding)
    
    print(f"Updated posteriors: {posteriors}")
    assert len(posteriors) == len(experts), "Posteriors should include all experts"
    assert abs(sum(posteriors.values()) - 1.0) < 1e-6, "Posteriors should sum to 1.0"
    
    print("✓ Basic functionality test passed!")
    return True


def test_expert_blending():
    """
    Test expert vector blending
    """
    print("\nTesting expert vector blending...")
    
    # Set the device to CPU for testing
    device = torch.device("cpu")
    
    # Create a controller with some experts
    experts = ["math", "code", "reasoning"]
    controller = BayesianController(expert_names=experts, device=device)
    
    # Custom priors
    custom_priors = {
        "math": 0.5,
        "code": 0.3,
        "reasoning": 0.2
    }
    controller.update_priors(custom_priors)
    
    # Create mock expert vectors
    expert_vectors = {
        "math": torch.ones(10, device=device) * 2.0,
        "code": torch.ones(10, device=device) * 0.5,
        "reasoning": torch.ones(10, device=device) * 1.0
    }
    
    # Test different blending modes
    print("Testing weighted average blending...")
    blended_weighted = controller.blend_expert_vectors(expert_vectors, mode="weighted_average")
    expected_weighted = 2.0 * 0.5 + 0.5 * 0.3 + 1.0 * 0.2
    print(f"Blended value (first element): {blended_weighted[0].item()}")
    print(f"Expected value: {expected_weighted}")
    assert abs(blended_weighted[0].item() - expected_weighted) < 1e-6, "Weighted average blending incorrect"
    
    print("Testing max mode blending...")
    blended_max = controller.blend_expert_vectors(expert_vectors, mode="max")
    print(f"Blended value (first element): {blended_max[0].item()}")
    print(f"Expected value: 2.0")
    assert abs(blended_max[0].item() - 2.0) < 1e-6, "Max mode blending incorrect"
    
    print("Testing top-k mode blending...")
    blended_topk = controller.blend_expert_vectors(expert_vectors, mode="top_k", top_k=2)
    expected_topk = (2.0 * 0.5 + 0.5 * 0.3) / (0.5 + 0.3)
    print(f"Blended value (first element): {blended_topk[0].item()}")
    print(f"Expected value: {expected_topk}")
    assert abs(blended_topk[0].item() - expected_topk) < 1e-6, "Top-k mode blending incorrect"
    
    print("✓ Expert blending test passed!")
    return True


def test_nested_vector_blending():
    """
    Test blending of nested expert vectors (SVF format)
    """
    print("\nTesting nested vector blending...")
    
    # Set the device to CPU for testing
    device = torch.device("cpu")
    
    # Create a controller with some experts
    experts = ["math", "code"]
    controller = BayesianController(expert_names=experts, device=device)
    
    # Set equal posteriors for simplicity
    controller.posteriors = {
        "math": 0.5,
        "code": 0.5
    }
    
    # Create mock SVF expert vectors
    expert_vectors = {
        "math": {
            "layer_0": {
                "fc1": torch.ones(10, device=device) * 2.0,
                "fc2": torch.ones(10, device=device) * 3.0
            },
            "layer_1": {
                "fc1": torch.ones(10, device=device) * 4.0,
                "fc2": torch.ones(10, device=device) * 5.0
            }
        },
        "code": {
            "layer_0": {
                "fc1": torch.ones(10, device=device) * 1.0,
                "fc2": torch.ones(10, device=device) * 1.5
            },
            "layer_1": {
                "fc1": torch.ones(10, device=device) * 2.0,
                "fc2": torch.ones(10, device=device) * 2.5
            }
        }
    }
    
    # Blend the vectors
    blended = controller.blend_expert_vectors(expert_vectors)
    
    # Check the results
    print(f"Blended layer_0/fc1 (first element): {blended['layer_0']['fc1'][0].item()}")
    print(f"Expected layer_0/fc1 (first element): {(2.0 * 0.5 + 1.0 * 0.5)}")
    assert abs(blended['layer_0']['fc1'][0].item() - 1.5) < 1e-6, "Nested blending incorrect for layer_0/fc1"
    
    print(f"Blended layer_1/fc2 (first element): {blended['layer_1']['fc2'][0].item()}")
    print(f"Expected layer_1/fc2 (first element): {(5.0 * 0.5 + 2.5 * 0.5)}")
    assert abs(blended['layer_1']['fc2'][0].item() - 3.75) < 1e-6, "Nested blending incorrect for layer_1/fc2"
    
    print("✓ Nested vector blending test passed!")
    return True


def test_prompt_classification():
    """
    Test classifying prompts into expert domains
    """
    print("\nTesting prompt classification...")
    
    # Set the device to CPU for testing
    device = torch.device("cpu")
    
    # Mock embedding function
    def mock_embedder(prompt):
        if "math" in prompt.lower():
            return torch.tensor([1.0, 0.2, 0.1], device=device)
        elif "code" in prompt.lower():
            return torch.tensor([0.2, 1.0, 0.1], device=device)
        elif "reasoning" in prompt.lower():
            return torch.tensor([0.1, 0.2, 1.0], device=device)
        else:
            return torch.tensor([0.3, 0.3, 0.3], device=device)
    
    # Create a controller with some experts
    experts = ["math", "code", "reasoning"]
    controller = BayesianController(expert_names=experts, temperature=0.5, device=device)
    
    # Set up expert embeddings
    for name in experts:
        # Use prompt with expert name to initialize the expert embedding
        prompt = f"This is a {name} problem"
        embedding = mock_embedder(prompt)
        controller.compute_likelihood(embedding, name)
    
    # Test with different prompts
    math_prompt = "What is the solution to this equation: 2x + 3 = 7?"
    code_prompt = "Write a Python function to sort a list."
    reasoning_prompt = "If all A are B, and some B are C, what can we say about A and C?"
    
    # Process math prompt
    math_embedding = mock_embedder(math_prompt)
    controller.update_beliefs(math_embedding)
    math_class = controller.get_classification(threshold=0.4)
    print(f"Math prompt classification: {math_class}")
    print(f"Posteriors: {controller.posteriors}")
    
    # Process code prompt
    controller.priors = {name: 1.0 / len(experts) for name in experts}  # Reset priors
    code_embedding = mock_embedder(code_prompt)
    controller.update_beliefs(code_embedding)
    code_class = controller.get_classification(threshold=0.4)
    print(f"Code prompt classification: {code_class}")
    print(f"Posteriors: {controller.posteriors}")
    
    # Process reasoning prompt
    controller.priors = {name: 1.0 / len(experts) for name in experts}  # Reset priors
    reasoning_embedding = mock_embedder(reasoning_prompt)
    controller.update_beliefs(reasoning_embedding)
    reasoning_class = controller.get_classification(threshold=0.4)
    print(f"Reasoning prompt classification: {reasoning_class}")
    print(f"Posteriors: {controller.posteriors}")
    
    print("✓ Prompt classification test passed!")
    return True


if __name__ == "__main__":
    # Run all tests
    basic_success = test_basic_functionality()
    blending_success = test_expert_blending()
    nested_success = test_nested_vector_blending()
    classification_success = test_prompt_classification()
    
    # Check overall success
    if all([basic_success, blending_success, nested_success, classification_success]):
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed!")
