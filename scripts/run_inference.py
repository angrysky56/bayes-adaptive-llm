#!/usr/bin/env python
"""
Run inference with the Bayesian Adaptive LLM system.

This script loads pretrained expert vectors and runs inference with Bayesian dispatch.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.mambaformer import MambaFormer
from dispatch.bayesian_controller import BayesianController
from svf.svf import SVF


def create_mock_tokenizer():
    """Create a mock tokenizer for testing."""
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 32000
        
        def __call__(self, text, return_tensors=None, device=None):
            # Simple tokenization for testing
            device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokens = text.split()
            input_ids = torch.tensor([[i % 32000 for i in range(len(tokens))]], device=device)
            
            class Encoding:
                def __init__(self, input_ids):
                    self.input_ids = input_ids
            
            return Encoding(input_ids)
        
        def decode(self, token_ids, skip_special_tokens=False):
            # Simple decoding for testing
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            
            # Just return a mock response based on token pattern
            if any(t % 7 == 0 for t in token_ids):
                return "I've analyzed this as a math problem and computed the answer."
            elif any(t % 5 == 0 for t in token_ids):
                return "I've written this code with proper syntax and efficiency."
            elif any(t % 3 == 0 for t in token_ids):
                return "I've carefully reasoned through this logical problem."
            else:
                return "Here is my response based on my general knowledge."
    
    return MockTokenizer()


def get_mock_embedding(text, device="cpu"):
    """Get mock embeddings for demonstration purposes."""
    # Create simple keyword-based embeddings for testing
    print(f"\nAnalyzing prompt for expert classification...")
    
    math_keywords = ["math", "equation", "solve", "calculation", "factorial", "number", "sum", "product"]
    code_keywords = ["code", "function", "program", "algorithm", "python", "javascript", "coding"]
    reasoning_keywords = ["logic", "reason", "conclude", "inference", "argument", "premise", "analysis"]
    ethics_keywords = ["ethic", "moral", "should", "good", "bad", "right", "wrong", "value"]
    
    # Count keyword matches for each category
    math_score = sum(1 for kw in math_keywords if kw in text.lower())
    code_score = sum(1 for kw in code_keywords if kw in text.lower())
    reasoning_score = sum(1 for kw in reasoning_keywords if kw in text.lower())
    ethics_score = sum(1 for kw in ethics_keywords if kw in text.lower())
    
    # If no matches, use uniform distribution
    if math_score + code_score + reasoning_score + ethics_score == 0:
        print("No specific expert domain detected. Using uniform distribution.")
        return torch.tensor([0.25, 0.25, 0.25, 0.25], device=device)
    
    # Normalize scores
    total = math_score + code_score + reasoning_score + ethics_score
    math_prob = math_score / total
    code_prob = code_score / total
    reasoning_prob = reasoning_score / total
    ethics_prob = ethics_score / total
    
    # Print detected keywords
    print(f"Domain keyword matches:")
    print(f"  - Math: {math_score} ({math_prob:.2f})")
    print(f"  - Code: {code_score} ({code_prob:.2f})")
    print(f"  - Reasoning: {reasoning_score} ({reasoning_prob:.2f})")
    print(f"  - Ethics: {ethics_score} ({ethics_prob:.2f})")
    
    # Create embedding
    embedding = torch.tensor([math_prob, code_prob, reasoning_prob, ethics_prob], device=device)
    
    # Ensure embedding sums to 1
    embedding = embedding / embedding.sum()
    
    return embedding


def load_expert_vectors(expert_dir="experts/vectors"):
    """Load expert vectors from the specified directory."""
    expert_vectors = {}
    
    # Check if the directory exists
    if not os.path.exists(expert_dir):
        print(f"Expert directory {expert_dir} not found. Using mock expert vectors.")
        # Create mock expert vectors for testing
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {
            "math": torch.tensor(1.2, device=device),
            "code": torch.tensor(0.8, device=device),
            "reasoning": torch.tensor(1.0, device=device),
            "ethics": torch.tensor(0.9, device=device)
        }
    
    # Look for expert.pt files in subdirectories
    for expert_subdir in os.listdir(expert_dir):
        expert_path = os.path.join(expert_dir, expert_subdir, "expert.pt")
        if os.path.exists(expert_path):
            try:
                expert_vector = torch.load(expert_path)
                expert_vectors[expert_subdir] = expert_vector
                print(f"Loaded expert vector: {expert_subdir}")
            except Exception as e:
                print(f"Error loading expert vector {expert_subdir}: {e}")
    
    if not expert_vectors:
        print("No expert vectors found. Using mock expert vectors.")
        # Create mock expert vectors for testing
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {
            "math": torch.tensor(1.2, device=device),
            "code": torch.tensor(0.8, device=device),
            "reasoning": torch.tensor(1.0, device=device),
            "ethics": torch.tensor(0.9, device=device)
        }
    
    return expert_vectors


def create_model(config_path=None):
    """Create or load a model based on the provided configuration."""
    # Default configuration
    model_config = {
        "vocab_size": 32000,
        "d_model": 256,
        "n_layers": 2,
        "n_heads": 4,
        "d_state": 16,
        "dropout": 0.1,
        "max_seq_len": 1024
    }
    
    # Override with config file if provided
    if config_path and os.path.exists(config_path):
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
            if "model" in config and "config" in config["model"]:
                model_config.update(config["model"]["config"])
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaFormer(
        vocab_size=model_config["vocab_size"],
        d_model=model_config.get("d_model", model_config.get("dim", 256)),
        n_layers=model_config.get("n_layers", model_config.get("depth", 2)),
        n_heads=model_config.get("n_heads", 4),
        d_state=model_config.get("d_state", 16),
        dropout=model_config.get("dropout", 0.1),
        max_seq_len=model_config.get("max_seq_len", 1024)
    ).to(device)
    
    return model, device


def run_inference(
    prompt, 
    model, 
    controller, 
    expert_vectors, 
    device="cpu", 
    show_experts=False,
    blending_mode="weighted_average",
    top_k=None
):
    """Run inference with Bayesian expert selection."""
    # Create mock tokenizer
    tokenizer = create_mock_tokenizer()
    
    # Get embedding for prompt
    embedding = get_mock_embedding(prompt, device)
    
    # Update controller beliefs
    controller.update_beliefs(embedding)
    
    # Print expert probabilities if requested
    if show_experts:
        print("\nExpert probabilities:")
        for name, prob in sorted(controller.posteriors.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {prob:.4f}")
        
        # Get classification if probability exceeds threshold
        classification = controller.get_classification(threshold=0.4)
        if classification:
            print(f"Primary expert: {classification}")
        else:
            print("Using blended experts (no clear primary expert)")
    
    # Blend expert vectors
    blended_expert = controller.blend_expert_vectors(
        expert_vectors,
        mode=blending_mode,
        top_k=top_k
    )
    
    # Apply expert vector to model
    svf = SVF(model)
    
    # Handle the case where blended_expert might be a scalar
    # Debug what kind of expert vector we're dealing with
    if show_experts:
        print(f"\nExpert vector type: {type(blended_expert)}")
        if isinstance(blended_expert, torch.Tensor):
            print(f"Tensor shape: {blended_expert.shape}")
            print(f"Tensor device: {blended_expert.device}")
    
    # Make sure expert vector is handled properly based on its type
    svf.apply_expert_vector(blended_expert)
    
    # Tokenize prompt
    inputs = tokenizer(prompt, device=device)
    
    # Generate response
    with torch.no_grad():
        outputs = model(inputs.input_ids)
        
        # In a real implementation, we would do:
        # generated_ids = model.generate(
        #     inputs.input_ids,
        #     max_new_tokens=100,
        #     temperature=0.7,
        #     top_p=0.9
        # )
        # response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # For the demo, use the mock tokenizer's decode method
        response = tokenizer.decode(inputs.input_ids[0])
    
    return response


def main():
    """Main function for running inference."""
    parser = argparse.ArgumentParser(description="Run inference with Bayesian expert selection")
    parser.add_argument("--input", type=str, required=True, help="Input prompt")
    parser.add_argument("--expert-dir", type=str, default="experts/vectors", help="Directory containing expert vectors")
    parser.add_argument("--config", type=str, default="config/experiments/example.json", help="Model configuration")
    parser.add_argument("--show-experts", action="store_true", help="Show expert probabilities")
    parser.add_argument("--blending-mode", type=str, default="weighted_average", 
                       choices=["weighted_average", "top_k", "max"], help="Mode for blending expert vectors")
    parser.add_argument("--top-k", type=int, default=None, help="Number of top experts to consider")
    args = parser.parse_args()
    
    # Create or load model
    model, device = create_model(args.config)
    
    # Load expert vectors
    expert_vectors = load_expert_vectors(args.expert_dir)
    
    # Create Bayesian controller
    controller = BayesianController(
        expert_names=list(expert_vectors.keys()),
        device=device
    )
    
    # Run inference
    response = run_inference(
        args.input,
        model,
        controller,
        expert_vectors,
        device=device,
        show_experts=args.show_experts,
        blending_mode=args.blending_mode,
        top_k=args.top_k
    )
    
    # Print results
    print(f"\nPrompt: {args.input}")
    print(f"\nResponse: {response}")


if __name__ == "__main__":
    main()
