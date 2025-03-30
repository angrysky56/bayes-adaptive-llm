#!/usr/bin/env python
"""
Bayesian Adaptive Inference for MambaFormer

This script demonstrates how to use the BayesianController with MambaFormer
for adaptive expert selection and inference.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.mambaformer import MambaFormer
from dispatch.bayesian_controller import BayesianController
# Import directly from SVF class instead
from svf.svf import SVF


def load_model(model_path: str, config: Dict[str, Any]) -> MambaFormer:
    """
    Load a pre-trained MambaFormer model
    
    Args:
        model_path: Path to the model checkpoint
        config: Model configuration dictionary
        
    Returns:
        Loaded MambaFormer model
    """
    # Create model with the provided configuration
    model = MambaFormer(
        vocab_size=config.get("vocab_size", 32000),
        d_model=config.get("d_model", 1024),
        n_layers=config.get("n_layers", 24),
        n_heads=config.get("n_heads", 16),
        d_state=config.get("d_state", 256),
        dropout=config.get("dropout", 0.1),
        max_seq_len=config.get("max_seq_len", 4096)
    )
    
    # Load weights from checkpoint
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"Warning: Model checkpoint {model_path} not found. Using randomly initialized model.")
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Set to evaluation mode
    model.eval()
    
    return model


def load_expert_vectors(expert_dir: str) -> Dict[str, Any]:
    """
    Load expert vectors from a directory
    
    Args:
        expert_dir: Directory containing expert vectors
        
    Returns:
        Dictionary mapping expert names to expert vectors
    """
    expert_vectors = {}
    expert_paths = list(Path(expert_dir).glob("*.pt"))
    
    if not expert_paths:
        print(f"Warning: No expert vectors found in {expert_dir}")
        return expert_vectors
    
    # Load each expert vector
    for path in expert_paths:
        expert_name = path.stem
        expert_vector = torch.load(path)
        expert_vectors[expert_name] = expert_vector
        print(f"Loaded expert vector: {expert_name}")
    
    return expert_vectors


def get_embeddings(text: str, model: Any, tokenizer: Any) -> torch.Tensor:
    """
    Get embeddings for a text prompt
    
    Args:
        text: Input text
        model: Model to use for generating embeddings
        tokenizer: Tokenizer to use for preprocessing
        
    Returns:
        Embeddings tensor
    """
    # Get the device from the model
    device = next(model.parameters()).device
    
    # For the purposes of testing, we'll create simplified embeddings based on keywords
    # rather than actual model embeddings, since we're using a mock model
    if "math" in text.lower() or "calculate" in text.lower() or "value" in text.lower():
        return torch.tensor([0.9, 0.1, 0.1, 0.1], device=device)
    elif "code" in text.lower() or "function" in text.lower() or "python" in text.lower():
        return torch.tensor([0.1, 0.9, 0.1, 0.1], device=device)
    elif "reason" in text.lower() or "why" in text.lower() or "explain" in text.lower():
        return torch.tensor([0.1, 0.1, 0.9, 0.1], device=device)
    elif "ethic" in text.lower() or "moral" in text.lower() or "should" in text.lower():
        return torch.tensor([0.1, 0.1, 0.1, 0.9], device=device)
    else:
        return torch.tensor([0.25, 0.25, 0.25, 0.25], device=device)
    
    # In a real implementation with a functioning model, you would use something like:
    # # Tokenize the input (with batch dimension 1)
    # input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    # 
    # # Generate embeddings (without gradients)
    # with torch.no_grad():
    #     # Extract a representation from the model's hidden states
    #     outputs = model(input_ids)
    #     # Use the final layer's output from the last token
    #     embeddings = outputs[:, -1, :]
    # 
    # return embeddings


def run_inference(
    prompt: str,
    model: MambaFormer,
    tokenizer: Any,
    controller: BayesianController,
    expert_vectors: Dict[str, Any],
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    blending_mode: str = "weighted_average",
    top_k: Optional[int] = None,
    debug: bool = False
) -> str:
    """
    Run inference with Bayesian adaptive expert selection
    
    Args:
        prompt: Input prompt
        model: MambaFormer model
        tokenizer: Tokenizer for preprocessing
        controller: BayesianController for expert selection
        expert_vectors: Dictionary of expert vectors
        max_length: Maximum output length
        temperature: Temperature for sampling
        top_p: Top-p probability for nucleus sampling
        blending_mode: Mode for blending expert vectors
        top_k: Number of top experts to consider
        debug: Whether to print debug information
        
    Returns:
        Generated text
    """
    # Get the device from the model
    device = next(model.parameters()).device
    
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Get embeddings for the prompt
    embeddings = get_embeddings(prompt, model, tokenizer)
    
    # Update controller beliefs and get blended expert vector
    controller.update_beliefs(embeddings)
    blended_expert = controller.blend_expert_vectors(
        expert_vectors, 
        mode=blending_mode,
        top_k=top_k
    )
    
    # Print debug information if requested
    if debug:
        print("\nDebug Information:")
        print(f"Posterior probabilities:")
        for name, prob in controller.posteriors.items():
            print(f"  {name}: {prob:.4f}")
        
        # Print classification if probability exceeds threshold
        classification = controller.get_classification(threshold=0.4)
        if classification:
            print(f"Classified as: {classification}")
        else:
            print("No clear classification")
    
    # Generate text with the adapted model
    with torch.no_grad():
        # Apply expert vector to model
        if blended_expert:
            # Use SVF class instead of direct function call
            svf_manager = SVF(model)
            svf_manager.apply_expert_vector(blended_expert)
            
        # In a real implementation, would call model.generate
        # For testing, just return a mock response
        output_ids = input_ids
    
    # Decode the output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove the prompt)
    generated_text = output_text[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
    
    return generated_text


def create_mock_tokenizer():
    """
    Create a mock tokenizer for testing
    
    Returns:
        Mock tokenizer object
    """
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 32000
        
        def __call__(self, text, return_tensors=None):
            # Simple tokenization for testing
            tokens = text.split()
            input_ids = torch.tensor([[i % 32000 for i in range(len(tokens))]])
            
            class Encoding:
                def __init__(self, input_ids):
                    self.input_ids = input_ids
            
            return Encoding(input_ids)
        
        def decode(self, token_ids, skip_special_tokens=False):
            # Simple decoding for testing
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            
            # Just return a placeholder string
            return "Generated text for testing"
    
    return MockTokenizer()


def simulate_embedding_model():
    """
    Simulate a model that produces embeddings
    
    Returns:
        Mock embedding model
    """
    class MockEmbeddingModel:
        def __init__(self):
            self.embedding_dim = 768
            
        def __call__(self, input_ids, output_hidden_states=False):
            batch_size = input_ids.shape[0]
            
            class Outputs:
                def __init__(self, hidden_states):
                    self.hidden_states = hidden_states
            
            # Generate random embeddings for testing
            device = input_ids.device
            hidden_states = [torch.randn(batch_size, input_ids.shape[1], self.embedding_dim, device=device)]
            
            return Outputs(hidden_states)
    
    return MockEmbeddingModel()


def main():
    """
    Main function for Bayesian adaptive inference
    """
    parser = argparse.ArgumentParser(description="Bayesian Adaptive Inference for MambaFormer")
    parser.add_argument("--model_path", type=str, default="model.pt", help="Path to model checkpoint")
    parser.add_argument("--expert_dir", type=str, default="experts", help="Directory containing expert vectors")
    parser.add_argument("--config", type=str, default="config.json", help="Path to model configuration")
    parser.add_argument("--prompt", type=str, default="Hello, world!", help="Input prompt")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum output length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p probability for nucleus sampling")
    parser.add_argument("--blending_mode", type=str, default="weighted_average", 
                        choices=["weighted_average", "top_k", "max"], 
                        help="Mode for blending expert vectors")
    parser.add_argument("--top_k", type=int, default=None, help="Number of top experts to consider")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    parser.add_argument("--test", action="store_true", help="Run in test mode with mock objects")
    
    args = parser.parse_args()
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.test:
        print("Running in test mode with mock objects")
        
        # Create mock objects for testing
        model_config = {
            "vocab_size": 32000,
            "d_model": 256,
            "n_layers": 2,
            "n_heads": 4,
            "d_state": 16,
            "dropout": 0.1,
            "max_seq_len": 1024
        }
        
        # Create a small model for testing
        model = MambaFormer(
            vocab_size=model_config["vocab_size"],
            d_model=model_config["d_model"],
            n_layers=model_config["n_layers"],
            n_heads=model_config["n_heads"],
            d_state=model_config["d_state"],
            dropout=model_config["dropout"],
            max_seq_len=model_config["max_seq_len"]
        ).to(device)
        
        # Create mock tokenizer and embedding model
        tokenizer = create_mock_tokenizer()
        embedding_model = simulate_embedding_model()
        
        # Create mock expert vectors
        expert_vectors = {
            "math": 1.2,
            "code": 0.8,
            "reasoning": 1.0,
            "ethics": 1.5
        }
        
        # Create Bayesian controller
        controller = BayesianController(
            expert_names=list(expert_vectors.keys()),
            device=device
        )
        
        # Initialize expert embeddings for the classifier
        controller.likelihood_cache = {
            "math": torch.tensor([0.9, 0.1, 0.1, 0.1], device=device),
            "code": torch.tensor([0.1, 0.9, 0.1, 0.1], device=device),
            "reasoning": torch.tensor([0.1, 0.1, 0.9, 0.1], device=device),
            "ethics": torch.tensor([0.1, 0.1, 0.1, 0.9], device=device)
        }
        
        # Define a custom embedding function for testing
        def mock_get_embeddings(text, model, tokenizer):
            # Simple embedding that directly matches patterns in the text to expert embeddings
            # These embeddings are designed to directly match the likelihood patterns in controller.likelihood_cache
            if "math" in text.lower() or "value" in text.lower() or "calculate" in text.lower():
                return torch.tensor([0.9, 0.1, 0.1, 0.1], device=device)
            elif "code" in text.lower() or "python" in text.lower() or "function" in text.lower():
                return torch.tensor([0.1, 0.9, 0.1, 0.1], device=device)
            elif "reason" in text.lower() or "why" in text.lower() or "explain" in text.lower():
                return torch.tensor([0.1, 0.1, 0.9, 0.1], device=device)
            elif "ethic" in text.lower() or "moral" in text.lower() or "should" in text.lower():
                return torch.tensor([0.1, 0.1, 0.1, 0.9], device=device)
            else:
                return torch.tensor([0.25, 0.25, 0.25, 0.25], device=device)
        
        # Test with a few prompts
        test_prompts = [
            "What is the value of 2 + 2?",
            "Write a Python function to sort a list.",
            "Why is the sky blue?",
            "Is it ethical to use AI for surveillance?",
            "Tell me about yourself."
        ]
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            
            # Get embeddings for the prompt
            embeddings = mock_get_embeddings(prompt, embedding_model, tokenizer)
            
            # Update controller beliefs and get blended expert vector
            controller.update_beliefs(embeddings)
            blended_expert = controller.blend_expert_vectors(
                expert_vectors, 
                mode=args.blending_mode,
                top_k=args.top_k
            )
            
            # Print debug information
            print("Posterior probabilities:")
            for name, prob in controller.posteriors.items():
                print(f"  {name}: {prob:.4f}")
            
            # Print classification if probability exceeds threshold
            classification = controller.get_classification(threshold=0.3)
            if classification:
                print(f"Classified as: {classification}")
            else:
                print("No clear classification")
            
            # Print the blended expert value
            print(f"Blended expert value: {blended_expert}")
            
            # For actual inference, we would do:
            # output = run_inference(
            #     prompt=prompt,
            #     model=model,
            #     tokenizer=tokenizer,
            #     controller=controller,
            #     expert_vectors=expert_vectors,
            #     max_length=args.max_length,
            #     temperature=args.temperature,
            #     top_p=args.top_p,
            #     blending_mode=args.blending_mode,
            #     top_k=args.top_k,
            #     debug=args.debug
            # )
            # print(f"Generated: {output}")
    else:
        # Load model configuration
        import json
        with open(args.config, 'r') as f:
            model_config = json.load(f)
        
        # Load the model
        model = load_model(args.model_path, model_config)
        
        # Load expert vectors
        expert_vectors = load_expert_vectors(args.expert_dir)
        
        # Load the tokenizer
        # In practice, you would use the appropriate tokenizer for your model
        # For example, from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer")
        print("Warning: Using mock tokenizer. In practice, use the appropriate tokenizer for your model.")
        tokenizer = create_mock_tokenizer()

        # Create Bayesian controller
        controller = BayesianController(
            expert_names=list(expert_vectors.keys()),
            device=device
        )
        
        # Initialize expert embeddings
        controller.likelihood_cache = {
            "math": torch.tensor([0.8, 0.2, 0.1, 0.1], device=device),
            "code": torch.tensor([0.2, 0.8, 0.1, 0.1], device=device),
            "reasoning": torch.tensor([0.1, 0.1, 0.8, 0.1], device=device),
            "ethics": torch.tensor([0.1, 0.1, 0.1, 0.8], device=device)
        }
        
        # Set initial priors (uniform by default)
        controller.update_priors({
            "math": 0.25,
            "code": 0.25,
            "reasoning": 0.25,
            "ethics": 0.25
        })
        
        # Define threshold for expert selection
        threshold = 0.2
        
        # Example of updating priors based on prompt type
        if "solve" in args.prompt.lower() or "calculate" in args.prompt.lower():
            controller.update_priors({
                "math": 0.6,
                "code": 0.2,
                "reasoning": 0.15,
                "ethics": 0.05
            })
        elif "write code" in args.prompt.lower() or "function" in args.prompt.lower():
            controller.update_priors({
                "math": 0.2,
                "code": 0.6,
                "reasoning": 0.15,
                "ethics": 0.05
            })
        
        # Run inference
        output = run_inference(
            prompt=args.prompt,
            model=model,
            tokenizer=tokenizer,
            controller=controller,
            expert_vectors=expert_vectors,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            blending_mode=args.blending_mode,
            top_k=args.top_k,
            debug=args.debug
        )
        
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {output}")


if __name__ == "__main__":
    main()
