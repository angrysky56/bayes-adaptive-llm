#!/usr/bin/env python
"""
Evaluation script for Bayesian Self-Adaptive LLM.

This script evaluates models with Bayesian adaptation on various tasks.
"""

import os
import json
import argparse
import logging
import torch
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from model.mambaformer import MambaFormer
from svf.svf import SVF
from dispatch.bayesian_controller import BayesianController


def setup_logger(log_file: Optional[str] = None):
    """Set up logger for evaluation."""
    # Create logger
    logger = logging.getLogger("evaluate")
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_model_and_experts(model_name: str, expert_names: List[str], device: str = "cuda"):
    """
    Load model and expert vectors.
    
    Args:
        model_name: Name of the model
        expert_names: List of expert names
        device: Device to load model on
        
    Returns:
        Tuple of (model, expert_vectors)
    """
    # Load model
    if model_name == "mambaformer":
        model = MambaFormer(dim=256, depth=4)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model = model.to(device)
    
    # Load expert vectors
    expert_vectors = {}
    
    for expert_name in expert_names:
        expert_path = Path("experts/vectors") / expert_name / "expert.pt"
        
        if not expert_path.exists():
            raise FileNotFoundError(f"Expert vector not found: {expert_path}")
        
        # Load expert vector
        expert_vector = torch.load(expert_path, map_location=device)
        expert_vectors[expert_name] = expert_vector
    
    return model, expert_vectors


def load_evidence_functions(config_path: str):
    """
    Load evidence functions from configuration.
    
    Args:
        config_path: Path to evidence functions configuration
        
    Returns:
        Dictionary of evidence functions
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Import evidence function module
    from dispatch.evidence import (
        KeywordEvidenceFunction,
        EmbeddingEvidenceFunction,
        create_evidence_function,
    )
    
    # Create evidence functions
    evidence_functions = {}
    
    for name, func_config in config.items():
        evidence_functions[name] = create_evidence_function(
            func_type=func_config["type"],
            config=func_config["config"],
        )
    
    return evidence_functions


def evaluate_model(
    input_text: str,
    model: torch.nn.Module,
    expert_vectors: Dict[str, torch.Tensor],
    controller: BayesianController,
    tokenizer,
    device: str = "cuda",
    max_length: int = 100,
    show_expert_activations: bool = True,
):
    """
    Evaluate model with Bayesian adaptation on input text.
    
    Args:
        input_text: Input text
        model: Model to evaluate
        expert_vectors: Dictionary of expert vectors
        controller: Bayesian controller
        tokenizer: Tokenizer
        device: Device to run inference on
        max_length: Maximum length of generated text
        show_expert_activations: Whether to show expert activations
        
    Returns:
        Generated text
    """
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # First pass: Analyze input
    with torch.no_grad():
        # Run first pass to analyze task
        outputs = model(**inputs)
    
    # Get expert activations
    expert_activations = controller.select_experts(input_text, outputs)
    
    # Combine expert vectors
    combined_expert = controller.combine_experts(expert_vectors, expert_activations)
    
    # Apply expert vector
    svf = SVF(model)
    svf.apply_expert_vector(combined_expert)
    
    # Second pass: Generate output
    with torch.no_grad():
        # Run second pass to generate output
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
        )
    
    # Decode output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Show expert activations if requested
    if show_expert_activations:
        print("\nExpert activations:")
        for expert_name, activation in expert_activations.items():
            print(f"  {expert_name}: {activation:.4f}")
    
    return output_text


def main():
    """Main function for evaluating models with Bayesian adaptation."""
    parser = argparse.ArgumentParser(description="Evaluate models with Bayesian adaptation")
    parser.add_argument("--input", type=str, required=True, help="Input text")
    parser.add_argument("--model", type=str, default="mambaformer", help="Model name")
    parser.add_argument("--experts", type=str, help="Comma-separated list of expert names")
    parser.add_argument("--evidence-config", type=str, default="config/evidence_functions.json", 
                        help="Path to evidence functions configuration")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--log-file", type=str, help="Path to log file")
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger(args.log_file)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Parse expert names
    expert_names = []
    if args.experts:
        expert_names = args.experts.split(",")
    else:
        # Use all experts if none specified
        expert_dir = Path("experts/vectors")
        expert_names = [d.name for d in expert_dir.iterdir() if d.is_dir()]
    
    logger.info(f"Using experts: {expert_names}")
    
    # Load model and expert vectors
    logger.info("Loading model and expert vectors...")
    
    try:
        # Import tokenizer
        # In a real implementation, you would use a proper tokenizer
        # For now, let's create a dummy tokenizer
        class DummyTokenizer:
            def __call__(self, text, return_tensors=None):
                return {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                }
            
            def decode(self, ids, skip_special_tokens=False):
                return "Generated text"
        
        tokenizer = DummyTokenizer()
        
        model, expert_vectors = load_model_and_experts(args.model, expert_names, device)
        
        # Load evidence functions
        logger.info("Loading evidence functions...")
        evidence_functions = load_evidence_functions(args.evidence_config)
        
        # Create Bayesian controller
        logger.info("Creating Bayesian controller...")
        controller = BayesianController(
            expert_names=expert_names,
            evidence_functions=evidence_functions,
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        output_text = evaluate_model(
            input_text=args.input,
            model=model,
            expert_vectors=expert_vectors,
            controller=controller,
            tokenizer=tokenizer,
            device=device,
            max_length=args.max_length,
        )
        
        # Print output
        print("\nInput:")
        print(args.input)
        print("\nOutput:")
        print(output_text)
        
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
