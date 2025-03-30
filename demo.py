#!/usr/bin/env python
"""
Demo script for the Bayesian Adaptive LLM system.

This script provides a simple interface to run inference with the Bayesian controller.
It loads pretrained expert vectors and demonstrates how the system adapts to different prompts.
"""

import os
import sys
import argparse

# Add scripts directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

# Import run_inference functionality
from run_inference import run_inference, create_model, load_expert_vectors
from dispatch.bayesian_controller import BayesianController


def main():
    """Main function for the demo."""
    parser = argparse.ArgumentParser(description="Demo for Bayesian Adaptive LLM")
    parser.add_argument("--prompt", type=str, help="Input prompt (or interactive mode if not provided)")
    parser.add_argument("--expert-dir", type=str, default="experts/vectors", help="Directory containing expert vectors")
    parser.add_argument("--config", type=str, default="config/experiments/example.json", help="Model configuration")
    parser.add_argument("--blending-mode", type=str, default="weighted_average", 
                       choices=["weighted_average", "top_k", "max"], help="Mode for blending expert vectors")
    parser.add_argument("--top-k", type=int, default=None, help="Number of top experts to consider")
    args = parser.parse_args()
    
    # Create or load model
    print("Creating model...")
    model, device = create_model(args.config)
    
    # Load expert vectors
    print("Loading expert vectors...")
    expert_vectors = load_expert_vectors(args.expert_dir)
    
    # Create Bayesian controller
    print("Initializing Bayesian controller...")
    controller = BayesianController(
        expert_names=list(expert_vectors.keys()),
        device=device
    )
    
    # Print available experts
    print("\nAvailable experts:")
    for expert_name in expert_vectors.keys():
        print(f"  - {expert_name}")
    
    if args.prompt:
        # Run inference on provided prompt
        response = run_inference(
            args.prompt,
            model,
            controller,
            expert_vectors,
            device=device,
            show_experts=True,
            blending_mode=args.blending_mode,
            top_k=args.top_k
        )
        
        # Print results
        print(f"\nPrompt: {args.prompt}")
        print(f"\nResponse: {response}")
    else:
        # Interactive mode
        print("\nEntering interactive mode. Type 'exit' or 'quit' to end the demo.")
        
        while True:
            try:
                prompt = input("\nEnter your prompt: ")
                if prompt.lower() in ['exit', 'quit']:
                    break
                
                # Run inference
                response = run_inference(
                    prompt,
                    model,
                    controller,
                    expert_vectors,
                    device=device,
                    show_experts=True,
                    blending_mode=args.blending_mode,
                    top_k=args.top_k
                )
                
                # Print response
                print(f"\nResponse: {response}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Demo ended. Thank you for trying the Bayesian Adaptive LLM system!")


if __name__ == "__main__":
    main()
