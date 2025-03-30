#!/bin/bash
# Demo script for training and evaluating Bayesian Self-Adaptive LLM

set -e  # Exit on error

echo "======== Bayesian Self-Adaptive LLM Demo ========"
echo "This script demonstrates training and inference with SVF experts"

# Create necessary directories
mkdir -p logs
mkdir -p experts/vectors

# Step 1: Train the math expert
echo ""
echo "Step 1: Training math expert..."
echo "================================"
python -m train_eval.train --config config/experiments/math_expert.json --log-file logs/math_training.log

# Step 2: Train the code expert
echo ""
echo "Step 2: Training code expert..."
echo "==============================="
python -m train_eval.train --config config/experiments/code_expert.json --log-file logs/code_training.log

# Step 3: Train the reasoning expert
echo ""
echo "Step 3: Training reasoning expert..."
echo "==================================="
python -m train_eval.train --config config/experiments/reasoning_expert.json --log-file logs/reasoning_training.log

# Step 4: Run inference with Bayesian dispatch
echo ""
echo "Step 4: Running inference with Bayesian dispatch..."
echo "=================================================="
python scripts/run_inference.py --input "Solve for x in the equation: x^2 + 5x + 6 = 0" --show-experts

# Step 5: Try another example
echo ""
echo "Step 5: Another example (code task)..."
echo "======================================"
python scripts/run_inference.py --input "Write a function to find the longest common subsequence of two strings." --show-experts

# Step 6: One more example
echo ""
echo "Step 6: Final example (reasoning task)..."
echo "========================================="
python scripts/run_inference.py --input "If all A are B, and some C are not B, what can we conclude about the relationship between A and C?" --show-experts

echo ""
echo "Demo complete!"
echo "=============="
echo "Expert vectors are saved in experts/vectors/"
echo "You can run additional inferences using scripts/run_inference.py"
