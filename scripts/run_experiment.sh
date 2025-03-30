#!/bin/bash
# run_experiment.sh - Run an experiment for Bayesian Self-Adaptive LLM
#
# This script:
# 1. Sets up the environment
# 2. Trains an expert vector
# 3. Evaluates the model with Bayesian adaptation

set -e  # Exit on error

# Parse arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <experiment_config> [--no-train] [--log-dir <dir>]"
    exit 1
fi

EXPERIMENT_CONFIG="$1"
TRAIN=true
LOG_DIR="logs/$(date +%Y-%m-%d_%H-%M-%S)"
shift

# Parse optional arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-train)
            TRAIN=false
            shift
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Directory setup
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Create log directory
mkdir -p "$LOG_DIR"

# Check if experiment config exists
if [ ! -f "$EXPERIMENT_CONFIG" ]; then
    echo "Error: Experiment config not found: $EXPERIMENT_CONFIG"
    exit 1
fi

# Extract experiment name from config
EXPERIMENT_NAME=$(basename "$EXPERIMENT_CONFIG" .json)

# Setup environment
echo "Setting up environment..."
source scripts/setup_env.sh

# Load experiment configuration
echo "Loading experiment configuration..."
CONFIG_JSON=$(cat "$EXPERIMENT_CONFIG")
EXPERT_NAME=$(echo "$CONFIG_JSON" | grep -o '"name"[[:space:]]*:[[:space:]]*"[^"]*"' | cut -d'"' -f4)
if [ -z "$EXPERT_NAME" ]; then
    echo "Error: Could not extract expert name from configuration"
    exit 1
fi

# Train expert
if [ "$TRAIN" = true ]; then
    echo "Training expert: $EXPERT_NAME..."
    python -m train_eval.train --config "$EXPERIMENT_CONFIG" --log-file "$LOG_DIR/train.log"
    echo "Training complete!"
else
    echo "Skipping training phase..."
fi

# Evaluate expert
echo "Evaluating model with Bayesian adaptation..."

# Generate a test input based on expert name
case "$EXPERT_NAME" in
    "math")
        TEST_INPUT="Solve the quadratic equation: x^2 + 5x + 6 = 0"
        ;;
    "code")
        TEST_INPUT="Write a Python function to calculate the factorial of a number."
        ;;
    "reasoning")
        TEST_INPUT="If all cats have tails, and Fluffy is a cat, what can we conclude about Fluffy?"
        ;;
    *)
        TEST_INPUT="This is a test input for the $EXPERT_NAME expert."
        ;;
esac

# Run evaluation
python -m train_eval.evaluate --input "$TEST_INPUT" --experts "$EXPERT_NAME" --log-file "$LOG_DIR/evaluate.log"

# Log experiment details
echo "Experiment: $EXPERIMENT_NAME" > "$LOG_DIR/experiment.log"
echo "Expert: $EXPERT_NAME" >> "$LOG_DIR/experiment.log"
echo "Test Input: $TEST_INPUT" >> "$LOG_DIR/experiment.log"
echo "Date: $(date)" >> "$LOG_DIR/experiment.log"

echo "Experiment complete! Logs available in: $LOG_DIR"
