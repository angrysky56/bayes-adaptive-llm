#!/bin/bash

echo "Setting up environment for Bayesian Self-Adaptive LLM..."
echo "Creating virtual environment if it doesn't exist..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete!"
