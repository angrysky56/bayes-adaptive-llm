# Bayesian Self-Adaptive LLM

A research mock prototype exploring Bayesian adaptation in large language models using MambaFormer and Singular Value Fine-tuning (SVF).

See [Development](https://github.com/angrysky56/bayes-adaptive-llm/blob/main/to_develop.md) for steps to create a actual prototype.

## Overview

This project implements a self-adaptive language model system that combines:

1. **MambaFormer Architecture**: A hybrid architecture combining Mamba (State Space Models) with Transformer attention blocks
2. **Singular Value Fine-Tuning (SVF)**: A parameter-efficient method that manipulates only the singular values of weight matrices
3. **Bayesian Expert Dispatch**: A probabilistic framework to select and combine domain-specific experts at inference time

The model dynamically adapts to tasks by employing a two-pass inference process:

- First pass: Analyze the input and determine task characteristics
- Second pass: Apply task-specific adaptations and generate the response

## Project Structure

- `model/`: MambaFormer architecture implementation
- `svf/`: Singular Value Fine-tuning implementation
- `experts/`: Expert vector training and management
- `dispatch/`: Bayesian controller for expert selection
- `tasks/`: Benchmark tasks for evaluation
- `train_eval/`: Training and evaluation utilities
- `scripts/`: Utility scripts
- `utils/`: Helper functions and utilities

## Getting Started

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd bayes-adaptive-llm

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Tests

The project includes several test scripts to verify that components are working correctly:

```bash
# Test the Bayesian Controller
python scripts/test_bayesian_controller.py

# Test the MambaFormer model
python scripts/test_mambaformer.py

# Run all tests
bash scripts/run_tests.sh
```

### Training Expert Vectors

To train domain-specific expert vectors using SVF:

```bash
# Train a single expert
python train_eval/train.py --config config/experiments/math_expert.json

# Train multiple experts sequentially
python scripts/train_multiple_experts.py --config-dir config/experiments
```

### Running Demos

The project includes demo scripts to showcase the adaptive capabilities:

```bash
# Run the interactive demo
python demo.py

# Try with specific prompts
python demo.py --prompt "Solve the quadratic equation x^2 - 5x + 6 = 0"
python demo.py --prompt "Write a function to calculate factorial in Python"
python demo.py --prompt "Is it ethically acceptable to use AI for surveillance?"

# Advanced inference with options
python scripts/run_inference.py --input "Analyze the complexity of quicksort" --show-experts --blending-mode weighted_average
```

### Expert Blending Modes

The system supports several modes for blending expert vectors:

- `weighted_average`: Blend experts based on their posterior probabilities
- `top_k`: Use only the top-k experts and renormalize their weights
- `max`: Use only the expert with the highest probability

Example:
```bash
python demo.py --prompt "What is the time complexity of quicksort?" --blending-mode top_k --top-k 2
```

## Development

### Directory Structure

```
bayes-adaptive-llm/
├── config/             # Configuration files
│   └── experiments/    # Experiment-specific configs
├── data/               # Training and evaluation data
│   ├── raw/            # Raw datasets
│   └── processed/      # Processed data for training
├── dispatch/           # Bayesian expert dispatch
├── experts/            # Expert vector management
│   └── vectors/        # Saved expert vectors
├── model/              # Model architecture implementation
├── scripts/            # Utility scripts
├── svf/                # Singular Value Fine-tuning
├── tasks/              # Benchmark tasks
├── train_eval/         # Training and evaluation code
└── utils/              # Helper functions
```

### Code Organization

- `bayesian_controller.py`: Implements Bayesian expert selection and blending
- `mambaformer.py`: Implements the hybrid Mamba-Transformer architecture
- `svf.py`: Implements Singular Value Fine-tuning for expert adaptation

## References

This project builds upon several key research works:

1. Mamba: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
2. MambaFormer: [Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks](https://arxiv.org/abs/2402.04248)
3. SVF: [Transformer²: Self-adaptive LLMs](https://arxiv.org/abs/2501.06252)
4. [SakanaAI/self-adaptive-llms](https://github.com/SakanaAI/self-adaptive-llms)

## License

[MIT License](LICENSE)
