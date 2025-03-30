# Training Expert Vectors for Bayesian Self-Adaptive LLM

This document explains how to train domain-specific expert vectors using Singular Value Fine-tuning (SVF) and how to use the trained experts with the Bayesian dispatch system.

## Training Individual Experts

To train a single expert vector, use:

```bash
python -m train_eval.train --config config/experiments/<expert_config>.json
```

Where `<expert_config>` is one of:
- `math_expert.json` - For advanced mathematical problem solving
- `code_expert.json` - For code generation and algorithm implementation
- `reasoning_expert.json` - For logical reasoning and inference

## Training Multiple Experts

To train multiple experts sequentially, use:

```bash
python scripts/train_multiple_experts.py --configs math_expert.json code_expert.json reasoning_expert.json
```

Or to train all experts in the experiments directory:

```bash
python scripts/train_multiple_experts.py
```

## Monitoring Training

Each expert training run creates a log file in the `logs` directory. You can monitor training progress with:

```bash
tail -f logs/<expert_name>_training.log
```

## Using Trained Experts

After training, expert vectors are saved in `experts/vectors/<expert_name>/expert.pt`. 

To evaluate a trained expert:

```bash
python -m train_eval.evaluate --expert experts/vectors/<expert_name>/expert.pt --input "Your input prompt"
```

## Using the Bayesian Dispatch System

The Bayesian dispatch system can automatically select and blend appropriate experts based on input:

```bash
python -m train_eval.evaluate --dispatch-config config/dispatch_config.json --input "Your input prompt"
```

## Fine-tuning Recommendations

For optimal performance, consider:

1. **Specialized Datasets**: Use dedicated datasets for each domain
2. **Hyperparameter Tuning**: Adjust learning rates and KL coefficients 
3. **Model Size**: Larger models can benefit from more specialized experts
4. **Rank Selection**: Higher ranks allow more expressivity but may overfit

## Expert Vector Size Impact

SVF expert vectors are extremely parameter-efficient:
- Small model (256 dim, 4 layers): ~100KB per expert
- Medium model (512 dim, 8 layers): ~500KB per expert
- Large model (768 dim, 12 layers): ~1.5MB per expert

This allows for storing and combining many specialized experts without significant overhead.