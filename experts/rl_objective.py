"""
RL-based training objective for expert vector optimization.

This module implements reinforcement learning techniques for training
SVF expert vectors, following the approach described in the Transformer²
paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Union, Any


class RLObjective:
    """Base class for RL-based training objectives."""
    
    def __init__(
        self,
        kl_coef: float = 0.1,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
    ):
        """
        Initialize the RL objective.
        
        Args:
            kl_coef: KL divergence coefficient for regularization
            gamma: Discount factor
            eps_clip: Clipping parameter for PPO
        """
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.eps_clip = eps_clip
    
    def compute_rewards(
        self, 
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: torch.Tensor,
        reward_fn: Callable
    ) -> torch.Tensor:
        """
        Compute rewards for each output.
        
        Args:
            outputs: Model outputs (tensor or dictionary with 'logits' key)
            targets: Target values
            reward_fn: Function that computes rewards
            
        Returns:
            Tensor of rewards
        """
        rewards = reward_fn(outputs, targets)
        # Make sure rewards is not empty and is properly shaped
        if rewards.numel() == 0:
            print("WARNING: Empty rewards tensor detected")
            rewards = torch.zeros(targets.size(0), device=targets.device)
        elif rewards.dim() == 0:
            # Expand scalar reward to batch size
            rewards = rewards.expand(targets.size(0))
        
        return rewards
    
    def compute_loss(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        rewards: torch.Tensor,
        advantages: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the PPO loss.
        
        Args:
            logprobs: Log probabilities of current policy
            old_logprobs: Log probabilities of old policy
            rewards: Rewards
            advantages: Advantages (optional)
            
        Returns:
            PPO loss
        """
        if advantages is None:
            advantages = rewards
            
        # Compute ratio
        ratio = torch.exp(logprobs - old_logprobs.detach())
        
        # Compute surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
        
        # Compute PPO loss
        ppo_loss = -torch.min(surr1, surr2).mean()
        
        return ppo_loss


class SVFReinforce(RLObjective):
    """
    REINFORCE algorithm for training SVF expert vectors, as described
    in the Transformer² paper.
    """
    
    def __init__(
        self,
        kl_coef: float = 0.1,
        baseline: Optional[Callable] = None,
    ):
        """
        Initialize the REINFORCE objective.
        
        Args:
            kl_coef: KL divergence coefficient for regularization
            baseline: Optional baseline function for variance reduction
        """
        super().__init__(kl_coef=kl_coef)
        self.baseline = baseline
    
    def compute_loss(
        self,
        model: nn.Module,
        base_model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        outputs: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the REINFORCE loss with KL regularization.
        
        Args:
            model: Model with expert vectors
            base_model: Base model without expert vectors
            inputs: Input tensors
            outputs: Model outputs
            rewards: Rewards
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Get device
        device = rewards.device
        
        # Check if model has get_logprobs method
        if not hasattr(model, 'get_logprobs'):
            print(f"WARNING: Model {type(model).__name__} does not have get_logprobs method.")
            print("Implementing a fallback method to estimate log probabilities.")
            
            def fallback_get_logprobs(model, inputs, outputs):
                # Extract logits - using our helper function
                logits = get_logits_from_outputs(outputs, targets=None)
                
                if logits.dim() <= 1:
                    print("WARNING: Logits have incorrect shape - cannot compute log probabilities")
                    return torch.zeros(rewards.size(0) if rewards.dim() > 0 else 1, device=device)
                    
                # Get last token predictions
                if logits.dim() >= 3:
                    last_logits = logits[:, -1, :]
                else:
                    last_logits = logits
                
                # Apply log softmax
                log_probs = F.log_softmax(last_logits, dim=-1)
                
                # Get highest probability as fallback
                token_log_probs = log_probs.max(dim=-1)[0]
                
                return token_log_probs
                
            # Use fallback method
            try:
                logprobs = fallback_get_logprobs(model, inputs, outputs)
            except Exception as e:
                print(f"Error in fallback_get_logprobs: {e}")
                logprobs = torch.zeros(rewards.size(0) if rewards.dim() > 0 else 1, device=device)
        else:
            # Use model's get_logprobs method
            try:
                logprobs = model.get_logprobs(inputs, outputs)
            except Exception as e:
                print(f"Error in model.get_logprobs: {e}")
                logprobs = torch.zeros(rewards.size(0) if rewards.dim() > 0 else 1, device=device)
        
        # Compute baseline if available
        if self.baseline is not None:
            baseline_value = self.baseline(inputs)
            advantages = rewards - baseline_value
        else:
            advantages = rewards
            
        # Compute policy gradient loss
        pg_loss = -(logprobs * advantages).mean()
        
        # Compute KL divergence for regularization
        try:
            with torch.no_grad():
                base_outputs = base_model(**inputs)
                
                if not hasattr(base_model, 'get_logprobs'):
                    # Use the same fallback method
                    base_logprobs = fallback_get_logprobs(base_model, inputs, base_outputs)
                else:
                    base_logprobs = base_model.get_logprobs(inputs, base_outputs)
            
            # Properly compute KL divergence using PyTorch's F.kl_div
            if isinstance(outputs, dict) and 'logits' in outputs and isinstance(base_outputs, dict) and 'logits' in base_outputs:
                # For models that return a dictionary with logits
                log_probs = F.log_softmax(outputs['logits'], dim=-1)
                base_probs = F.softmax(base_outputs['logits'], dim=-1)
                kl_div = F.kl_div(
                    log_probs,
                    base_probs,
                    reduction="batchmean",
                    log_target=False
                )
            elif hasattr(outputs, 'logits') and hasattr(base_outputs, 'logits'):
                # For models that return a structured output with logits
                log_probs = F.log_softmax(outputs.logits, dim=-1)
                base_probs = F.softmax(base_outputs.logits, dim=-1)
                kl_div = F.kl_div(
                    log_probs,
                    base_probs,
                    reduction="batchmean",
                    log_target=False
                )
            else:
                # Fallback to simpler KL calculation if we only have logprobs
                kl_div = (logprobs - base_logprobs).mean()
        except Exception as e:
            print(f"Error computing KL divergence: {e}")
            kl_div = torch.tensor(0.0, device=device)
        
        # Compute total loss
        total_loss = pg_loss + self.kl_coef * kl_div
        
        # Return loss and metrics
        metrics = {
            "pg_loss": pg_loss.item(),
            "kl_div": kl_div.item() if isinstance(kl_div, torch.Tensor) else 0.0,
            "total_loss": total_loss.item(),
            "mean_reward": rewards.mean().item(),
        }
        
        return total_loss, metrics


def get_logits_from_outputs(outputs, targets=None):
    """Extract logits from model outputs of various structures."""
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    elif isinstance(outputs, torch.Tensor):
        return outputs
    elif hasattr(outputs, "logits"):
        return outputs.logits
    elif hasattr(outputs, "last_hidden_state"):
        # For models that return only hidden states, we use the last hidden state
        # This is common in causal LMs where the final hidden state is used for prediction
        return outputs.last_hidden_state
    else:
        # Try to handle any other output structure with minimal logging
        if isinstance(outputs, dict):
            # Try to find any tensor in the dictionary that could serve as logits
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and value.dim() > 1:
                    return value
        elif not isinstance(outputs, torch.Tensor) and hasattr(outputs, "__dict__"):
            # Attempt to find any tensor that looks like logits
            for attr in dir(outputs):
                if not attr.startswith("_") and not callable(getattr(outputs, attr)):
                    try:
                        value = getattr(outputs, attr)
                        if isinstance(value, torch.Tensor) and value.dim() > 1 and value.size(-1) > 1:
                            return value
                    except:
                        pass
        
        # Default fallback - use targets for device if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if targets is not None and hasattr(targets, 'device'):
            device = targets.device
        
        # Return a dummy tensor with the right device - silent to avoid log spam
        return torch.zeros(1, device=device)

# Registry of reward functions
def compute_accuracy_reward(outputs, targets):
    """
    Compute accuracy reward, handling various shapes of logits and targets.
    
    Args:
        outputs: Model outputs
        targets: Target values
        
    Returns:
        Accuracy reward tensor
    """
    logits = get_logits_from_outputs(outputs, targets)
    
    if logits.dim() <= 1:
        print(f"WARNING: Logits dimension too low: {logits.dim()}")
        return torch.zeros_like(targets, dtype=torch.float)
    
    # For sequence output (batch_size, seq_len, vocab_size), use only last token prediction
    if logits.dim() == 3:
        # Detailed shape information
        print(f"DEBUG: Sequence logits shape: {logits.shape}")
        
        # Get last token prediction for each sequence
        try:
            last_token_logits = logits[:, -1, :]
            print(f"DEBUG: Last token logits shape: {last_token_logits.shape}")
            
            last_token_predictions = last_token_logits.argmax(dim=-1)
            print(f"DEBUG: Last predictions shape: {last_token_predictions.shape}, targets: {targets.shape}")
            print(f"DEBUG: Sample predictions: {last_token_predictions[:5]}, targets: {targets[:5]}")
            
            # Make sure predictions and targets are compatible
            return (last_token_predictions == targets).float()
        except Exception as e:
            print(f"ERROR in last token processing: {e}")
            return torch.zeros_like(targets, dtype=torch.float)
    
    # For simpler output (batch_size, vocab_size), compare directly
    try:
        predictions = logits.argmax(dim=-1)
        print(f"DEBUG: Predictions shape: {predictions.shape}, targets: {targets.shape}")
        
        if predictions.shape == targets.shape:
            return (predictions == targets).float()
    except Exception as e:
        print(f"ERROR in prediction comparison: {e}")
    
    # Handle dimension mismatch by returning zeros
    print(f"WARNING: Shape mismatch in accuracy calculation - logits: {logits.shape}, targets: {targets.shape}")
    return torch.zeros_like(targets, dtype=torch.float)

REWARD_FUNCTIONS = {
    "accuracy": compute_accuracy_reward,
    "f1_score": lambda outputs, targets: f1_score(get_logits_from_outputs(outputs, targets).argmax(dim=-1), targets),
    "exact_match": lambda outputs, targets: (get_logits_from_outputs(outputs, targets) == targets).all(dim=-1).float(),
}


def f1_score(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute F1 score.
    
    Args:
        preds: Predictions
        targets: Targets
        
    Returns:
        F1 score
    """
    # Convert to binary predictions if needed
    if preds.dim() > 1 and preds.shape[-1] > 1:
        preds = preds.argmax(dim=-1)
    
    # Compute true positives, false positives, false negatives
    tp = ((preds == 1) & (targets == 1)).float().sum(dim=-1)
    fp = ((preds == 1) & (targets == 0)).float().sum(dim=-1)
    fn = ((preds == 0) & (targets == 1)).float().sum(dim=-1)
    
    # Compute precision and recall
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    # Compute F1 score
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return f1


def get_reward_function(name: str) -> Callable:
    """
    Get reward function by name.
    
    Args:
        name: Name of the reward function
        
    Returns:
        Reward function
    """
    if name not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward function: {name}")
    
    return REWARD_FUNCTIONS[name]


def create_rl_objective(
    objective_type: str,
    **kwargs
) -> RLObjective:
    """
    Create an RL objective by name.
    
    Args:
        objective_type: Type of RL objective
        **kwargs: Additional arguments for the objective
        
    Returns:
        RL objective
    """
    if objective_type.lower() == "reinforce":
        return SVFReinforce(**kwargs)
    else:
        raise ValueError(f"Unknown RL objective type: {objective_type}")
