"""
Example usage of SVF for fine-tuning a model

This script demonstrates how to use the SVF module to create and apply experts
to a model.
"""

import torch
import torch.nn as nn

from svf import SVFFinetuner

# Add the SVD capability to a linear layer
class SVDLinear(nn.Linear):
    """Linear layer with SVD capabilities for SVF"""
    
    def get_singular_values(self) -> torch.Tensor:
        """Get singular values of the weight matrix"""
        _, S, _ = self.get_svd_components()
        return S
    
    def get_svd_components(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get SVD components of the weight matrix"""
        from ..utils.svd import apply_svd
        return apply_svd(self.weight)
    
    def update_weights(self, modified_S: torch.Tensor) -> None:
        """Update weights using modified singular values"""
        from ..utils.svd import update_with_svf
        U, S, V = self.get_svd_components()
        self.weight.data = update_with_svf(self.weight, U, S, V, modified_S)


# Create a simple model with SVD-capable layers
class SimpleMLP(nn.Module):
    """Simple MLP with SVD-capable layers"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = SVDLinear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = SVDLinear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def main():
    # Create a model
    model = SimpleMLP(10, 20, 5)
    
    # Create SVF finetuner
    finetuner = SVFFinetuner(model)
    
    # Create experts
    math_expert = finetuner.create_expert("math")
    code_expert = finetuner.create_expert("code")
    
    # Train the experts (in a real scenario)
    # This would involve optimizing the expert vectors using task-specific data
    
    # Apply a single expert
    finetuner.apply_to_model(expert_name="math")
    
    # Or blend multiple experts
    expert_weights = {"math": 0.7, "code": 0.3}
    finetuner.apply_to_model(expert_weights=expert_weights)
    
    # Save and load experts
    finetuner.save_expert("math", "math_expert.pt")
    finetuner.load_expert("math_loaded", "math_expert.pt")


if __name__ == "__main__":
    main()
