# custom_loss.py

import torch
import torch.nn as nn
from typing import Dict, Optional


class PhysioFormerLoss(nn.Module):
    """
    Custom loss function for the PhysioFormer model, as described in
    the paper on Page 13, Formula (13).

    This loss is a combination of two components:
    1. A standard Cross-Entropy Loss for the classification task.
    2. A custom regularization term that penalizes the mean squared error
       of the contribution alphas from the ContribNet modules, encouraging them
       to be close to 1.
    """

    def __init__(self, lambda_reg: float = 0.1, class_weights: Optional[torch.Tensor] = None):
        """
        Args:
            lambda_reg (float): The regularization strength hyperparameter (lambda).
                                The paper does not specify a value, so 0.1 is a
                                reasonable starting point.
        """
        super(PhysioFormerLoss, self).__init__()
        self.lambda_reg = lambda_reg
        # We use PyTorch's built-in CrossEntropyLoss, which combines LogSoftmax and NLLLoss
        # for better numerical stability.
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, model_output: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """
        Calculates the total loss.

        Args:
            model_output (Dict[str, torch.Tensor]): The output dictionary from the PhysioFormer model.
                                                    It must contain:
                                                    - "logits": The final classification logits.
                                                    - "alphas": A dictionary of contribution scores from each ContribNet.
            labels (torch.Tensor): The ground truth labels for the classification task.

        Returns:
            torch.Tensor: The final, combined loss value.
        """
        # --- 1. Calculate Cross-Entropy Loss ---
        logits = model_output["logits"]
        loss_ce = self.cross_entropy_loss(logits, labels)

        # --- 2. Calculate Regularization Loss ---
        alphas_dict = model_output["alphas"]

        # Check if there are any alphas to regularize
        if not alphas_dict or self.lambda_reg == 0:
            return loss_ce

        all_alphas = []
        # Collect all alpha tensors from the dictionary
        for key in alphas_dict:
            all_alphas.append(alphas_dict[key])

        # Concatenate all alpha tensors into a single tensor
        # This handles cases with varying numbers of signals (Wrist vs. Chest)
        alpha_tensor = torch.cat(all_alphas, dim=1)  # Shape: (batch_size, num_signals)

        # Calculate the Mean Squared Error between alphas and 1.0
        # This corresponds to the (alpha_bj - 1)^2 part of the formula
        loss_reg = torch.mean((alpha_tensor - 1.0) ** 2)

        # --- 3. Combine the two losses ---
        total_loss = loss_ce + self.lambda_reg * loss_reg

        return total_loss