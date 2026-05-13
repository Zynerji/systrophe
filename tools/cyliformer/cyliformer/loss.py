"""Cyliformer training losses.

Two flavours:

  - `cyliformer_loss`: function CE + lambda_2 floor penalty.
  - `TorsionalResonanceLoss`: full hybrid for torsional-fine-tune use,
    with CE + lambda_2 floor + derivative-smoothness + beam-alignment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def cyliformer_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    lambda2_per_layer_per_cylinder: list[list[float]],
    lambda_target: float = 0.20,
    lambda_weight: float = 0.4,
    ignore_index: int = -100,
) -> torch.Tensor:
    """CE loss + lambda_2 floor penalty.

    The penalty is `relu(lambda_target - lambda_2)^2`: zero when the
    catcher signal at least matches the target, positive (penalising)
    when it falls below.
    """
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )
    flat_lambdas = [v for layer in lambda2_per_layer_per_cylinder for v in layer]
    if not flat_lambdas:
        return ce
    lam = torch.tensor(flat_lambdas, dtype=ce.dtype, device=ce.device)
    floor_pen = F.relu(float(lambda_target) - lam).pow(2).mean()
    return ce + float(lambda_weight) * floor_pen


class TorsionalResonanceLoss(nn.Module):
    """Hybrid loss for torsional-style Cyliformer training.

    L = CE + alpha * floor(lambda_target - lambda_2)^2
        + beta * mean(|d lambda_2 / d step|)^2   [smoothness across steps]
        + gamma * (1 - beam_gain)^2               [beam alignment]

    The derivative and beam-gain terms encourage the cylinder phasors
    and catcher to evolve smoothly and to favour constructive
    interference. See Cyliformer.txt for the design motivation.
    """

    def __init__(
        self,
        alpha: float = 0.35,
        beta: float = 0.15,
        gamma: float = 0.20,
        lambda_target: float = 0.20,
        ignore_index: int = -100,
        derivative_window: int = 4,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.lambda_target = float(lambda_target)
        self.ignore_index = int(ignore_index)
        self.derivative_window = int(derivative_window)
        # Rolling history of mean(lambda_2) per step
        self.lambda_history: list[float] = []

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lambda2_per_layer_per_cylinder: list[list[float]],
        beam_gains: list[float],
    ) -> torch.Tensor:
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=self.ignore_index,
        )
        flat_lambdas = [v for layer in lambda2_per_layer_per_cylinder for v in layer]
        device = ce.device
        dtype = ce.dtype

        if not flat_lambdas:
            return ce

        lam_tensor = torch.tensor(flat_lambdas, dtype=dtype, device=device)
        floor = F.relu(self.lambda_target - lam_tensor).pow(2).mean()

        # Derivative smoothness across steps
        mean_lambda_now = float(lam_tensor.mean().item())
        self.lambda_history.append(mean_lambda_now)
        if len(self.lambda_history) > self.derivative_window:
            self.lambda_history = self.lambda_history[-self.derivative_window:]
        if len(self.lambda_history) >= 2:
            diffs = [
                self.lambda_history[i] - self.lambda_history[i - 1]
                for i in range(1, len(self.lambda_history))
            ]
            deriv_loss = torch.tensor(
                sum(d ** 2 for d in diffs) / len(diffs),
                dtype=dtype, device=device,
            )
        else:
            deriv_loss = torch.zeros((), dtype=dtype, device=device)

        # Beam alignment
        if beam_gains:
            beam_tensor = torch.tensor(beam_gains, dtype=dtype, device=device)
            beam_loss = (1.0 - beam_tensor).pow(2).mean()
        else:
            beam_loss = torch.zeros((), dtype=dtype, device=device)

        return ce + self.alpha * floor + self.beta * deriv_loss + self.gamma * beam_loss
