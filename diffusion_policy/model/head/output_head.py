"""Output head abstractions for transformer / MLP policies.

An ``OutputHead`` consumes the per-step hidden state produced by a policy
backbone (shape ``(B, T, hidden_dim)``) and is responsible for:

1. Producing an action prediction (``predict``) for rollout / evaluation.
2. Computing the training loss against a ground-truth action target
   (``compute_loss``) with a token-level ``loss_mask``.

Concrete subclasses implement different parameterizations (e.g. a Gaussian
policy with mean/log-std heads, a deterministic MLP head, a categorical /
mixture-of-experts head, etc.). The backbone policy is intentionally kept
agnostic to the choice of head so new parameterizations can be plugged in via
Hydra configuration without touching the transformer code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn


class OutputHead(nn.Module, ABC):
    """Abstract base class for policy output heads.

    Subclasses must implement :meth:`compute_loss` and :meth:`predict`.
    Optional :meth:`forward` returns a distribution / raw tensor if a caller
    needs direct access (e.g. for entropy logging), but is not required by
    the training loop.
    """

    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

    @abstractmethod
    def compute_loss(
        self,
        hidden: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a scalar training loss.

        Args:
            hidden: ``(B, T, hidden_dim)`` backbone output.
            target: ``(B, T, action_dim)`` normalized ground-truth actions.
            loss_mask: ``(B, T)`` float / bool mask. Only positions where the
                mask is non-zero contribute to the loss.
        Returns:
            Scalar loss tensor.
        """

    @abstractmethod
    def predict(
        self,
        hidden: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        """Produce a (normalized) action prediction.

        Args:
            hidden: ``(B, T, hidden_dim)`` backbone output.
            sample: when True, sample stochastically (if applicable).
                Deterministic heads may ignore this argument.
        Returns:
            ``(B, T, action_dim)`` normalized action tensor.
        """

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Default: return the deterministic prediction in a dict.

        Subclasses with a richer output (e.g. distribution parameters) should
        override this.
        """
        return {"action": self.predict(hidden, sample=False)}


class GaussianOutputHead(OutputHead):
    """Diagonal-Gaussian head parameterized by separate mean / log-std MLPs.

    Matches the original ``mean_head`` + ``log_std_head`` behavior of
    :class:`TransformerImagePolicy` so existing checkpoints and configs can be
    migrated by switching to this head explicitly.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__(hidden_dim=hidden_dim, action_dim=action_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def _distribution(self, hidden: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(
            min=self.log_std_min, max=self.log_std_max)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mean, std)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        dist = self._distribution(hidden)
        return {
            "dist": dist,
            "mean": dist.mean,
            "std": dist.stddev,
        }

    def compute_loss(
        self,
        hidden: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        dist = self._distribution(hidden)
        log_prob = dist.log_prob(target).sum(dim=-1)  # (B, T)
        mask = loss_mask.to(log_prob.dtype)
        denom = mask.sum().clamp(min=1.0)
        return -(log_prob * mask).sum() / denom

    def predict(
        self,
        hidden: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        dist = self._distribution(hidden)
        if sample:
            return dist.rsample()
        return dist.mean
