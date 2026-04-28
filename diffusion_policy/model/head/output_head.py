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
import torch.nn.functional as F


class OutputHead(nn.Module, ABC):
    """Abstract base class for policy output heads.

    Subclasses must implement :meth:`compute_loss` and :meth:`predict`.
    Optional :meth:`forward` returns a distribution / raw tensor if a caller
    needs direct access (e.g. for entropy logging), but is not required by
    the training loop.
    """

    # Heads that consume / emit raw (un-normalized) action values set this to
    # True so the host policy skips ``LinearNormalizer`` on training target
    # and predicted action.
    outputs_raw_action: bool = False

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


class DiscreteAROutputHead(OutputHead):
    """Autoregressive discrete action head — driven by a host trunk.

    The head owns no transformer; the host policy runs its own trunk D times
    per env-step (once for ``h_obs``, then per dim with the previously-decided
    bin as input). Arm dims share ``bin_proj`` over ``num_bins`` evenly-spaced
    centers on ``[-clip_val, +clip_val]``; the gripper (last dim) uses its own
    1-logit ``gripper_proj`` with sigmoid-BCE — ``0 → -1`` (open), ``1 → +1``
    (close).
    """

    outputs_raw_action: bool = True

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int = 7,
        num_bins: int = 20,
        clip_val: float = 2.0,
        gripper_dim: int = 6,
    ):
        super().__init__(hidden_dim=hidden_dim, action_dim=action_dim)
        assert num_bins >= 2, f"num_bins must be >= 2, got {num_bins}"
        assert gripper_dim == action_dim - 1, (
            f"gripper_dim must be last; got gripper_dim={gripper_dim}, action_dim={action_dim}."
        )
        self.num_bins = int(num_bins)
        self.clip_val = float(clip_val)
        self.gripper_dim = int(gripper_dim)

        bin_centers = torch.linspace(-clip_val, clip_val, num_bins)
        self.register_buffer("arm_bin_centers", bin_centers)

        self.bin_proj = nn.Linear(hidden_dim, num_bins)
        self.gripper_proj = nn.Linear(hidden_dim, 1)

        # ``dim_embed[k]`` marks the AR input token at sequence position ``k+1``
        # (the input that predicts dim ``k+1``). Token 0 is plain ``h_obs``.
        self.bin_embed = nn.Embedding(num_bins, hidden_dim)
        self.dim_embed = nn.Embedding(action_dim - 1, hidden_dim)

    def get_spec(self) -> Dict:
        """JSON-serializable spec describing the bin vocabulary."""
        return {
            "num_bins": self.num_bins,
            "clip_val": self.clip_val,
            "arm_bin_centers": self.arm_bin_centers.detach().cpu().tolist(),
            "gripper_bins": [-1.0, 1.0],
            "action_dim": self.action_dim,
            "arm_dims": [d for d in range(self.action_dim) if d != self.gripper_dim],
            "gripper_dim": self.gripper_dim,
        }

    def decode_bins_to_action(self, indices: torch.Tensor) -> torch.Tensor:
        """``(..., D)`` indices → raw actions. Arm: bin idx; gripper: ``{0,1} → {-1,+1}``."""
        out = self.arm_bin_centers[indices]  # (..., D); gripper slot overwritten below
        gripper_cls = indices[..., self.gripper_dim]
        out[..., self.gripper_dim] = torch.where(
            gripper_cls == 0,
            torch.full_like(out[..., self.gripper_dim], -1.0),
            torch.full_like(out[..., self.gripper_dim], 1.0),
        )
        return out

    def ar_input_token(self, prev_bin: torch.Tensor, dim: int) -> torch.Tensor:
        """AR input token whose output predicts ``dim`` ∈ ``[1, D-1]``."""
        return self.bin_embed(prev_bin) + self.dim_embed.weight[dim - 1]

    def ar_input_sequence_train(self, prev_bins: torch.Tensor) -> torch.Tensor:
        """Teacher-forced AR tokens at sequence positions ``1..D-1``.
        ``prev_bins``: ``(N, D-1)`` — gt bins for dims ``0..D-2``.
        """
        D = self.action_dim
        assert prev_bins.shape[-1] == D - 1
        return self.bin_embed(prev_bins) + self.dim_embed.weight.unsqueeze(0)

    def step_inference(self, hidden: torch.Tensor, dim: int, sample: bool) -> torch.Tensor:
        """Sample / argmax the index for ``dim`` (bin idx for arm, ``{0,1}`` for gripper)."""
        if dim == self.gripper_dim:
            logit = self.gripper_proj(hidden).squeeze(-1)
            if sample:
                return torch.bernoulli(torch.sigmoid(logit)).long()
            return (logit > 0).long()
        logits = self.bin_proj(hidden)
        if sample:
            return torch.distributions.Categorical(logits=logits).sample()
        return logits.argmax(dim=-1)

    def compute_loss_from_per_dim_hidden(
        self,
        per_dim_hidden: torch.Tensor,
        target_indices: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean per-dim loss over masked positions.

        ``per_dim_hidden``: ``(B, T, D, H)`` — position ``k`` predicts dim ``k``.
        ``target_indices``: ``(B, T, D)`` — arm: bin idx; gripper: ``{0, 1}``.
        ``loss_mask``: ``(B, T)``.
        """
        B, T, D, H = per_dim_hidden.shape
        assert D == self.action_dim
        mask_flat = loss_mask.reshape(-1).bool()
        if mask_flat.sum() == 0:
            return per_dim_hidden.sum() * 0.0

        h_flat = per_dim_hidden.reshape(-1, D, H)[mask_flat]  # (N, D, H)
        target_flat = target_indices.reshape(-1, D)[mask_flat]  # (N, D)

        arm_h = h_flat[:, :-1, :]
        arm_logits = self.bin_proj(arm_h)
        arm_targets = target_flat[:, :-1]
        if D > 1:
            arm_loss = F.cross_entropy(
                arm_logits.reshape(-1, self.num_bins),
                arm_targets.reshape(-1),
            )
        else:
            arm_loss = h_flat.new_zeros(())

        gripper_logit = self.gripper_proj(h_flat[:, -1, :]).squeeze(-1)
        gripper_target = target_flat[:, -1].to(gripper_logit.dtype)
        gripper_loss = F.binary_cross_entropy_with_logits(gripper_logit, gripper_target)

        # Equivalent to mean of D per-dim means: arm dims share equal N, so
        # their D-1 means collapse into one mean over N*(D-1) samples.
        return ((D - 1) * arm_loss + gripper_loss) / D

    # OutputHead abstract interface — unused; the host policy calls
    # compute_loss_from_per_dim_hidden / step_inference directly.
    def compute_loss(self, hidden, target, loss_mask):
        raise NotImplementedError("Use compute_loss_from_per_dim_hidden via the host policy.")

    def predict(self, hidden, sample=True):
        raise NotImplementedError("Use the host policy's predict_action / kv_cached_step.")
