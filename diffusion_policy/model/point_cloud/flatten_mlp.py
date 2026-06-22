# VENDORED from UWLab-patrick-private/scripts/imitation_learning/point_cloud/flatten_mlp.py
# Verbatim copy — keep in sync with upstream if the BC PointNet model/checkpoint format changes.
# Deployed checkpoint: pnocc_xl_residual_big_ee (residual_point_net, point_dim=4, proprio_dim=18, action_dim=7).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP


class FlattenMLP(nn.Module):
    """BC policy: flatten the point cloud, encode with one MLP, concat raw proprio, predict action."""

    def __init__(
        self,
        encoder_hidden_dims,
        action_hidden_dims,
        proprio_dim,
        action_dim,
        predict_std,
        num_points,
        point_dim=3,
    ):
        super().__init__()
        self.predict_std = predict_std
        self.point_dim = point_dim
        self.num_points = num_points
        self.pooled_dim = encoder_hidden_dims[-1]  # encoder output width (aux-probe input)
        self.pc_encoder = MLP(
            in_channels=num_points * point_dim,
            hidden_channels=encoder_hidden_dims,
            norm_layer=nn.LayerNorm,
        )
        out_dim = action_dim * 2 if predict_std else action_dim
        self.action_head = MLP(
            in_channels=encoder_hidden_dims[-1] + proprio_dim,
            hidden_channels=[*action_hidden_dims, out_dim],
        )

    def forward(self, x, proprio, return_pooled=False):  # x: (B, N, point_dim)
        assert x.ndim == 3 and x.shape[-1] == self.point_dim and x.shape[1] == self.num_points
        assert proprio.ndim == 2
        pooled = self.pc_encoder(x.reshape(x.shape[0], -1))  # (B, pooled_dim) encoded cloud
        feats = torch.cat([pooled, proprio], dim=-1)
        out = self.action_head(feats)
        if self.predict_std:
            out = out.chunk(2, dim=-1)                        # (mean, log_std)
        return (out, pooled) if return_pooled else out

    def calculate_loss(self, points, proprio, targets):
        if self.predict_std:
            mean, log_std = self.forward(points, proprio)
            var = torch.exp(2 * log_std)
            return F.gaussian_nll_loss(mean, targets, var)
        return F.mse_loss(self.forward(points, proprio), targets)
