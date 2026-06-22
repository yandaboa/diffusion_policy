# VENDORED from UWLab-patrick-private/scripts/imitation_learning/point_cloud/point_net.py
# Verbatim copy — keep in sync with upstream if the BC PointNet model/checkpoint format changes.
# Deployed checkpoint: pnocc_xl_residual_big_ee (residual_point_net, point_dim=4, proprio_dim=18, action_dim=7).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP

class PointNet(nn.Module):
    def __init__(self, encoder_hidden_dims, action_hidden_dims, proprio_dim, action_dim, predict_std, point_dim=3):
        super().__init__()
        self.predict_std = predict_std
        self.point_dim = point_dim  # 3 = xyz; 4 = xyz + per-point segmentation label
        self.pooled_dim = encoder_hidden_dims[-1]  # width of the pooled global feature (aux-probe input)
        self.set_encoder = MLP(
            in_channels=point_dim,
            hidden_channels=encoder_hidden_dims,
            norm_layer=nn.LayerNorm,
        )
        self.point_norm = torch.nn.LayerNorm(encoder_hidden_dims[-1])
        self.proprio_encoder = torch.nn.Sequential(torch.nn.Linear(proprio_dim, encoder_hidden_dims[-1]), torch.nn.LayerNorm(encoder_hidden_dims[-1]))
        out_dim = action_dim * 2 if predict_std else action_dim
        self.action_head = MLP(
            in_channels=encoder_hidden_dims[-1]*2,
            hidden_channels=[*action_hidden_dims, out_dim],
        )

    def forward(self, x, proprio, return_pooled=False):  # x: (B, N, point_dim)
        assert x.ndim == 3 and x.shape[-1] == self.point_dim
        assert proprio.ndim == 2
        feats = self.set_encoder(x)
        pooled = self.point_norm(feats.amax(dim=1))      # (B, pooled_dim) global PC feature
        proprio_feats = self.proprio_encoder(proprio)
        feats = torch.cat([pooled, proprio_feats], dim=-1)
        out = self.action_head(feats)
        if self.predict_std:
            out = out.chunk(2, dim=-1)                    # (mean, log_std)
        # return_pooled exposes the pooled feature for an auxiliary probe (e.g. object-pose head).
        return (out, pooled) if return_pooled else out

    def calculate_loss(self, points, proprio, targets):
        if self.predict_std:
            mean, log_std = self.forward(points, proprio)
            var = torch.exp(2 * log_std)      # std = exp(log_std) > 0, so var > 0
            return F.gaussian_nll_loss(mean, targets, var)
        return F.mse_loss(self.forward(points, proprio), targets)