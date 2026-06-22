# VENDORED from UWLab-patrick-private/scripts/imitation_learning/point_cloud/residual_point_net.py
# Verbatim copy — keep in sync with upstream if the BC PointNet model/checkpoint format changes.
# Deployed checkpoint: pnocc_xl_residual_big_ee (residual_point_net, point_dim=4, proprio_dim=18, action_dim=7).

import torch
import torch.nn as nn

from point_net import PointNet


class ResidualMLP(nn.Module):
    """Drop-in replacement for ``torchvision.ops.MLP`` with per-layer residual connections.

    Each hidden layer is ``relu(norm(Linear(x)))``, with an identity skip added *only* when
    ``d_in == d_out`` (no projection shortcut -- width-changing layers stay plain, so no extra
    parameters are spent where a residual can't apply cleanly). Operates on the last dim, so it
    works on both per-point ``(B, N, C)`` and pooled ``(B, C)`` tensors, exactly like the MLPs it
    replaces in :class:`PointNet`.

    The final entry of ``hidden_channels`` is treated as the output projection: a plain ``Linear``
    with no norm / activation / residual (matching how ``torchvision.ops.MLP`` ends and how the
    PointNet action head emits raw action logits).
    """

    def __init__(self, in_channels, hidden_channels, norm_layer=nn.LayerNorm, activation=nn.ReLU):
        super().__init__()
        dims = [in_channels, *hidden_channels]
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.has_skip = []  # identity residual only where input/output widths match
        self.n_hidden = len(hidden_channels) - 1  # all but the output projection get norm + activation
        for i in range(len(hidden_channels)):
            d_in, d_out = dims[i], dims[i + 1]
            self.linears.append(nn.Linear(d_in, d_out))
            if i < self.n_hidden:
                self.norms.append(norm_layer(d_out))
                self.has_skip.append(d_in == d_out)
        self.act = activation()

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            if i < self.n_hidden:
                h = self.act(self.norms[i](linear(x)))
                x = h + x if self.has_skip[i] else h
            else:
                x = linear(x)  # output projection: raw, no residual
        return x


class ResidualPointNet(PointNet):
    """:class:`PointNet` with residual connections in the set encoder and action head.

    Identical to :class:`PointNet` (same forward, pooling, proprio fusion, loss) -- only the two deep
    MLPs are swapped for :class:`ResidualMLP`. Motivation: as the encoder/head grow deep, residual
    connections ease optimization. ``point_norm`` and ``proprio_encoder`` are shallow and left as-is.
    """

    def __init__(self, encoder_hidden_dims, action_hidden_dims, proprio_dim, action_dim, predict_std, point_dim=3):
        super().__init__(encoder_hidden_dims, action_hidden_dims, proprio_dim, action_dim, predict_std, point_dim)
        self.set_encoder = ResidualMLP(in_channels=point_dim, hidden_channels=encoder_hidden_dims)
        out_dim = action_dim * 2 if predict_std else action_dim
        self.action_head = ResidualMLP(in_channels=encoder_hidden_dims[-1] * 2, hidden_channels=[*action_hidden_dims, out_dim])
