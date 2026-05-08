from typing import Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from torch.distributions import Normal

class MLPImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict[str, Any],
            obs_encoder: MultiImageObsEncoder,
            n_action_steps: int,
            n_obs_steps: int,
            hidden_dim: int = 512,
            hidden_depth: int = 4,
            **kwargs):
        assert n_action_steps == 1, "MLPImagePolicy only supports n_action_steps=1"
        
        super().__init__()
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]
        self.obs_encoder = obs_encoder
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.normalizer = LinearNormalizer()
        self.kwargs = kwargs
        
        # Input: all obs steps concatenated
        input_dim = obs_feature_dim * n_obs_steps
        
        # Shared trunk
        layers = []
        last_dim = input_dim
        for _ in range(hidden_depth):
            layers += [nn.Linear(last_dim, hidden_dim), nn.ReLU()]
            last_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        
        # Separate heads for mean and log std
        self.mean_head = nn.Linear(last_dim, action_dim)
        self.log_std_head = nn.Linear(last_dim, action_dim)
        
        self.log_std_limits = (-5.0, 2.0)
        # Honor DiffusionPolicyWrapper.sample_action: True (collection) → rsample,
        # False (eval) → distribution mean. Default True so collection inherits the
        # historical stochastic behavior; the wrapper flips this to False at eval.
        self.sample_action = True

    def forward(self, obs_features: torch.Tensor) -> Normal:
        """
        Returns a Normal(mean, std) distribution over actions given observation features.
        """
        h = self.trunk(obs_features)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(min=self.log_std_limits[0], max=self.log_std_limits[1])
        std = torch.exp(log_std)
        return Normal(mean, std)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        # The shared workspace adds context keys (attention_mask, action, reward) for the
        # transformer policy; drop them here so the MLP normalizer only sees obs keys.
        obs_dict = {k: v for k, v in obs_dict.items() if k in self.normalizer.params_dict}
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        To = self.n_obs_steps
        device = self.device
        dtype = self.dtype
        # Encode obs: flatten all obs steps
        if isinstance(nobs, dict):
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        else:
            this_nobs = nobs[:,:To,...].reshape(-1,*nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(B, To, -1)
        mlp_input = nobs_features.reshape(B, -1)
        
        # Get action distribution
        dist = self.forward(mlp_input)
        action_pred = dist.rsample() if self.sample_action else dist.mean
        action = self.normalizer['action'].unnormalize(action_pred)
        return {
            'action': action,
            'action_pred': action_pred
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        B = nactions.shape[0]
        T = nactions.shape[1]
        Ta = self.n_action_steps
        Da = self.action_dim
        assert Ta == 1, "MLPImagePolicy only supports n_action_steps=1"

        # Encode obs at every (batch, timestep) — markovian per-step prediction. The
        # earlier implementation only used (o_0, a_0) per episode, which trained the
        # MLP only on starting frames and led to catastrophic OOD failure at inference.
        if isinstance(nobs, dict):
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))  # (B*T, ...)
        else:
            this_nobs = nobs.reshape(-1, *nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs)  # (B*T, feat)

        # Per-step Gaussian over all (B, T) action targets; loss is masked to the
        # expert + attention timesteps so padding / non-expert steps don't contribute.
        dist = self.forward(nobs_features)  # Normal((B*T, Da), (B*T, Da))
        target = nactions.reshape(-1, Da)  # (B*T, Da)
        log_prob = dist.log_prob(target).sum(dim=-1).reshape(B, T)  # (B, T)

        expert_mask = batch['expert_mask']
        loss_mask = expert_mask[..., 0] if expert_mask.dim() == 3 else expert_mask
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            loss_mask = loss_mask * attention_mask

        denom = loss_mask.sum().clamp_min(1.0)
        return -(log_prob * loss_mask).sum() / denom
