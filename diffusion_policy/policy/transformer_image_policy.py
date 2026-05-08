from typing import Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from torch.distributions import Normal

from transformers import GPT2Config, GPT2Model

class TransformerImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict[str, Any],
            obs_encoder: MultiImageObsEncoder,
            n_action_steps: int,
            n_obs_steps: int,
            hidden_dim: int = 512,
            hidden_depth: int = 4,
            n_head: int = 8,
            dropout: float = 0.1,
            sample_timesteps: bool = False,
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

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
        )
        # Shared trunk
        cfg = GPT2Config(
            n_positions=1024,
            n_embd=hidden_dim,
            n_layer=hidden_depth,
            n_head=n_head,
            resid_pdrop=float(dropout),
            embd_pdrop=float(dropout),
            attn_pdrop=float(dropout),
        )
        self.transformer = GPT2Model(cfg)

        # Separate heads for mean and log std
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        self.log_std_limits = (-5.0, 2.0)
        self.sample_timesteps = sample_timesteps

    def forward(self, obs_features: torch.Tensor, attention_mask: torch.Tensor = None, sample_timesteps: bool = False) -> Normal:
        """
        Returns a Normal(mean, std) distribution over actions given observation features.
        """
        obs_features = self.input_proj(obs_features)
        position_ids = None
        if sample_timesteps:
            B, T, D = obs_features.shape
            # sample initial timesteps
            rand_timesteps = torch.randint(0, self.kwargs['horizon'] - T + 1, (B,), device=obs_features.device)
            position_ids = torch.arange(T, device=obs_features.device).unsqueeze(0).expand(B, -1)
            position_ids = position_ids + rand_timesteps.unsqueeze(1)

        h = self.transformer(inputs_embeds=obs_features, attention_mask=attention_mask, position_ids=position_ids).last_hidden_state
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(min=self.log_std_limits[0], max=self.log_std_limits[1])
        std = torch.exp(log_std)
        return Normal(mean, std) # B x T x Da

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        assert 'attention_mask' in obs_dict, "attention_mask is required for TransformerImagePolicy"
        attention_mask = obs_dict.pop('attention_mask')
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, T = value.shape[:2]
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype
        # Encode obs: flatten all obs steps
        if isinstance(nobs, dict):
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1,*x.shape[2:]))
        else:
            this_nobs = nobs.reshape(-1,*nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs).reshape(B, T, -1)

        # Get action distribution
        dist = self.forward(nobs_features, attention_mask=attention_mask, sample_timesteps=False)
        action_pred = dist.rsample()  # Sample from distribution
        action = self.normalizer['action'].unnormalize(action_pred)
        seq_lens = attention_mask.sum(dim=1)
        action_pred = action_pred[torch.arange(B), seq_lens - 1, :]  # Get last valid action prediction
        action = action[torch.arange(B), seq_lens - 1, :]  # Get last valid action
        return {
            'action': action,
            'action_pred': action_pred
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        expert_mask = batch['expert_mask']
        B = nactions.shape[0]
        T = nactions.shape[1]
        Da = self.action_dim

        # Encode obs
        if isinstance(nobs, dict):
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1,*x.shape[2:]))
        else:
            this_nobs = nobs.reshape(-1,*nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(B, T, -1)
        attention_mask = batch['attention_mask']
        target = nactions

        # Get action distribution and compute loss
        dist = self.forward(nobs_features, attention_mask=attention_mask, sample_timesteps=self.sample_timesteps) # B x T x Da
        loss_mask = expert_mask[...,0] * attention_mask  # B x T
        log_prob = dist.log_prob(target).sum(dim=-1) # B x T
        loss = -(log_prob * loss_mask).sum() / loss_mask.sum()

        return loss

class DPTImagePolicy(TransformerImagePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, obs_features: torch.Tensor, target: torch.Tensor = None, attention_mask: torch.Tensor = None, sample_timesteps: bool = False) -> Normal:
        """
        Returns a Normal(mean, std) distribution over actions given observation features.
        """
        obs_features = self.input_proj(obs_features)

        # randomly permute the obs_features along the time dimension for DPT (only valid positions),
        # target action is the first action of the permuted sequence
        target_actions = []
        for i in range(obs_features.shape[0]):
            seq_len = attention_mask[i].sum().item()
            perm = torch.randperm(seq_len)
            obs_features[i, :seq_len] = obs_features[i, :seq_len][perm]
            if target is not None:
                target_actions.append(target[i, perm[0], :].unsqueeze(0))
        if target is not None:
            target = torch.cat(target_actions, dim=0) # B x Da
            # repeat to make it B x T x Da
            target = target.unsqueeze(1).repeat(1, obs_features.shape[1], 1)

        position_ids = torch.zeros_like(attention_mask).long()
        position_ids[:, 0] = 1

        h = self.transformer(inputs_embeds=obs_features, attention_mask=attention_mask, position_ids=position_ids).last_hidden_state
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(min=self.log_std_limits[0], max=self.log_std_limits[1])
        std = torch.exp(log_std)
        return Normal(mean, std), target  # B x T x Da

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        assert 'attention_mask' in obs_dict, "attention_mask is required for TransformerImagePolicy"
        attention_mask = obs_dict.pop('attention_mask')
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, T = value.shape[:2]
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype
        # Encode obs: flatten all obs steps
        if isinstance(nobs, dict):
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1,*x.shape[2:]))
        else:
            this_nobs = nobs.reshape(-1,*nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs).reshape(B, T, -1)

        obs_features = self.input_proj(nobs_features) # B x T x D

        # Permute it such that the last valid observation is at position 0, and rest follow
        for i in range(B):
            seq_len = attention_mask[i].sum().item()
            if seq_len < 2:
                continue
            perm = torch.arange(seq_len)
            perm = torch.cat([perm[-1:], perm[:-1]], dim=0)
            obs_features[i, :seq_len] = obs_features[i, :seq_len][perm]

        position_ids = torch.zeros_like(attention_mask).long()
        position_ids[:, 0] = 1

        h = self.transformer(inputs_embeds=obs_features, attention_mask=attention_mask, position_ids=position_ids).last_hidden_state
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(min=self.log_std_limits[0], max=self.log_std_limits[1])
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        action_pred = dist.rsample()  # Sample from distribution
        action = self.normalizer['action'].unnormalize(action_pred)
        seq_lens = attention_mask.sum(dim=1)
        action_pred = action_pred[torch.arange(B), seq_lens - 1, :]  # Get last valid action prediction
        action = action[torch.arange(B), seq_lens - 1, :]  # Get last valid action
        return {
            'action': action,
            'action_pred': action_pred
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['expert_action'])
        # expert_mask = batch['expert_mask']
        B = nactions.shape[0]
        T = nactions.shape[1]
        Da = self.action_dim

        # Encode obs
        if isinstance(nobs, dict):
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1,*x.shape[2:]))
        else:
            this_nobs = nobs.reshape(-1,*nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(B, T, -1)
        attention_mask = batch['attention_mask']
        target = nactions

        # Get action distribution and compute loss
        dist, target_action = self.forward(nobs_features, target, attention_mask=attention_mask, sample_timesteps=self.sample_timesteps) # B x T x Da
        loss_mask = attention_mask  # B x T
        log_prob = dist.log_prob(target_action).sum(dim=-1) # B x T
        loss = -(log_prob * loss_mask).sum() / loss_mask.sum()

        return loss

class AAWRImagePolicy(TransformerImagePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.critic_obs_dim = kwargs.get('critic_obs_dim')
        self.beta = kwargs.get('beta', 0.5)  # Temperature parameter for advantage weighting
        self.use_exp_weights = kwargs.get('use_exp_weights', True)  # Use exponential vs indicator weighting

        # Privileged critic networks (Q and V)
        self.q_network = nn.Sequential(
            nn.Linear(self.critic_obs_dim + self.action_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.v_network = nn.Sequential(
            nn.Linear(self.critic_obs_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # For IQL-style training
        self.expectile = kwargs.get('expectile', 0.7)
        self.discount = kwargs.get('discount', 0.99)

    def l2_expectile_loss(self, diff, expectile=0.7):
        """Asymmetric L2 loss used in IQL for V-function training."""
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return weight * (diff ** 2)

    def compute_loss(self, batch):
        """
        AAWR loss implementation based on https://github.com/penn-pal-lab/aawr

        Combines three components:
        1. Q-network loss (TD error)
        2. V-network loss (expectile regression)
        3. Policy loss (advantage-weighted behavioral cloning)
        """
        # Normalize observations and actions
        nobs = self.normalizer.normalize(batch['obs'])
        nexpert_obs = batch['expert_obs'] #self.normalizer.normalize(batch['expert_obs'])
        nactions = self.normalizer['action'].normalize(batch['expert_action'])

        B = nactions.shape[0]
        T = nactions.shape[1]

        # Encode partial observations for policy
        if isinstance(nobs, dict):
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        else:
            this_nobs = nobs.reshape(-1, *nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs).reshape(B, T, -1)

        # Get privileged expert observations (full state) for critic
        # Assuming expert_obs is already in the right format (B, T, critic_obs_dim)
        expert_obs_flat = nexpert_obs.reshape(B * T, -1)
        actions_flat = nactions.reshape(B * T, -1)

        attention_mask = batch['attention_mask']
        attention_mask_flat = attention_mask.reshape(B * T)

        # Get rewards and dones
        rewards = batch['reward'].reshape(B * T)
        dones = batch['done'].reshape(B * T)

        # ===== 1. Q-Network Loss (TD Error) =====
        # Q(s, a) for current state-action pairs
        q_input = torch.cat([expert_obs_flat, actions_flat], dim=-1)
        q_values = self.q_network(q_input).squeeze(-1)  # (B*T,)

        # V(s') for next states (shifted by 1)
        with torch.no_grad():
            # Create next obs by shifting
            next_expert_obs = torch.zeros_like(expert_obs_flat)
            next_expert_obs[:-1] = expert_obs_flat[1:]
            next_v_values = self.v_network(next_expert_obs).squeeze(-1)  # (B*T,)

            # TD target: r + gamma * (1 - done) * V(s')
            td_targets = rewards + self.discount * (1 - dones) * next_v_values

        # Q-value loss (only on valid timesteps)
        q_loss = F.mse_loss(q_values[attention_mask_flat.bool()],
                           td_targets[attention_mask_flat.bool()])

        # ===== 2. V-Network Loss (Expectile Regression) =====
        v_values = self.v_network(expert_obs_flat).squeeze(-1)  # (B*T,)

        # V-target is based on Q-value
        with torch.no_grad():
            v_targets = q_values.detach()

        # Expectile loss: asymmetric squared error
        v_diff = v_targets - v_values
        v_loss = self.l2_expectile_loss(v_diff, self.expectile)
        v_loss = v_loss[attention_mask_flat.bool()].mean()

        # ===== 3. Policy Loss (Advantage-Weighted BC) =====
        # Get action distribution from policy
        dist = self.forward(nobs_features, attention_mask=attention_mask,
                          sample_timesteps=self.sample_timesteps)  # B x T x Da

        # Compute advantages: A(s,a) = Q(s,a) - V(s)
        with torch.no_grad():
            advantages = (q_values - v_values).reshape(B, T)  # B x T

            # Compute advantage weights
            if self.use_exp_weights:
                # Exponential weighting: exp(beta * A)
                adv_weights = torch.exp(self.beta * advantages).clamp(max=100.0)
            else:
                # Indicator weighting: 1 if A > 0, else 0
                adv_weights = (advantages > 0).float()

        # Weighted behavioral cloning loss
        log_probs = dist.log_prob(nactions).sum(dim=-1)  # B x T
        loss_mask = attention_mask.float()  # B x T

        # Apply advantage weights and attention mask
        weighted_log_probs = adv_weights * log_probs * loss_mask
        pi_loss = -weighted_log_probs.sum() / loss_mask.sum()

        # ===== Total Loss =====
        total_loss = q_loss + v_loss + pi_loss

        # Return loss with components for logging
        loss_dict = {
            'loss': total_loss,
            'q_loss': q_loss.detach(),
            'v_loss': v_loss.detach(),
            'pi_loss': pi_loss.detach(),
            'mean_advantage': advantages[attention_mask.bool()].mean().detach(),
            'mean_adv_weight': adv_weights[attention_mask.bool()].mean().detach(),
        }

        return total_loss
