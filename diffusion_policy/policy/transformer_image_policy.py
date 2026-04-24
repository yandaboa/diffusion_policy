import functools
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.head.output_head import OutputHead, GaussianOutputHead
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from torch.distributions import Normal

from transformers import GPT2Config, GPT2Model


def _build_output_head(
    output_head: Optional[Union[OutputHead, DictConfig, dict, functools.partial]],
    hidden_dim: int,
    action_dim: int,
) -> OutputHead:
    """Instantiate an OutputHead from a config / partial / built module.

    Accepts an already-instantiated OutputHead, a ``functools.partial``
    (produced by Hydra when the child node sets ``_partial_: true``), a
    Hydra-style config dict (with ``_target_``), or ``None`` (defaults to
    GaussianOutputHead). ``hidden_dim`` / ``action_dim`` are injected here
    so downstream configs don't need to hard-code trunk / action dims.
    """
    if isinstance(output_head, OutputHead):
        return output_head
    if output_head is None:
        return GaussianOutputHead(hidden_dim=hidden_dim, action_dim=action_dim)
    if isinstance(output_head, functools.partial):
        head = output_head(hidden_dim=hidden_dim, action_dim=action_dim)
    else:
        cfg = dict(output_head)
        cfg.setdefault("hidden_dim", hidden_dim)
        cfg.setdefault("action_dim", action_dim)
        head = hydra.utils.instantiate(cfg)
    assert isinstance(head, OutputHead), (
        f"output_head must be an OutputHead subclass, got {type(head)}")
    return head


def _shift_right(x: torch.Tensor) -> torch.Tensor:
    """Shift ``x`` right along the time dim by 1, zero-padding position 0.

    Given ``x_t`` for ``t = 0..T-1``, returns ``y_t = x_{t-1}`` for ``t >= 1``
    and ``y_0 = 0``. Used to build "previous action / previous reward"
    features without leaking the current step's info.
    """
    pad = torch.zeros_like(x[:, :1])
    return torch.cat([pad, x[:, :-1]], dim=1)


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
            output_head: Optional[Union[OutputHead, DictConfig, dict, functools.partial]] = None,
            include_action_in_context: bool = False,
            include_reward_in_context: bool = False,
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

        self.include_action_in_context = bool(include_action_in_context)
        self.include_reward_in_context = bool(include_reward_in_context)

        # Input: all obs steps concatenated, optionally augmented per-step
        # with the previous-step action and/or reward (feature-concat style).
        per_step_dim = obs_feature_dim
        if self.include_action_in_context:
            per_step_dim += action_dim
        if self.include_reward_in_context:
            per_step_dim += 1
        input_dim = per_step_dim * n_obs_steps

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

        self.output_head: OutputHead = _build_output_head(
            output_head, hidden_dim=hidden_dim, action_dim=action_dim)

        self.log_std_limits = (-5.0, 2.0)
        self.sample_timesteps = sample_timesteps

    # ------------------------------------------------------------------
    # Feature-concat helpers for prev-action / prev-reward augmentation
    # ------------------------------------------------------------------
    def _augment_with_prev_action_reward(
        self,
        obs_features: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        prev_reward: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Append previous-step action / reward as extra features per timestep.

        ``obs_features`` is ``(B, T, obs_feature_dim)``. Returns a tensor of
        shape ``(B, T, per_step_dim)`` where ``per_step_dim`` matches the
        ``input_proj`` layer configured in ``__init__``.
        """
        parts = [obs_features]
        if self.include_action_in_context:
            assert prev_action is not None, (
                "prev_action must be provided when include_action_in_context=True")
            parts.append(prev_action)
        if self.include_reward_in_context:
            assert prev_reward is not None, (
                "prev_reward must be provided when include_reward_in_context=True")
            if prev_reward.dim() == 2:
                prev_reward = prev_reward.unsqueeze(-1)
            parts.append(prev_reward)
        return torch.cat(parts, dim=-1)

    def _build_prev_action_reward_train(
        self, batch: Dict[str, torch.Tensor]
    ):
        """Build normalized prev-action / raw prev-reward from a training batch.

        Actions / rewards are shifted right by 1 step (zero-padded at t=0)
        so the model only sees strictly previous information.
        """
        prev_action, prev_reward = None, None
        if self.include_action_in_context:
            assert 'action' in batch, (
                "batch must contain 'action' (rolled-out action) when "
                "include_action_in_context=True")
            nactions_roll = self.normalizer['action'].normalize(batch['action'])
            prev_action = _shift_right(nactions_roll)
        if self.include_reward_in_context:
            assert 'reward' in batch, (
                "batch must contain 'reward' when include_reward_in_context=True")
            reward = batch['reward']
            if reward.dim() == 2:
                reward = reward.unsqueeze(-1)
            prev_reward = _shift_right(reward)
        return prev_action, prev_reward

    def _build_prev_action_reward_predict(
        self, obs_dict: Dict[str, torch.Tensor],
    ):
        """Same shift-right as training, but reads from / pops from ``obs_dict``."""
        prev_action, prev_reward = None, None
        if self.include_action_in_context:
            assert 'action' in obs_dict, (
                "include_action_in_context=True requires 'action' in obs_dict "
                "for predict_action (rolled-out action history per step)")
            nactions_roll = self.normalizer['action'].normalize(obs_dict.pop('action'))
            prev_action = _shift_right(nactions_roll)
        if self.include_reward_in_context:
            assert 'reward' in obs_dict, (
                "include_reward_in_context=True requires 'reward' in obs_dict "
                "for predict_action")
            reward = obs_dict.pop('reward')
            if reward.dim() == 2:
                reward = reward.unsqueeze(-1)
            prev_reward = _shift_right(reward)
        return prev_action, prev_reward

    def forward(
        self,
        obs_features: torch.Tensor,
        attention_mask: torch.Tensor = None,
        sample_timesteps: bool = False,
        prev_action: Optional[torch.Tensor] = None,
        prev_reward: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns per-step hidden states (B, T, hidden_dim). The output head
        consumes these to produce an action distribution / loss.
        """
        obs_features = self._augment_with_prev_action_reward(
            obs_features, prev_action=prev_action, prev_reward=prev_reward)
        obs_features = self.input_proj(obs_features)
        position_ids = None
        if sample_timesteps:
            B, T, D = obs_features.shape
            # sample initial timesteps
            rand_timesteps = torch.randint(0, self.kwargs['horizon'] - T + 1, (B,), device=obs_features.device)
            position_ids = torch.arange(T, device=obs_features.device).unsqueeze(0).expand(B, -1)
            position_ids = position_ids + rand_timesteps.unsqueeze(1)

        h = self.transformer(inputs_embeds=obs_features, attention_mask=attention_mask, position_ids=position_ids).last_hidden_state
        return h  # (B, T, hidden_dim)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        assert 'attention_mask' in obs_dict, "attention_mask is required for TransformerImagePolicy"
        attention_mask = obs_dict.pop('attention_mask')

        prev_action, prev_reward = self._build_prev_action_reward_predict(obs_dict)

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

        # Get hidden states and sample an action via the output head
        h = self.forward(
            nobs_features, attention_mask=attention_mask,
            sample_timesteps=False,
            prev_action=prev_action, prev_reward=prev_reward,
        )
        action_pred = self.output_head.predict(h, sample=True)  # Sample from distribution
        action = self.normalizer['action'].unnormalize(action_pred)
        seq_lens = attention_mask.sum(dim=1)
        action_pred = action_pred[torch.arange(B), seq_lens - 1, :]  # Get last valid action prediction
        action = action[torch.arange(B), seq_lens - 1, :]  # Get last valid action
        return {
            'action': action,
            'action_pred': action_pred
        }

    # ------------------------------------------------------------------
    # KV-cached single-step inference
    #
    # The three-piece split below decouples the cache plumbing
    # (:meth:`kv_cached_step`, which just builds masks / position ids and
    # forwards the transformer) from the model-specific parts:
    #
    #   * :meth:`_embed_new_step` defines **how many tokens this step
    #     produces and what they are** — subclass this to interleave action
    #     tokens between obs tokens, prepend learned separators, fuse several
    #     modalities into ``k`` tokens, etc. It just needs to return
    #     ``(inputs_embeds, num_new_tokens)`` with ``num_new_tokens`` uniform
    #     across the batch.
    #
    #   * :meth:`_decode_step` defines **which token(s) produce the action**
    #     given the transformer's hidden state over just the new tokens.
    #
    # The cache manager and the wrapper are token-structure agnostic and
    # plumb only the ``num_new_tokens`` count that the policy returns.
    # ------------------------------------------------------------------

    def _embed_new_step(self, inputs: Dict[str, torch.Tensor]) -> tuple:
        """Embed one env step's inputs into transformer tokens.

        Returns ``(inputs_embeds, num_new_tokens)`` where ``inputs_embeds`` has shape
        ``(B, num_new_tokens, hidden_dim)``. ``num_new_tokens`` must be identical for every
        env in the batch (a policy that varies ``k_new`` dynamically per env would need
        ragged new-token dim support, which is deliberately out of scope here).

        Default implementation: one observation token per step, mirroring
        :meth:`predict_action`. Honors ``include_action_in_context`` /
        ``include_reward_in_context`` — the caller must supply ``action`` /
        ``reward`` in ``inputs`` (the *previous* step's value) when those flags are on.

        Subclasses override this to change token structure without touching the KV cache.
        """
        inputs = dict(inputs)
        # Pop prev-action / prev-reward before normalizing obs; these are
        # already the previous-step values (caller responsibility).
        prev_action = None
        prev_reward = None
        if self.include_action_in_context:
            assert 'action' in inputs, (
                "include_action_in_context=True requires 'action' in inputs "
                "(previous-step rolled-out action)")
            prev_action = self.normalizer['action'].normalize(inputs.pop('action'))
        if self.include_reward_in_context:
            assert 'reward' in inputs, (
                "include_reward_in_context=True requires 'reward' in inputs "
                "(previous-step reward)")
            prev_reward = inputs.pop('reward')

        obs_with_time = {k: v.unsqueeze(1) for k, v in inputs.items()}
        nobs = self.normalizer.normalize(obs_with_time)
        first_value = next(iter(nobs.values()))
        B, T = first_value.shape[:2]
        assert T == 1, f"_embed_new_step expects single-step obs (T=1), got T={T}"

        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs).reshape(B, 1, -1)

        if prev_action is not None and prev_action.dim() == 2:
            prev_action = prev_action.unsqueeze(1)
        if prev_reward is not None and prev_reward.dim() == 1:
            prev_reward = prev_reward.unsqueeze(1)
        nobs_features = self._augment_with_prev_action_reward(
            nobs_features, prev_action=prev_action, prev_reward=prev_reward)
        inputs_embeds = self.input_proj(nobs_features)
        return inputs_embeds, 1

    def _decode_step(self, last_hidden_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode an action from the transformer's per-new-token hidden states.

        ``last_hidden_state`` has shape ``(B, num_new_tokens, hidden_dim)`` — states for
        *just this step's* new tokens (KV-cached forward returns one output per input
        token). Override if the action should be read from a different token position
        (e.g. the middle token in a ``[pre, obs, post]`` triple) or if you want to emit
        auxiliary heads alongside the action.

        Default: take the last new token, run it through ``self.output_head``.
        """
        h_last = last_hidden_state[:, -1:, :]  # (B, 1, H) — preserve time dim for output head
        action_pred = self.output_head.predict(h_last, sample=True)
        action = self.normalizer['action'].unnormalize(action_pred)
        return {
            'action': action.squeeze(1),
            'action_pred': action_pred.squeeze(1),
        }

    def kv_cached_step(
        self,
        inputs: Dict[str, Any],
        past_key_values=None,
        past_lengths: Optional[torch.Tensor] = None,
        max_past: int = 0,
    ) -> Dict[str, Any]:
        """Single-step KV-cached inference; agnostic to per-step token structure.

        This method owns the cache plumbing (attention mask / position id construction
        and the transformer forward), delegating the token-structure-specific parts to
        :meth:`_embed_new_step` and :meth:`_decode_step`. The cache manager only needs
        to know how many tokens were appended this step (via ``num_new_tokens`` in the
        return dict) to slot them into per-env storage.

        Args:
            inputs: Per-env step inputs (each value with leading dim ``B``). The meaning
                of keys is up to :meth:`_embed_new_step`.
            past_key_values: Padded-batched ``(B, n_heads, max_past, head_dim)`` per layer
                from :class:`TransformerKVCacheManager.gather`, or ``None`` if all envs in
                the batch are starting fresh.
            past_lengths: ``(B,)`` true per-env past lengths; used to mask padded positions
                in ``past_key_values`` and to build per-env absolute ``position_ids``.
            max_past: ``int(past_lengths.max())`` — padded length of ``past_key_values``.

        Returns a dict with ``action`` ``(B, action_dim)``, ``action_pred``,
        ``past_key_values`` (updated transformer cache), and ``num_new_tokens``
        (``int``) for the cache manager.
        """
        inputs_embeds, num_new_tokens = self._embed_new_step(inputs)
        B = inputs_embeds.shape[0]
        assert inputs_embeds.shape[1] == num_new_tokens, (
            f"_embed_new_step returned T={inputs_embeds.shape[1]} but claimed "
            f"num_new_tokens={num_new_tokens}; both must match."
        )
        device = inputs_embeds.device
        if past_lengths is None:
            past_lengths = torch.zeros(B, device=device, dtype=torch.long)

        # attention_mask covers [padded_past | num_new_tokens new tokens]. Past-pad
        # positions are masked out per env's true length; new tokens are all valid.
        if max_past > 0:
            valid_past = (
                torch.arange(max_past, device=device).unsqueeze(0).expand(B, -1)
                < past_lengths.unsqueeze(1)
            ).long()
        else:
            valid_past = torch.zeros(B, 0, device=device, dtype=torch.long)
        new_mask = torch.ones(B, num_new_tokens, device=device, dtype=torch.long)
        attention_mask = torch.cat([valid_past, new_mask], dim=1)

        # Absolute positions: new tokens occupy past_len .. past_len + k - 1 per env.
        offsets = torch.arange(num_new_tokens, device=device).unsqueeze(0)
        position_ids = past_lengths.unsqueeze(1) + offsets

        out = self.transformer(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        decoded = self._decode_step(out.last_hidden_state)
        return {
            **decoded,
            'past_key_values': out.past_key_values,
            'num_new_tokens': num_new_tokens,
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

        prev_action, prev_reward = self._build_prev_action_reward_train(batch)

        # Get hidden states and compute loss via output head
        h = self.forward(
            nobs_features, attention_mask=attention_mask,
            sample_timesteps=self.sample_timesteps,
            prev_action=prev_action, prev_reward=prev_reward,
        )
        loss_mask = expert_mask[...,0] * attention_mask  # B x T
        return self.output_head.compute_loss(h, target, loss_mask)

class DPTImagePolicy(TransformerImagePolicy):
    """DPT variant: permutes the observation sequence and predicts the action
    at the (pre-permutation) first position. Assumes a Gaussian output head.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.output_head, GaussianOutputHead), (
            "DPTImagePolicy currently assumes a GaussianOutputHead")

    def forward(self, obs_features: torch.Tensor, target: torch.Tensor = None, attention_mask: torch.Tensor = None, sample_timesteps: bool = False):
        """
        Returns (dist, permuted_target) — dist is a Normal over actions given
        permuted observation features.
        """
        # NOTE: DPTImagePolicy predates the include_action/include_reward
        # abstraction and continues to treat obs_features as the sole input.
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
        dist = self.output_head._distribution(h)
        return dist, target  # B x T x Da

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
        action_pred = self.output_head.predict(h, sample=True)  # Sample from distribution
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
        assert isinstance(self.output_head, GaussianOutputHead), (
            "AAWRImagePolicy currently assumes a GaussianOutputHead")
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
        prev_action, prev_reward = self._build_prev_action_reward_train(batch)
        h = self.forward(
            nobs_features, attention_mask=attention_mask,
            sample_timesteps=self.sample_timesteps,
            prev_action=prev_action, prev_reward=prev_reward,
        )
        dist = self.output_head._distribution(h)  # B x T x Da

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
