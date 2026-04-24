"""Decision-Transformer-style interleaved-token policy.

This is a sibling of :class:`TransformerImagePolicy` that treats observations,
actions, and rewards as *separate tokens* along the sequence dimension
(instead of concatenating them as extra features on the observation token):

    [ obs_0, a_0, r_0, obs_1, a_1, r_1, ..., obs_{T-1}, a_{T-1}, r_{T-1} ]

Each token type has its own linear projection to ``hidden_dim`` plus a learned
token-type embedding. All ``K = 1 + include_action + include_reward`` tokens
within a step share the same GPT-2 position id (DT convention); the token-type
embedding alone distinguishes them.

Causality is handled by GPT-2's default causal mask: the hidden state at the
``obs_t`` position can only attend to strictly earlier tokens — at most
``r_{t-1}`` — so ``a_t`` / ``r_t`` never leak into the obs-token output used
to predict ``a_t``.
"""

import functools
from typing import Dict, Any, Optional, Union

import torch
import torch.nn as nn

from omegaconf import DictConfig
from transformers import GPT2Config, GPT2Model

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.head.output_head import OutputHead, GaussianOutputHead
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.transformer_image_policy import (
    TransformerImagePolicy,
    _build_output_head,
)


class InterleavedTransformerImagePolicy(TransformerImagePolicy):
    """Interleaved-token variant of :class:`TransformerImagePolicy`."""

    # Token-type ids (stable regardless of which flags are enabled; unused
    # types are simply not inserted into the sequence).
    TOKEN_TYPE_OBS = 0
    TOKEN_TYPE_ACTION = 1
    TOKEN_TYPE_REWARD = 2
    NUM_TOKEN_TYPES = 3

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

        # Intentionally skip TransformerImagePolicy.__init__: its input_proj
        # is sized for feature-concat, which is incompatible with the
        # per-token-type projections used here.
        nn.Module.__init__(self)

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

        # Per-token-type projections. Action / reward projections are only
        # created when the corresponding flag is enabled.
        obs_input_dim = obs_feature_dim * n_obs_steps
        # Named ``input_proj`` for symmetry with the base class; it is the
        # projection for the observation token type.
        self.input_proj = nn.Sequential(nn.Linear(obs_input_dim, hidden_dim))
        if self.include_action_in_context:
            self.action_proj = nn.Linear(action_dim, hidden_dim)
        if self.include_reward_in_context:
            self.reward_proj = nn.Linear(1, hidden_dim)

        # Learned additive embedding per token type.
        self.token_type_embed = nn.Embedding(self.NUM_TOKEN_TYPES, hidden_dim)

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
    # Interleaving helpers
    # ------------------------------------------------------------------
    @property
    def tokens_per_step(self) -> int:
        """Number of tokens this policy emits per environment step ``K``."""
        return 1 + int(self.include_action_in_context) + int(self.include_reward_in_context)

    def _token_types_in_order(self) -> list:
        """Token-type ids in the canonical per-step order (obs, action, reward)."""
        types = [self.TOKEN_TYPE_OBS]
        if self.include_action_in_context:
            types.append(self.TOKEN_TYPE_ACTION)
        if self.include_reward_in_context:
            types.append(self.TOKEN_TYPE_REWARD)
        return types

    def _type_embed(self, type_id: int, ref: torch.Tensor) -> torch.Tensor:
        """Broadcasted ``(1, 1, hidden_dim)`` type embedding on ref's device."""
        idx = torch.full((1,), type_id, dtype=torch.long, device=ref.device)
        return self.token_type_embed(idx).view(1, 1, -1)

    def _project_tokens(
        self,
        obs_features: torch.Tensor,
        action: Optional[torch.Tensor],
        reward: Optional[torch.Tensor],
    ) -> list:
        """Project each token type to ``hidden_dim`` and add its type embedding.

        ``obs_features`` is ``(B, T, obs_input_dim)``. ``action`` is
        ``(B, T, action_dim)`` normalized (or None). ``reward`` is
        ``(B, T)`` / ``(B, T, 1)`` (or None). Returns a list of
        ``(B, T, hidden_dim)`` tensors in canonical per-step order.
        """
        out = []
        type_ids = self._token_types_in_order()

        obs_tok = self.input_proj(obs_features) + self._type_embed(
            type_ids[0], obs_features)
        out.append(obs_tok)
        idx = 1

        if self.include_action_in_context:
            assert action is not None, (
                "action must be provided when include_action_in_context=True")
            act_tok = self.action_proj(action) + self._type_embed(
                type_ids[idx], action)
            out.append(act_tok)
            idx += 1

        if self.include_reward_in_context:
            assert reward is not None, (
                "reward must be provided when include_reward_in_context=True")
            if reward.dim() == 2:
                reward = reward.unsqueeze(-1)
            rew_tok = self.reward_proj(reward) + self._type_embed(
                type_ids[idx], reward)
            out.append(rew_tok)

        return out

    def _interleave_tokens(self, per_type_tokens: list) -> torch.Tensor:
        """Interleave per-type tokens along the time axis into ``(B, T*K, H)``."""
        if len(per_type_tokens) == 1:
            return per_type_tokens[0]
        stacked = torch.stack(per_type_tokens, dim=2)  # (B, T, K, H)
        B, T, K, H = stacked.shape
        return stacked.reshape(B, T * K, H)

    def _expand_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Repeat per-step attention mask ``K`` times to ``(B, T*K)``."""
        K = self.tokens_per_step
        if K == 1:
            return attention_mask
        return attention_mask.unsqueeze(-1).expand(-1, -1, K).reshape(
            attention_mask.shape[0], -1).contiguous()

    def _build_position_ids(
        self, B: int, T: int, device: torch.device,
        sample_timesteps: bool = False,
    ) -> torch.Tensor:
        """Per-step position ids of shape ``(B, T*K)`` (all K tokens share pos ``t``)."""
        K = self.tokens_per_step
        steps = torch.arange(T, device=device)
        pos = steps.unsqueeze(-1).expand(T, K).reshape(-1)  # (T*K,)
        pos = pos.unsqueeze(0).expand(B, -1)
        if sample_timesteps:
            horizon = self.kwargs['horizon']
            rand_offset = torch.randint(0, horizon - T + 1, (B,), device=device)
            pos = pos + rand_offset.unsqueeze(1)
        return pos

    def _extract_obs_hidden(self, h: torch.Tensor, T: int) -> torch.Tensor:
        """Slice the obs-token hidden states out of the interleaved stream."""
        K = self.tokens_per_step
        if K == 1:
            return h
        return h[:, ::K, :]

    # ------------------------------------------------------------------
    # Input assembly (action / reward fetched raw — NO right-shift, since
    # causal masking already prevents a_t/r_t from leaking into obs_t).
    # ------------------------------------------------------------------
    def _build_action_reward_train(self, batch: Dict[str, torch.Tensor]):
        action, reward = None, None
        if self.include_action_in_context:
            assert 'action' in batch, (
                "batch must contain 'action' when include_action_in_context=True")
            action = self.normalizer['action'].normalize(batch['action'])
        if self.include_reward_in_context:
            assert 'reward' in batch, (
                "batch must contain 'reward' when include_reward_in_context=True")
            reward = batch['reward']
        return action, reward

    def _build_action_reward_predict(self, obs_dict: Dict[str, torch.Tensor]):
        action, reward = None, None
        if self.include_action_in_context:
            assert 'action' in obs_dict, (
                "include_action_in_context=True requires 'action' in obs_dict "
                "for predict_action")
            action = self.normalizer['action'].normalize(obs_dict.pop('action'))
        if self.include_reward_in_context:
            assert 'reward' in obs_dict, (
                "include_reward_in_context=True requires 'reward' in obs_dict "
                "for predict_action")
            reward = obs_dict.pop('reward')
        return action, reward

    # ------------------------------------------------------------------
    # Core forward / predict
    # ------------------------------------------------------------------
    def forward(
        self,
        obs_features: torch.Tensor,
        attention_mask: torch.Tensor = None,
        sample_timesteps: bool = False,
        action: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
        **_ignored,  # swallow base-class kwargs (prev_action/prev_reward)
    ) -> torch.Tensor:
        """Run the trunk and return per-step *obs-token* hidden states.

        Output shape: ``(B, T, hidden_dim)``. Action / reward tokens
        contribute to the attention context but their output positions are
        discarded here — nothing currently trains from them.
        """
        B, T, _ = obs_features.shape
        device = obs_features.device

        per_type = self._project_tokens(obs_features, action=action, reward=reward)
        inputs_embeds = self._interleave_tokens(per_type)

        expanded_attn = None
        if attention_mask is not None:
            expanded_attn = self._expand_attention_mask(attention_mask)

        position_ids = self._build_position_ids(
            B, T, device=device, sample_timesteps=sample_timesteps)

        h = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=expanded_attn,
            position_ids=position_ids,
        ).last_hidden_state
        return self._extract_obs_hidden(h, T)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        assert 'attention_mask' in obs_dict, \
            "attention_mask is required for InterleavedTransformerImagePolicy"
        attention_mask = obs_dict.pop('attention_mask')

        action, reward = self._build_action_reward_predict(obs_dict)

        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, T = value.shape[:2]

        if isinstance(nobs, dict):
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        else:
            this_nobs = nobs.reshape(-1, *nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs).reshape(B, T, -1)

        h = self.forward(
            nobs_features,
            attention_mask=attention_mask,
            sample_timesteps=False,
            action=action, reward=reward,
        )
        action_pred = self.output_head.predict(h, sample=True)
        action_out = self.normalizer['action'].unnormalize(action_pred)
        seq_lens = attention_mask.sum(dim=1)
        action_pred = action_pred[torch.arange(B), seq_lens - 1, :]
        action_out = action_out[torch.arange(B), seq_lens - 1, :]
        return {
            'action': action_out,
            'action_pred': action_pred
        }

    # ------------------------------------------------------------------
    # KV-cached single-step inference — staggered layout
    #
    # Training sequence per env:
    #     [obs_0, a_0, (r_0), obs_1, a_1, (r_1), ..., obs_{T-1}, a_{T-1}, (r_{T-1})]
    # with each K-tuple (obs_t, a_t, r_t) sharing GPT-2 position id ``t``.
    #
    # KV inference cannot append ``a_t`` / ``r_t`` at the same forward as
    # ``obs_t`` (they don't exist yet — we need ``obs_t``'s hidden state to
    # sample ``a_t``). Instead we stagger: step 0 commits ``[obs_0]``; step
    # ``t >= 1`` commits ``[a_{t-1}, (r_{t-1}), obs_t]`` (obs last, so the
    # decoder reads it off the end). The resulting committed stream is
    # bit-for-bit the training sequence — each ``a_k`` lives next to its
    # ``obs_k`` with the same shared ``pos_id=k``.
    #
    # ``num_new_tokens`` is 1 at step 0 and ``K`` at step ``t >= 1``. The
    # wrapper buckets envs by is-first-step so each call here is uniform.
    # ------------------------------------------------------------------
    def _embed_new_step(
        self,
        inputs: Dict[str, torch.Tensor],
        prepend_prev_block: bool = True,
    ) -> tuple:
        """Emit tokens for one env step in the staggered KV-cache layout.

        ``prepend_prev_block=True`` (step ``t >= 1``): emit
        ``[a_{t-1}, (r_{t-1}), obs_t]`` — ``K`` tokens, obs last.
        ``prepend_prev_block=False`` (step ``t == 0``): emit ``[obs_0]`` — 1
        token; any ``action`` / ``reward`` passed in is discarded (no
        previous step exists).
        """
        inputs = dict(inputs)
        action = None
        reward = None
        if prepend_prev_block and self.include_action_in_context:
            assert 'action' in inputs, (
                "include_action_in_context=True requires 'action' in inputs "
                "(previous-step rolled-out action)")
            action = self.normalizer['action'].normalize(inputs.pop('action'))
        else:
            inputs.pop('action', None)
        if prepend_prev_block and self.include_reward_in_context:
            assert 'reward' in inputs, (
                "include_reward_in_context=True requires 'reward' in inputs "
                "(previous-step reward)")
            reward = inputs.pop('reward')
        else:
            inputs.pop('reward', None)

        obs_with_time = {k: v.unsqueeze(1) for k, v in inputs.items()}
        nobs = self.normalizer.normalize(obs_with_time)
        first_value = next(iter(nobs.values()))
        B, T = first_value.shape[:2]
        assert T == 1, f"_embed_new_step expects single-step obs (T=1), got T={T}"

        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs).reshape(B, 1, -1)

        if not prepend_prev_block:
            # Step 0: emit only the obs token (no prev action/reward to condition on).
            # Build it directly to avoid tripping _project_tokens's "action required"
            # assertion that guards the full-K training/inference path.
            type_ids = self._token_types_in_order()
            obs_tok = self.input_proj(nobs_features) + self._type_embed(
                type_ids[0], nobs_features)
            return obs_tok, 1

        if action is not None and action.dim() == 2:
            action = action.unsqueeze(1)
        if reward is not None and reward.dim() == 1:
            reward = reward.unsqueeze(1)

        per_type = self._project_tokens(nobs_features, action=action, reward=reward)

        # Canonical _project_tokens order is [obs, action, (reward)]; swap so obs is
        # LAST (so _decode_step's last-token read hits obs_t): [action, (reward), obs].
        obs_tok = per_type[0]
        prev_toks = per_type[1:]
        inputs_embeds = torch.cat(prev_toks + [obs_tok], dim=1)
        return inputs_embeds, self.tokens_per_step

    def kv_cached_step(
        self,
        inputs: Dict[str, Any],
        past_key_values=None,
        past_lengths: Optional[torch.Tensor] = None,
        max_past: int = 0,
    ) -> Dict[str, Any]:
        """Staggered KV-cached step: 1 token at step 0, ``K`` tokens otherwise.

        Requires the batch to be uniform on is-first-step (``past_lengths == 0``
        for all envs, or ``> 0`` for all envs). The wrapper is responsible for
        bucketing envs so this holds.

        Position ids mirror :meth:`_build_position_ids`: every token in the new
        block gets the env-step index of the pair it belongs to, so
        ``(obs_k, a_k, r_k)`` share ``pos_id=k`` exactly as in training.
        """
        K = self.tokens_per_step
        if K == 1:
            return super().kv_cached_step(
                inputs, past_key_values=past_key_values,
                past_lengths=past_lengths, max_past=max_past,
            )

        B = None
        for v in inputs.values():
            if torch.is_tensor(v):
                B = v.shape[0]
                break
        assert B is not None, "inputs must contain at least one tensor"

        if past_lengths is None:
            first_device = next(iter(inputs.values())).device if inputs else torch.device('cpu')
            past_lengths = torch.zeros(B, device=first_device, dtype=torch.long)

        all_first = bool((past_lengths == 0).all().item())
        none_first = bool((past_lengths > 0).all().item())
        assert all_first or none_first, (
            "InterleavedTransformerImagePolicy.kv_cached_step requires a uniform "
            "is-first-step batch (all past_lengths == 0 or all > 0). The wrapper "
            "must bucket envs before calling."
        )

        inputs_embeds, num_new_tokens = self._embed_new_step(
            inputs, prepend_prev_block=not all_first)
        assert inputs_embeds.shape[1] == num_new_tokens
        device = inputs_embeds.device
        past_lengths = past_lengths.to(device)

        # step_idx per env: 0 at first step; 1 + (L-1)/K afterwards since the
        # first step committed 1 token and every later step committed K tokens.
        if all_first:
            step_idx = torch.zeros(B, device=device, dtype=torch.long)
            position_ids = step_idx.unsqueeze(1)  # (B, 1)
        else:
            step_idx = 1 + (past_lengths - 1) // K  # (B,)
            # First (K-1) new tokens are (a_{t-1}, r_{t-1}) → pos_id=step_idx-1;
            # last new token is obs_t → pos_id=step_idx.
            prev_pos = (step_idx - 1).unsqueeze(1).expand(B, K - 1)
            cur_pos = step_idx.unsqueeze(1)
            position_ids = torch.cat([prev_pos, cur_pos], dim=1)

        if max_past > 0:
            valid_past = (
                torch.arange(max_past, device=device).unsqueeze(0).expand(B, -1)
                < past_lengths.unsqueeze(1)
            ).long()
        else:
            valid_past = torch.zeros(B, 0, device=device, dtype=torch.long)
        new_mask = torch.ones(B, num_new_tokens, device=device, dtype=torch.long)
        attention_mask = torch.cat([valid_past, new_mask], dim=1)

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

        if isinstance(nobs, dict):
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        else:
            this_nobs = nobs.reshape(-1, *nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(B, T, -1)
        attention_mask = batch['attention_mask']
        target = nactions

        action, reward = self._build_action_reward_train(batch)

        h = self.forward(
            nobs_features,
            attention_mask=attention_mask,
            sample_timesteps=self.sample_timesteps,
            action=action, reward=reward,
        )
        loss_mask = expert_mask[..., 0] * attention_mask  # (B, T)
        return self.output_head.compute_loss(h, target, loss_mask)
