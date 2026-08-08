"""SAPG mixed-experience augmentation for recurrent PPO."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import unpad_trajectories


def _create_coef_embd(num_blocks: int, embd_size: int, device: str) -> torch.Tensor:
    return torch.linspace(50.0, 0.0, num_blocks, device=device)[:, None].repeat(1, embd_size)


def _slice_obs(obs: TensorDict, indices: torch.Tensor, dim: int) -> TensorDict:
    return TensorDict(
        {key: value.index_select(dim, indices) for key, value in obs.items()},
        batch_size=[obs.shape[i] if i != dim else len(indices) for i in range(len(obs.shape))],
    )


def _cat_obs(observations: list[TensorDict]) -> TensorDict:
    return TensorDict(
        {
            key: torch.cat([observation[key] for observation in observations], dim=1)
            for key in observations[0].keys()  # ruff: ignore[SIM118] - TensorDict iteration yields values
        },
        batch_size=[observations[0].shape[0], sum(observation.shape[1] for observation in observations)],
    )


def _replace_tail(obs: TensorDict, tail: torch.Tensor, embd_size: int) -> TensorDict:
    values = {}
    for group in ("actor", "critic"):
        value = obs[group]
        group_tail = tail.to(device=value.device, dtype=value.dtype)
        while group_tail.ndim < value.ndim:
            group_tail = group_tail.unsqueeze(0)
        group_tail = group_tail.expand(*value.shape[:-1], embd_size)
        values[group] = torch.cat((value[..., :-embd_size], group_tail), dim=-1)
    return TensorDict(values, batch_size=obs.batch_size)


def _slice_hidden(hidden: HiddenState, indices: torch.Tensor) -> HiddenState:
    if hidden is None:
        return None
    if isinstance(hidden, tuple):
        return tuple(value.index_select(-2, indices) for value in hidden)
    return hidden.index_select(-2, indices)


class SAPG(PPO):
    """PPO with OG SimToolReal's leader/follower mixed exploration update."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        sapg_cfg: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize PPO with SAPG rollout configuration."""
        super().__init__(actor, critic, storage, **kwargs)
        cfg = sapg_cfg or {}
        self.block_size = int(cfg.get("expl_coef_block_size", 4096))
        self.embd_size = (
            1 if "learn_param" in cfg.get("expl_type", "") else int(cfg.get("expl_reward_coef_embd_size", 32))
        )
        self.scale = float(cfg.get("expl_reward_coef_scale", 0.002))
        self.expl_reward_type = cfg.get("expl_reward_type", "none")
        self.off_policy_ratio = float(cfg.get("off_policy_ratio", 1.0))
        self.use_others_experience = cfg.get("use_others_experience", "lf")
        self.num_blocks = storage.num_envs // self.block_size
        if storage.num_envs % self.block_size:
            raise ValueError(f"num_envs {storage.num_envs} must be divisible by block size {self.block_size}")
        self.coef_embd = _create_coef_embd(self.num_blocks, self.embd_size, self.device)
        self.env_coef_embd = self.coef_embd.repeat_interleave(self.block_size, dim=0)
        self.entropy_coefs = (
            torch.linspace(0.5, 0.0, self.num_blocks, device=self.device)
            .repeat_interleave(self.block_size)
            .mul(self.scale)
        )
        storage.shuffle_trajectories = True
        self._update_rollout: RolloutStorage | None = None

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute PPO targets and add the mixed-experience follower blocks."""
        last_hidden = self.critic.get_hidden_state() if self.critic.is_recurrent else None
        normalize = self.normalize_advantage_per_mini_batch
        self.normalize_advantage_per_mini_batch = True
        try:
            super().compute_returns(obs)
        finally:
            self.normalize_advantage_per_mini_batch = normalize
        self._update_rollout = self._augment_storage(obs, last_hidden)

    def _update_storage(self) -> RolloutStorage:
        return self._update_rollout or self.storage

    def _after_update(self) -> None:
        self.storage.clear()
        self._update_rollout = None

    def _entropy_loss(self, entropy: torch.Tensor, batch: RolloutStorage.Batch) -> torch.Tensor:
        if self.expl_reward_type != "entropy":
            return super()._entropy_loss(entropy, batch)
        actor_obs = batch.observations["actor"]
        if batch.masks is not None:
            actor_obs = unpad_trajectories(actor_obs, batch.masks)
        coefficient = actor_obs[..., -self.embd_size]
        indices = (coefficient[..., None] == self.coef_embd[:, 0]).long().argmax(-1)
        coef = self.entropy_coefs[indices]
        while entropy.ndim > coef.ndim:
            entropy = entropy.squeeze(-1)
        return (coef * entropy).mean()

    @torch.no_grad()
    def _values_for(
        self, observations: TensorDict, last_obs: TensorDict, source: torch.Tensor, last_hidden: HiddenState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.critic.is_recurrent:
            flat_observations = observations.flatten(0, 1)
            values = self._critic_in_chunks(flat_observations).reshape(*observations.shape[:2], -1)
            last = self._critic_in_chunks(last_obs)
            return values, last

        saved_hidden = self.storage.saved_hidden_state_c
        current_hidden = self.critic.get_hidden_state()
        values = []
        for step in range(observations.shape[0]):
            hidden = None
            if saved_hidden is not None:
                parts = tuple(value[step][:, source] for value in saved_hidden)
                hidden = parts[0] if len(parts) == 1 else parts
            self.critic.reset(hidden_state=hidden)
            values.append(self._denormalize_values(self.critic(observations[step])).detach())
        self.critic.reset(hidden_state=_slice_hidden(last_hidden, source))
        last = self._denormalize_values(self.critic(last_obs)).detach()
        self.critic.reset(hidden_state=current_hidden)
        return torch.stack(values), last

    def _critic_in_chunks(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self._denormalize_values(
                    self.critic(observations[start : start + self.block_size])
                ).detach()
                for start in range(0, observations.shape[0], self.block_size)
            ]
        )

    def _targets_for(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        last_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_values = torch.cat((values[1:], last_values.unsqueeze(0)))
        returns = rewards + self.gamma * (1.0 - dones.float()) * next_values
        return returns, returns - values

    def _augment_storage(self, last_obs: TensorDict, last_hidden: HiddenState) -> RolloutStorage:
        storage = self.storage
        repeat_count = min(self.num_blocks, int(self.off_policy_ratio) + 1)
        repeat_idxs = [0]
        if repeat_count > 1 and self.gpu_global_rank == 0:
            repeat_idxs += torch.randperm(self.num_blocks - 1, device=self.device)[: repeat_count - 1].add(1).tolist()
        if self.is_multi_gpu:
            repeat_payload = [repeat_idxs]
            torch.distributed.broadcast_object_list(repeat_payload, src=0)
            repeat_idxs = repeat_payload[0]
        if self.use_others_experience == "none" or len(repeat_idxs) == 1:
            return storage.view(advantages=self._normalized_advantages(storage.returns - storage.values))

        base_n = storage.num_envs
        source_ids = [
            torch.arange((repeat_idx - 1) * self.block_size, repeat_idx * self.block_size, device=self.device)
            for repeat_idx in repeat_idxs[1:]
        ]
        observations = [storage.observations]
        values = [storage.values]
        returns = [storage.returns]
        advantages = [storage.advantages]
        tensor_parts = {name: [getattr(storage, name)] for name in ("actions", "rewards", "dones", "actions_log_prob")}
        distribution_parts = [[part] for part in storage.distribution_params]
        hidden_a = [storage.saved_hidden_state_a]
        hidden_c = [storage.saved_hidden_state_c]
        embedding = self.env_coef_embd

        for repeat_idx, source in zip(repeat_idxs[1:], source_ids):
            tail = torch.roll(embedding, self.block_size * repeat_idx, dims=0)[source]
            follower_obs = _replace_tail(_slice_obs(storage.observations, source, 1), tail, self.embd_size)
            follower_last_obs = _replace_tail(_slice_obs(last_obs, source, 0), tail, self.embd_size)
            follower_values, follower_last_values = self._values_for(
                follower_obs, follower_last_obs, source, last_hidden
            )
            follower_returns, follower_advantages = self._targets_for(
                storage.rewards[:, source],
                storage.dones[:, source],
                follower_values,
                follower_last_values,
            )
            observations.append(follower_obs)
            values.append(follower_values)
            returns.append(follower_returns)
            advantages.append(follower_advantages)
            for name, parts in tensor_parts.items():
                parts.append(getattr(storage, name)[:, source])
            for i, part in enumerate(storage.distribution_params):
                distribution_parts[i].append(part[:, source])
            if storage.saved_hidden_state_a is not None:
                hidden_a.append([part[:, :, source] for part in storage.saved_hidden_state_a])
            if storage.saved_hidden_state_c is not None:
                hidden_c.append([part[:, :, source] for part in storage.saved_hidden_state_c])

        updates = {"observations": _cat_obs(observations)}
        for name, parts in tensor_parts.items():
            updates[name] = torch.cat(parts, dim=1)
        updates["values"] = torch.cat(values, dim=1)
        updates["returns"] = torch.cat(returns, dim=1)
        updates["advantages"] = torch.cat(advantages, dim=1)
        updates["distribution_params"] = tuple(torch.cat(parts, dim=1) for parts in distribution_parts)
        if storage.saved_hidden_state_a is not None:
            updates["saved_hidden_state_a"] = [torch.cat(parts, dim=2) for parts in zip(*hidden_a)]
        if storage.saved_hidden_state_c is not None:
            updates["saved_hidden_state_c"] = [torch.cat(parts, dim=2) for parts in zip(*hidden_c)]
        updates["num_envs"] = base_n + len(source_ids) * self.block_size
        updates["advantages"] = self._normalized_advantages(updates["advantages"])
        return storage.view(**updates)

    def _normalized_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        if self.normalize_advantage_per_mini_batch:
            return advantages
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


__all__ = ["SAPG"]
