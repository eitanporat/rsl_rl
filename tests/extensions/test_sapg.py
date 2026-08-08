"""Tests for SimToolReal SAPG mixed-experience updates."""

import torch
from tensordict import TensorDict

from rsl_rl.extensions.sapg import SAPG
from rsl_rl.models import MLPModel, RNNModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import split_and_pad_trajectories

NUM_ENVS, NUM_STEPS, NUM_ACTIONS = 4, 2, 2


def _make_sapg() -> tuple[SAPG, TensorDict]:
    coefficient = torch.tensor([50.0, 50.0, 0.0, 0.0])[:, None]
    obs = TensorDict(
        {
            "actor": torch.cat((torch.randn(NUM_ENVS, 3), coefficient), -1),
            "critic": torch.cat((torch.randn(NUM_ENVS, 4), coefficient), -1),
        },
        batch_size=[NUM_ENVS],
    )
    groups = {"actor": ["actor"], "critic": ["critic"]}
    actor = MLPModel(
        obs,
        groups,
        "actor",
        NUM_ACTIONS,
        hidden_dims=[8],
        aux_value=True,
        coefficient_embedding_cfg={
            "condition_group": "actor",
            "condition_index": -1,
            "embedding_dim": 32,
        },
        distribution_cfg={
            "class_name": "CoefficientGaussianDistribution",
            "condition_group": "actor",
            "condition_index": -1,
            "extra_info_dim": 1,
            "std_type": "log",
        },
    )
    critic = MLPModel(
        obs,
        groups,
        "critic",
        1,
        hidden_dims=[8],
        coefficient_embedding_cfg={
            "condition_group": "critic",
            "condition_index": -1,
            "embedding_dim": 32,
        },
    )
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    alg = SAPG(
        actor,
        critic,
        storage,
        gamma=0.9,
        lam=0.95,
        num_learning_epochs=1,
        num_mini_batches=1,
        sapg_cfg={
            "expl_coef_block_size": 2,
            "expl_reward_coef_embd_size": 32,
            "expl_reward_coef_scale": 0.002,
            "expl_type": "mixed_expl_learn_param",
            "off_policy_ratio": 1,
            "use_others_experience": "lf",
        },
        aux_value_loss_coef=2,
    )
    return alg, obs


def test_learned_parameter_uses_scalar_coefficient() -> None:
    """Learn-param SAPG exposes one scalar coefficient per environment."""
    alg, _ = _make_sapg()
    assert alg.embd_size == 1
    torch.testing.assert_close(alg.coef_embd, torch.tensor([[50.0], [0.0]]))


def test_network_inputs_match_upstream_learned_parameter_expansion() -> None:
    """Actor and critic replace the scalar with their own learned 32-vector."""
    alg, obs = _make_sapg()
    for model, group, base_dim in ((alg.actor, "actor", 3), (alg.critic, "critic", 4)):
        table = model.coefficient_embedding.embedding
        values = model.coefficient_embedding.values
        indices = (obs[group][:, -1, None] == values).long().argmax(-1)
        upstream = torch.cat((obs[group][:, :base_dim], table[indices]), -1)

        torch.testing.assert_close(model.get_latent(obs), upstream)
        assert model.input_dim == base_dim + 32

    for value, std in ((50, 2), (0, 3)):
        row = (alg.actor.distribution.condition_values == value).nonzero().item()
        alg.actor.distribution.log_std_param.data[row].fill_(torch.log(torch.tensor(std)))
    alg.actor(obs, stochastic_output=True)
    torch.testing.assert_close(
        alg.actor.output_std,
        torch.tensor([[2.0], [2.0], [3.0], [3.0]]).expand(-1, NUM_ACTIONS),
    )
    exported = torch.jit.script(alg.actor.as_jit())
    torch.testing.assert_close(exported(obs["actor"]), alg.actor(obs))


def test_follower_targets_match_upstream_one_step_bootstrap() -> None:
    """Follower targets use the upstream one-step bootstrap."""
    alg, _ = _make_sapg()
    rewards = torch.ones(2, 1, 1)
    values = torch.tensor([[[1.0]], [[2.0]]])
    returns, advantages = alg._targets_for(
        rewards,
        torch.zeros_like(rewards),
        values,
        torch.tensor([[3.0]]),
    )
    expected = torch.tensor([[[2.8]], [[3.7]]])
    torch.testing.assert_close(returns, expected)
    torch.testing.assert_close(advantages, expected - values)


def test_leader_follower_augmentation_keeps_leader_and_one_follower_block() -> None:
    """Leader/follower filtering retains all leaders and one source block."""
    alg, obs = _make_sapg()
    for _ in range(NUM_STEPS):
        alg.act(obs)
        alg.process_env_step(obs, torch.ones(NUM_ENVS), torch.zeros(NUM_ENVS), {})
    original_actions = alg.storage.actions.clone()
    alg.compute_returns(obs)
    update_storage = alg._update_storage()

    assert alg.storage.num_envs == NUM_ENVS
    assert update_storage.num_envs == 6
    torch.testing.assert_close(alg.storage.actions[:, :NUM_ENVS], original_actions)
    torch.testing.assert_close(update_storage.actions[:, NUM_ENVS:], original_actions[:, :2])
    torch.testing.assert_close(
        update_storage.observations["actor"][:, NUM_ENVS:, -1],
        torch.zeros(NUM_STEPS, 2),
    )


def test_rollout_to_update_matches_upstream_leader_follower_reference() -> None:
    """A complete update matches an independent upstream-reference calculation."""
    torch.manual_seed(7)
    alg, obs = _make_sapg()
    rewards = []
    dones = []
    for step in range(NUM_STEPS):
        alg.act(obs)
        reward = torch.arange(NUM_ENVS, dtype=torch.float32) + step
        done = torch.zeros(NUM_ENVS)
        done[1] = step
        rewards.append(reward[:2, None])
        dones.append(done[:2, None])
        obs = obs.clone()
        obs["actor"][:, :-1] += 0.1
        obs["critic"][:, :-1] += 0.2
        alg.process_env_step(obs, reward, done, {})

    follower_obs = alg.storage.observations[:, :2].clone()
    follower_obs["actor"][:, :, -1] = 0
    follower_obs["critic"][:, :, -1] = 0
    follower_last = obs[:2].clone()
    follower_last["actor"][:, -1] = 0
    follower_last["critic"][:, -1] = 0
    with torch.no_grad():
        values = alg.critic(follower_obs)
        last_values = alg.critic(follower_last)
    next_values = torch.cat((values[1:], last_values[None]))
    upstream_returns = torch.stack(rewards) + alg.gamma * (1 - torch.stack(dones)) * next_values

    alg.compute_returns(obs)
    update_storage = alg._update_storage()

    torch.testing.assert_close(update_storage.observations[:, NUM_ENVS:], follower_obs)
    torch.testing.assert_close(update_storage.values[:, NUM_ENVS:], values)
    torch.testing.assert_close(update_storage.returns[:, NUM_ENVS:], upstream_returns)
    losses = alg.update()
    assert {"aux_value", "entropy", "surrogate", "value"} <= losses.keys()
    assert alg.storage.num_envs == NUM_ENVS
    assert alg.storage.step == 0


def test_recurrent_condition_ignores_trajectory_padding() -> None:
    """Recurrent coefficient selection ignores trajectory padding."""
    _, obs = _make_sapg()
    model = RNNModel(
        obs,
        {"actor": ["actor"]},
        "actor",
        NUM_ACTIONS,
        hidden_dims=[8],
        rnn_type="gru",
        rnn_hidden_dim=8,
        distribution_cfg={
            "class_name": "CoefficientGaussianDistribution",
            "condition_group": "actor",
            "condition_index": -1,
            "extra_info_dim": 1,
            "std_type": "log",
        },
        coefficient_embedding_cfg={
            "condition_group": "actor",
            "condition_index": -1,
            "embedding_dim": 32,
        },
    )
    assert model.rnn.rnn.input_size == 35
    sequence = TensorDict(
        {"actor": obs["actor"].repeat(NUM_STEPS, 1, 1)},
        batch_size=[NUM_STEPS, NUM_ENVS],
    )
    dones = torch.zeros(NUM_STEPS, NUM_ENVS, 1)
    dones[0, 0] = 1
    padded, masks = split_and_pad_trajectories(sequence, dones)
    hidden = torch.zeros(1, padded.shape[1], 8)

    actions = model(padded, masks=masks, hidden_state=hidden, stochastic_output=True)
    assert actions.shape == (NUM_STEPS, NUM_ENVS, NUM_ACTIONS)
    assert model.output_std.shape == actions.shape
