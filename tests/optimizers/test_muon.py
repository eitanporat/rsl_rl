# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Muon/AdamW composite optimizer."""

from __future__ import annotations

import torch

import pytest

from rsl_rl.optimizers import MuonWithAuxAdamW


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 8)
        self.rnn = torch.nn.LSTM(8, 8)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ELU(),
            torch.nn.Linear(16, 16),
            torch.nn.ELU(),
            torch.nn.Linear(16, 16),
            torch.nn.ELU(),
            torch.nn.Linear(16, 2),
        )
        self.log_std = torch.nn.Parameter(torch.zeros(2))


def _parameters(optimizer: MuonWithAuxAdamW, offset: int) -> set[int]:
    return {id(parameter) for group in optimizer.param_groups[offset : offset + 2] for parameter in group["params"]}


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="PyTorch does not provide Muon")
def test_partitions_only_hidden_mlp_weights_into_muon() -> None:
    """Only internal MLP matrices should use Muon."""
    actor = _Model()
    critic = _Model()
    optimizer = MuonWithAuxAdamW((("actor", actor, 1e-4), ("critic", critic, 2e-4)))

    muon_parameters = _parameters(optimizer, 0)
    adamw_parameters = _parameters(optimizer, 2)
    expected_muon = {
        id(actor.mlp[2].weight),
        id(actor.mlp[4].weight),
        id(critic.mlp[2].weight),
        id(critic.mlp[4].weight),
    }
    all_parameters = {id(parameter) for model in (actor, critic) for parameter in model.parameters()}

    assert muon_parameters == expected_muon
    assert muon_parameters.isdisjoint(adamw_parameters)
    assert muon_parameters | adamw_parameters == all_parameters
    assert id(actor.mlp[0].weight) in adamw_parameters
    assert id(actor.mlp[-1].weight) in adamw_parameters
    assert id(actor.rnn.weight_hh_l0) in adamw_parameters
    assert id(actor.embedding.weight) in adamw_parameters


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="PyTorch does not provide Muon")
def test_steps_and_round_trips_optimizer_state() -> None:
    """Composite state should survive a complete optimizer round trip."""
    actor = _Model()
    critic = _Model()
    optimizer = MuonWithAuxAdamW((("actor", actor, 1e-4), ("critic", critic, 2e-4)))
    loss = sum(parameter.square().sum() for parameter in [*actor.parameters(), *critic.parameters()])

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    saved = optimizer.state_dict()
    restored = MuonWithAuxAdamW((("actor", actor, 9e-4), ("critic", critic, 9e-4)))
    restored.load_state_dict(saved)

    assert saved["format"] == "rsl_rl.muon_with_aux_adamw.v1"
    assert [group["lr"] for group in restored.param_groups] == [1e-4, 2e-4, 1e-4, 2e-4]
    assert all(parameter.grad is None for parameter in [*actor.parameters(), *critic.parameters()])


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="PyTorch does not provide Muon")
def test_rejects_incompatible_optimizer_state() -> None:
    """A legacy Adam checkpoint must not be mistaken for Muon state."""
    model = _Model()
    optimizer = MuonWithAuxAdamW((("actor", model, 1e-4),))

    with pytest.raises(ValueError, match="non-Muon optimizer state"):
        optimizer.load_state_dict(torch.optim.Adam(model.parameters()).state_dict())
