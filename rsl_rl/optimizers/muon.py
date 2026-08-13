# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Muon for hidden MLP weights, with AdamW for all remaining parameters."""

from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Callable, Iterable, Iterator
from typing import Any


class MuonWithAuxAdamW(torch.optim.Optimizer):
    """Combine PyTorch Muon with AdamW using the parameter split Muon expects.

    Muon is applied only to weights of hidden ``Linear`` layers in each model's
    MLP. Input/output projections, recurrent weights, embeddings, normalization
    parameters, biases, and distribution parameters remain on AdamW.

    ``adjust_lr_fn="match_rms_adamw"`` keeps the configured learning rate on
    the same scale as an AdamW learning rate. Weight decay is disabled in both
    optimizers so selecting Muon does not also change regularization.
    """

    _STATE_FORMAT = "rsl_rl.muon_with_aux_adamw.v1"

    def __init__(
        self,
        models: Iterable[tuple[str, nn.Module, float]],
    ) -> None:
        """Build disjoint Muon and AdamW parameter groups for the models."""
        muon_cls = getattr(torch.optim, "Muon", None)
        if muon_cls is None:
            raise RuntimeError("optimizer='muon' requires a PyTorch release that provides torch.optim.Muon")

        muon_groups, adamw_groups = _partition_parameters(models)
        if not muon_groups:
            raise ValueError("Muon requires at least one hidden MLP Linear weight")

        self._muon = muon_cls(
            muon_groups,
            weight_decay=0.0,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            adjust_lr_fn="match_rms_adamw",
        )
        self._adamw = torch.optim.AdamW(adamw_groups, weight_decay=0.0)

        # Initialize Optimizer's hooks and GradScaler integration. The public
        # groups mirror the two delegates and remain the source of truth for
        # learning-rate schedulers.
        child_groups = [*self._muon.param_groups, *self._adamw.param_groups]
        super().__init__(child_groups, defaults={})
        self._copy_child_options_to_public_groups()

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients in both delegate optimizers."""
        self._muon.zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Apply one Muon step and one AdamW step to disjoint parameters."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._copy_public_options_to_child_groups()
        self._muon.step()
        self._adamw.step()
        return loss

    def state_dict(self) -> dict[str, Any]:
        """Serialize both delegate optimizers with an explicit format marker."""
        return {
            "format": self._STATE_FORMAT,
            "muon": self._muon.state_dict(),
            "adamw": self._adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore a checkpoint created by this composite optimizer."""
        if state_dict.get("format") != self._STATE_FORMAT:
            raise ValueError(
                "cannot load a non-Muon optimizer state into Muon; "
                "resume that checkpoint without loading optimizer state"
            )
        self._muon.load_state_dict(state_dict["muon"])
        self._adamw.load_state_dict(state_dict["adamw"])
        self._copy_child_options_to_public_groups()

    def _paired_groups(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        children = [*self._muon.param_groups, *self._adamw.param_groups]
        if len(children) != len(self.param_groups):
            raise RuntimeError("Muon optimizer parameter groups changed unexpectedly")
        return zip(self.param_groups, children)

    def _copy_public_options_to_child_groups(self) -> None:
        for public, child in self._paired_groups():
            for key, value in public.items():
                if key != "params":
                    child[key] = value

    def _copy_child_options_to_public_groups(self) -> None:
        for public, child in self._paired_groups():
            for key, value in child.items():
                if key != "params":
                    public[key] = value


def _partition_parameters(
    models: Iterable[tuple[str, nn.Module, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split model parameters into Muon hidden matrices and AdamW auxiliaries."""
    muon_groups: list[dict[str, Any]] = []
    adamw_groups: list[dict[str, Any]] = []
    seen: set[int] = set()

    for name, model, learning_rate in models:
        muon_ids = {id(parameter) for parameter in _hidden_mlp_weights(model)}
        muon_parameters = []
        adamw_parameters = []
        for parameter in model.parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            destination = muon_parameters if id(parameter) in muon_ids else adamw_parameters
            destination.append(parameter)

        if muon_parameters:
            muon_groups.append({"params": muon_parameters, "lr": learning_rate, "name": name})
        if adamw_parameters:
            adamw_groups.append({"params": adamw_parameters, "lr": learning_rate, "name": name})

    return muon_groups, adamw_groups


def _hidden_mlp_weights(model: nn.Module) -> list[nn.Parameter]:
    """Return hidden Linear weights, excluding the MLP input and output layers."""
    mlp = getattr(model, "mlp", None)
    if not isinstance(mlp, nn.Module):
        return []
    linear_layers = [module for module in mlp.modules() if isinstance(module, nn.Linear)]
    return [layer.weight for layer in linear_layers[1:-1]]
