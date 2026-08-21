"""`JoshuaPolicy`: JoshuaNet behind SB3's `ActorCriticPolicy` interface.

SB3's PPO only needs four entry points from a policy -- `forward` (rollout),
`evaluate_actions` (update), `predict_values` (bootstrap), and
`get_distribution` (predict) -- all of which reduce to one JoshuaNet call
returning masked logits and a value. The legality mask travels inside the
observation (`opt_mask`), so masked PPO needs no special algorithm class:
illegal options get `finfo.min` logits, zero probability, zero gradient.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from gymnasium import spaces
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from struggler.bots.joshua.model import JoshuaConfig, JoshuaNet

#: `fp32`: plain float32. `bf16`: the network's matmuls run in bfloat16
#: under autocast (weights and the loss stay float32) -- about half the
#: update's cost on an AVX-512 CPU, at a precision cost PPO tolerates.
PRECISIONS = ("fp32", "bf16")


class JoshuaPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        *,
        joshua_config: Mapping[str, Any] | JoshuaConfig | None = None,
        precision: str = "fp32",
        **kwargs: Any,
    ) -> None:
        if isinstance(joshua_config, JoshuaConfig):
            self.joshua_config = joshua_config
        else:
            self.joshua_config = JoshuaConfig.from_dict(joshua_config or {})
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
        self.precision = precision
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        # Replaces SB3's extractor/MLP/action-net stack entirely; the
        # features extractor SB3 built in __init__ is parameter-free and unused.
        self.net = JoshuaNet(self.joshua_config)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data["joshua_config"] = self.joshua_config.to_dict()
        data["precision"] = self.precision
        return data

    def _distribution_and_values(self, obs: PyTorchObs) -> tuple[CategoricalDistribution, torch.Tensor]:
        # `bf16` runs the network's matmuls in bfloat16 under autocast --
        # weights, the loss, and everything downstream stay float32. It is
        # applied to every call, rollout and update alike, so the log-probs
        # PPO's ratio compares were produced by the same arithmetic.
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.precision == "bf16"):
            logits, values = self.net(obs)
        logits, values = logits.float(), values.float()
        distribution = CategoricalDistribution(logits.shape[-1]).proba_distribution(action_logits=logits)
        return distribution, values.unsqueeze(-1)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self._distribution_and_values(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, values, distribution.log_prob(actions)

    def evaluate_actions(self, obs: PyTorchObs, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self._distribution_and_values(obs)
        return values, distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs: PyTorchObs) -> CategoricalDistribution:
        return self._distribution_and_values(obs)[0]

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        return self._distribution_and_values(obs)[1]
