"""A rollout buffer whose GAE understands that consecutive rows of one slot
may belong to opposite sides.

Values and rewards are always from the *mover's* perspective. In a
zero-sum game the value of the next state for its mover is the negative of
its value for me, so whenever the mover changes between step t and t+1
the bootstrap term flips sign:

    delta_t = r_t + gamma * s_t * V(s_{t+1}) - V(s_t)
    A_t     = delta_t + gamma * lambda * s_t * A_{t+1}
    s_t     = +1 if mover(t+1) == mover(t) else -1

With one fixed learner seat per slot (learner vs. pool) every `s_t` is +1
and this is plain SB3 GAE. With both seats played by the learner
(self-play) it is what makes one network, one buffer, and one PPO update
learn both seats from the same games.

The mover of the step *after* the last stored one (the bootstrap
observation) is not in the buffer; `WoprVecEnv.current_am_us()` provides
it, wired in through `next_mover_source`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from stable_baselines3.common.buffers import DictRolloutBuffer

from struggler.bots.joshua.features import AM_US_INDEX


class AlternatingRolloutBuffer(DictRolloutBuffer):
    next_mover_source: Callable[[], np.ndarray] | None = None

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        if self.next_mover_source is None:
            raise RuntimeError("AlternatingRolloutBuffer.next_mover_source is not set (see wopr.train)")
        last_values = last_values.clone().cpu().numpy().flatten()
        movers = self.observations["globals"][:, :, AM_US_INDEX]
        next_movers = np.asarray(self.next_mover_source(), dtype=np.float32)
        if next_movers.shape != (self.n_envs,):
            raise ValueError(f"next_mover_source returned shape {next_movers.shape}, expected ({self.n_envs},)")

        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
                next_mover = next_movers
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
                next_mover = movers[step + 1]
            sign = np.where(next_mover == movers[step], 1.0, -1.0).astype(np.float32)
            delta = self.rewards[step] + self.gamma * sign * next_values * next_non_terminal - self.values[step]
            last_gae = delta + self.gamma * self.gae_lambda * sign * next_non_terminal * last_gae
            self.advantages[step] = last_gae
        self.returns = self.advantages + self.values
