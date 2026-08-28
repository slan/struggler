"""`WoprVecEnv`: a WOPR backend as a Stable-Baselines3 `VecEnv`.

One SB3 "env" = one slot. The twist, compared to a single-agent env, is
*who the row belongs to*: after the learner's action in a slot, the
backend fast-forwards through CHANCE frames and through every decision
that belongs to a non-learner seat (answered in batch by the registered
opponents) and stops at the next decision the learner must make -- which,
in a self-play game where both seats are the learner, may be the *other*
side's. Every row therefore carries its mover in `globals[AM_US_INDEX]`,
and the reward of a row is the game's outcome *for that row's mover*:
+1 win, -1 loss, 0 draw, on the row after which the game ended.
`buffer.AlternatingRolloutBuffer` turns that into correct advantages.

The stepping itself is a `backend.Backend` -- in-process engines, or k
collector processes over shared memory -- and this class only adapts it
to SB3: episodes auto-reset, `infos[i]["episode"]` follows SB3's Monitor
convention (`r`, `l`) so `rollout/ep_rew_mean` works unchanged, plus
`winner`, `seats`, `mover`, `turn`, `vp`, `seed` for the WOPR callback.
No `terminal_observation`: SB3 reads it only to bootstrap truncated
episodes, and games here always end for real.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from struggler.bots.joshua import features as F
from wopr.arena import Arena
from wopr.backend import LEARNER, Backend, InProcessBackend, OpponentResolver

__all__ = ["LEARNER", "OpponentResolver", "WoprVecEnv", "observation_space"]


def observation_space() -> spaces.Dict:
    boxes: dict[str, spaces.Space] = {}
    for name, (shape, dtype) in F.LAYOUT.items():
        if name == "opt_mask":
            boxes[name] = spaces.MultiBinary(shape)
        elif dtype is np.int64:
            high = {"card_loc": F.N_CARD_LOCATIONS - 1, "focus": F.N_CARDS,
                    "opt_country": F.N_COUNTRIES, "opt_card": F.N_CARDS}[name]
            boxes[name] = spaces.Box(low=0, high=high, shape=shape, dtype=np.int64)
        else:
            boxes[name] = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
    return spaces.Dict(boxes)


class WoprVecEnv(VecEnv):
    render_mode = None

    def __init__(self, backend: Backend | Arena, opponents: OpponentResolver | None = None, *, learner: str = LEARNER) -> None:
        """Either a `Backend`, or an `Arena` plus an opponent resolver for the
        in-process one."""
        if isinstance(backend, Arena):
            if opponents is None:
                raise ValueError("an Arena needs an opponent resolver")
            backend = InProcessBackend(backend, opponents, learner=learner)
        self.backend = backend
        super().__init__(backend.n_slots, observation_space(), spaces.Discrete(F.K_MAX))
        self._actions: np.ndarray | None = None

    # -- VecEnv API --------------------------------------------------------------

    def reset(self) -> dict[str, np.ndarray]:
        self.backend.reset()
        return self._observations()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions)

    def step_wait(self) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if self._actions is None:
            raise RuntimeError("step_wait called before step_async")
        rewards, dones, records = self.backend.step(self._actions)
        self._actions = None
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for slot, record in enumerate(records):
            if record is not None:
                infos[slot]["episode"] = record.summary()
        return self._observations(), rewards, dones, infos

    def close(self) -> None:
        self.backend.close()

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        return [getattr(self, attr_name) for _ in self._indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        raise NotImplementedError("WoprVecEnv has no per-env attributes to set")

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list[Any]:
        raise NotImplementedError("WoprVecEnv slots are not gym envs")

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False for _ in self._indices(indices)]

    def seed(self, seed: int | None = None) -> Sequence[None]:
        # Seeding is fixed at Arena construction (mandate #3): same seed, same games.
        return [None] * self.num_envs

    def _indices(self, indices) -> range | list[int]:
        if indices is None:
            return range(self.num_envs)
        return [indices] if isinstance(indices, int) else list(indices)

    # -- WOPR extras ---------------------------------------------------------------

    def current_am_us(self) -> np.ndarray:
        """1.0 where the learner's *next* row is played as US -- the buffer's
        bootstrap needs the mover of the observation after the last stored step."""
        return self.backend.am_us()

    # -- internals -------------------------------------------------------------------

    def _observations(self) -> dict[str, np.ndarray]:
        # SB3 keeps what we return until after the next step, and the backend
        # overwrites its rows in place: hand it copies.
        return {name: array.copy() for name, array in self.backend.buffers.items()}
