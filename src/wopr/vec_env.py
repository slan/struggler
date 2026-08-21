"""`WoprVecEnv`: the arena as a Stable-Baselines3 `VecEnv`.

One SB3 "env" = one arena slot. The twist, compared to a single-agent
env, is *who the row belongs to*: after the learner's action in a slot,
the arena fast-forwards through CHANCE frames and through every decision
that belongs to a non-learner seat (answered in batch by the registered
opponents) and stops at the next decision the learner must make -- which,
in a self-play game where both seats are the learner, may be the *other*
side's. Every row therefore carries its mover in `globals[AM_US_INDEX]`,
and the reward of a row is the game's outcome *for that row's mover*:
+1 win, -1 loss, 0 draw, on the row after which the game ended.
`buffer.AlternatingRolloutBuffer` turns that into correct advantages.

Episodes auto-reset. `infos[i]["episode"]` follows SB3's Monitor
convention (`r`, `l`) so `rollout/ep_rew_mean` works unchanged, plus
`winner`, `seats`, `mover`, `turn`, `vp` for the WOPR callback.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from struggler.bots.joshua import features as F
from struggler.engine import Side
from wopr.arena import Arena, Opponent, PendingRow

LEARNER = "learner"

#: Resolves a non-learner policy id to something that can answer rows.
OpponentResolver = Callable[[str], Opponent]


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

    def __init__(self, arena: Arena, opponents: OpponentResolver, *, learner: str = LEARNER) -> None:
        super().__init__(arena.n_games, observation_space(), spaces.Discrete(F.K_MAX))
        self.arena = arena
        self._resolve = opponents
        self._opponents: dict[str, Opponent] = {}
        self._learner = learner
        self._buffers = F.allocate(arena.n_games)
        self._rows: list[PendingRow | None] = [None] * arena.n_games
        self._actions: np.ndarray | None = None
        self._episode_steps = np.zeros(arena.n_games, dtype=np.int64)
        self._results: list[dict[str, Any]] = []

    # -- VecEnv API --------------------------------------------------------------

    def reset(self) -> dict[str, np.ndarray]:
        self.arena.reset_all()
        self._episode_steps[:] = 0
        self._advance(range(self.num_envs), rewards=None, dones=None, infos=None)
        return self._encode_all()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions)

    def step_wait(self) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if self._actions is None:
            raise RuntimeError("step_wait called before step_async")
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        # Snapshot the movers before stepping: a row's reward is from the
        # perspective of whoever acted on it, which `_advance` needs after
        # the arena has moved on.
        movers = [row.side for row in self._rows]
        for slot, action in enumerate(self._actions):
            self.arena.apply(slot, int(action))
            self._episode_steps[slot] += 1
        self._actions = None
        self._advance(range(self.num_envs), rewards=rewards, dones=dones, infos=infos, movers=movers)
        return self._encode_all(), rewards, dones, infos

    def close(self) -> None:
        return None

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
        return np.array([1.0 if row.side is Side.US else 0.0 for row in self._rows], dtype=np.float32)

    def drain_results(self) -> list[dict[str, Any]]:
        results, self._results = self._results, []
        return results

    # -- internals -------------------------------------------------------------------

    def _opponent(self, policy_id: str) -> Opponent:
        opponent = self._opponents.get(policy_id)
        if opponent is None:
            opponent = self._opponents[policy_id] = self._resolve(policy_id)
        return opponent

    def _advance(self, slots, *, rewards, dones, infos, movers: Sequence[Side] | None = None) -> None:
        """Bring every slot in `slots` to a learner decision, answering other
        seats' decisions with their opponents and closing finished games."""
        active = list(slots)
        while active:
            still_active: list[int] = []
            by_policy: dict[str, list[PendingRow]] = {}
            for slot in active:
                if self.arena.is_terminal(slot):
                    if rewards is not None:
                        self._finish(slot, movers[slot], rewards, dones, infos)
                    else:
                        self.arena.reset(slot)  # a game over at reset time: extremely unlikely, but legal
                    still_active.append(slot)
                    continue
                policy_id = self.arena.policy_for(slot)
                if policy_id == self._learner:
                    self._rows[slot] = self.arena.row(slot)
                    continue
                by_policy.setdefault(policy_id, []).append(self.arena.row(slot))
                still_active.append(slot)
            for policy_id, rows in by_policy.items():
                choices = self._opponent(policy_id).choose(rows)
                if len(choices) != len(rows):
                    raise ValueError(f"opponent {policy_id!r} answered {len(choices)} of {len(rows)} rows")
                for row, choice in zip(rows, choices):
                    self.arena.apply(row.slot, int(choice))
            active = still_active

    def _finish(self, slot: int, mover: Side, rewards, dones, infos) -> None:
        result = self.arena.result(slot)
        if result.winner is None:
            reward = 0.0
        else:
            reward = 1.0 if result.winner is mover else -1.0
        rewards[slot] = reward
        dones[slot] = True
        # SB3 reads `terminal_observation` only to bootstrap truncated
        # episodes; games here always end for real, so the last encoded row
        # is enough.
        infos[slot]["terminal_observation"] = {name: array[slot].copy() for name, array in self._buffers.items()}
        summary = {
            "r": reward,
            "l": int(self._episode_steps[slot]),
            "winner": None if result.winner is None else result.winner.value,
            "seats": {side.value: policy for side, policy in result.seats.items()},
            "mover": mover.value,
            "turn": result.turn,
            "vp": result.vp,
            "seed": result.seed,
        }
        infos[slot]["episode"] = summary
        self._results.append(summary)
        self._episode_steps[slot] = 0
        self.arena.reset(slot)

    def _encode_all(self) -> dict[str, np.ndarray]:
        for slot, row in enumerate(self._rows):
            F.encode_into(row.observation, self._buffers, slot)
        # SB3 keeps references to what we return; hand it copies.
        return {name: array.copy() for name, array in self._buffers.items()}
