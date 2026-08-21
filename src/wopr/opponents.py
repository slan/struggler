"""Policies the arena can seat: anything answering `PendingRow`s with option indices.

Three flavours:

- `RandomOpponent`: uniform over legal options, with its own RNG.
- `PlayerOpponent`: adapts any engine `Player` (Greedy, First, ...). Its
  `history` is empty -- the bundled bots never read it; an LLM player would.
- `NetOpponent`: a frozen `JoshuaNet`, batched over all rows it is asked
  about in one forward pass. Samples by default: a pool opponent that
  always plays its argmax line is easy to overfit against.
"""

from __future__ import annotations

import dataclasses
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from struggler.bots.joshua import features as F
from struggler.bots.joshua.model import JoshuaNet, load_checkpoint, to_tensors
from struggler.engine.player import Player
from wopr.arena import PendingRow


class RandomOpponent:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def choose(self, rows: Sequence[PendingRow]) -> list[int]:
        return [self._rng.randrange(len(row.observation.pending_decision.options)) for row in rows]


class PlayerOpponent:
    def __init__(self, player: Player) -> None:
        self._player = player

    def choose(self, rows: Sequence[PendingRow]) -> list[int]:
        choices = []
        for row in rows:
            options = row.observation.pending_decision.options
            action = self._player.choose_action(row.observation, ())
            choices.append(options.index(action))
        return choices


class NetOpponent:
    def __init__(
        self,
        net: JoshuaNet,
        *,
        seed: int,
        deterministic: bool = False,
        temperature: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        self._net = net.to(device).eval()
        self._device = torch.device(device)
        self._deterministic = deterministic
        self._temperature = temperature
        self._generator = torch.Generator().manual_seed(seed)
        self._buffers = F.allocate(0)

    @classmethod
    def from_checkpoint(cls, path: str, *, seed: int, device: torch.device | str = "cpu", **kwargs) -> "NetOpponent":
        net, _ = load_checkpoint(path, device=device)
        return cls(net, seed=seed, device=device, **kwargs)

    def choose(self, rows: Sequence[PendingRow]) -> list[int]:
        if not rows:
            return []
        if self._buffers["opt_mask"].shape[0] < len(rows):
            self._buffers = F.allocate(len(rows))
        batch = {name: array[: len(rows)] for name, array in self._buffers.items()}
        for i, row in enumerate(rows):
            F.encode_into(row.observation, batch, i)
        with torch.no_grad():
            logits, _ = self._net(to_tensors(batch, self._device))
        logits = logits.cpu()
        if self._deterministic:
            return torch.argmax(logits, dim=-1).tolist()
        probabilities = torch.softmax(logits / self._temperature, dim=-1)
        return torch.multinomial(probabilities, 1, generator=self._generator).squeeze(-1).tolist()


def masked_argmax(logits: np.ndarray, mask: np.ndarray) -> int:
    return int(np.argmax(np.where(mask.astype(bool), logits, -np.inf)))


@dataclass(frozen=True)
class StandardOpponents:
    """The opponent resolver `train.py` seats: `random`, `greedy`, `first`,
    and `pool:<name>` snapshots read from `pool_dir`. Plain data, so a
    collector process can be handed one; each resolves its opponents
    itself, seeded by `(seed, salt, policy id)` -- the same stream for the
    same policy in the same process, distinct across `salt`s."""

    pool_dir: str
    seed: int
    device: str = "cpu"
    salt: int = 0

    def for_worker(self, index: int) -> "StandardOpponents":
        """The resolver a collector process should use: same policies, its own streams."""
        return dataclasses.replace(self, salt=index + 1)

    def __call__(self, policy_id: str):
        from struggler.bots.greedy import GreedyPlayer
        from struggler.bots.naive import FirstLegalPlayer
        from wopr.pool import POOL_PREFIX

        opponent_seed = self.seed * 7919 + self.salt * 104_729 + zlib.crc32(policy_id.encode()) % 7919
        if policy_id == "random":
            return RandomOpponent(opponent_seed)
        if policy_id == "greedy":
            return PlayerOpponent(GreedyPlayer())
        if policy_id == "first":
            return PlayerOpponent(FirstLegalPlayer())
        if policy_id.startswith(POOL_PREFIX):
            path = Path(self.pool_dir) / f"{policy_id[len(POOL_PREFIX):]}.pt"
            return NetOpponent.from_checkpoint(str(path), seed=opponent_seed, device=self.device)
        raise KeyError(f"unknown opponent {policy_id!r}")
