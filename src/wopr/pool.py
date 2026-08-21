"""The checkpoint pool: frozen past selves the learner trains against.

Self-play against only the latest policy cycles (rock beats scissors beats
paper beats rock); a pool of past checkpoints with *prioritised fictitious
self-play* sampling -- opponents the learner still loses to are drawn more
often -- is the standard fix. Each snapshot is a `model.save_checkpoint`
file under `directory`; `stats.json` beside them tracks the learner's
record against each one, and is what the sampling weights read.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from struggler.bots.joshua.model import JoshuaNet, save_checkpoint

POOL_PREFIX = "pool:"


class CheckpointPool:
    def __init__(self, directory: str | Path, *, hardness: float = 2.0, floor: float = 0.05, window: int | None = None) -> None:
        """`hardness` is PFSP's exponent: weight = (1 - learner win rate) ** hardness,
        `floor` keeps every opponent drawable, `window` limits sampling to the
        newest N snapshots (None: all)."""
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hardness = hardness
        self.floor = floor
        self.window = window
        self._stats_path = self.directory / "stats.json"
        self.names: list[str] = []
        self.stats: dict[str, dict[str, int]] = {}
        if self._stats_path.exists():
            data = json.loads(self._stats_path.read_text())
            self.names = list(data["names"])
            self.stats = {name: dict(record) for name, record in data["stats"].items()}

    def __len__(self) -> int:
        return len(self.names)

    def path(self, name: str) -> Path:
        return self.directory / f"{name}.pt"

    def add(self, name: str, net: JoshuaNet, *, extra: dict[str, Any] | None = None) -> str:
        if name in self.stats:
            raise ValueError(f"pool already has a snapshot named {name!r}")
        save_checkpoint(net, self.path(name), extra=extra)
        self.names.append(name)
        self.stats[name] = {"games": 0, "learner_wins": 0}
        self.save()
        return POOL_PREFIX + name

    def record(self, policy_id: str, learner_won: bool) -> None:
        """Count one learner game against `policy_id` (in memory; call `save`)."""
        if not policy_id.startswith(POOL_PREFIX):
            return
        record = self.stats[policy_id[len(POOL_PREFIX):]]
        record["games"] += 1
        record["learner_wins"] += int(learner_won)

    def learner_win_rate(self, name: str) -> float:
        record = self.stats[name]
        # An unplayed snapshot counts as an even match so it gets tried.
        return record["learner_wins"] / record["games"] if record["games"] else 0.5

    def sample(self, rng: random.Random) -> str:
        if not self.names:
            raise ValueError("cannot sample from an empty pool")
        candidates = self.names[-self.window:] if self.window else self.names
        weights = [(1.0 - self.learner_win_rate(n)) ** self.hardness + self.floor for n in candidates]
        return POOL_PREFIX + rng.choices(candidates, weights=weights, k=1)[0]

    def save(self) -> None:
        tmp = self._stats_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"names": self.names, "stats": self.stats}, indent=2))
        tmp.replace(self._stats_path)
