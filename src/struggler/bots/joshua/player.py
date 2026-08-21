"""JoshuaPlayer: a `Player` that answers every decision with a trained JoshuaNet.

Inference only. It sees exactly what any `Player` sees (`Observation`,
mandate #4), encodes it with `features.encode_single`, and returns the
option the network scores highest (or samples from, when not
deterministic) -- always one of `pending_decision.options` (mandate #2).
`history` is ignored: Joshua is trained on `Observation` alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from struggler.bots.joshua import features as F
from struggler.bots.joshua.model import JoshuaNet, load_checkpoint, to_tensors
from struggler.engine import Action, Observation
from struggler.engine.player import Event


class JoshuaPlayer:
    def __init__(
        self,
        net: JoshuaNet,
        *,
        deterministic: bool = True,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        self._net = net.to(device).eval()
        self._device = torch.device(device)
        self._deterministic = deterministic
        # Own generator, never the engine's RNG (mandate #3): a bot's sampling
        # must not perturb the dice.
        self._generator = torch.Generator().manual_seed(seed)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        deterministic: bool = True,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> "JoshuaPlayer":
        net, _ = load_checkpoint(path, device=device)
        return cls(net, deterministic=deterministic, seed=seed, device=device)

    def choose_action(self, observation: Observation, history: Sequence[Event]) -> Action:
        options = observation.pending_decision.options
        with torch.no_grad():
            logits, _ = self._net(to_tensors(F.encode_single(observation), self._device))
        logits = logits[0].cpu()
        if self._deterministic:
            index = int(torch.argmax(logits))
        else:
            index = int(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=self._generator))
        return options[index]
