"""SearchPlayer: inference-time lookahead over a trained JoshuaNet.

Two evaluators, one simulation harness, no training change (the
checkpoint and `LAYOUT_VERSION` are untouched); pre-registered in
docs/JOSHUA.md (2026-08-25), mechanics in docs/WOPR.md:

- `evaluator="value"` (one-ply search): every legal option is played out
  on a determinized copy of the game to the next non-CHANCE decision,
  where the value head scores the position for its mover, sign-flipped
  to the root mover; argmax over scores.
- `evaluator="terminal"` (the veto): an option that *provably* loses --
  terminally, or through a forced sequence the opponent can drive to a
  terminal loss within the current play -- is masked, and the policy's
  own argmax picks among the survivors. This is the ablation the search
  subsumes (a provable loss scores -1).

The simulation state comes from `Engine.determinize(side, seed)`, never
the live engine: unseen cards are reshuffled, the RNG reseeded, and d6
CHANCE frames expose all six outcomes (`expose_chance_outcomes`), so
nothing hidden from the mover can steer the search (mandate #4 in
spirit). A branch that consumed no randomness -- no chance frame, no
card leaving or entering the draw pile -- is exact from one simulation;
one that did is averaged over `k` determinizations, with chance frames
enumerated exactly while the branch's outcome count stays within
`chance_cap` and sampled beyond it. Terminal leaves score +/-1 (draws
0) and the value head's estimate is clamped just inside that, so a
certain result always outranks an estimate.

The player needs the engine it is seated at solely to call
`determinize()`: the runner that owns the engine calls `bind(engine)`
before play (`src/main.py`, `wopr.playdek.operator.play_match`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from struggler.bots.joshua import features as F
from struggler.bots.joshua.model import JoshuaNet, load_checkpoint, to_tensors
from struggler.engine import Action, Engine, Observation, Side
from struggler.engine.player import Event
from struggler.engine.types import Decision

# The value head's estimate is kept strictly inside the terminal payoffs:
# a found win (+1) must outrank any estimate, a found loss (-1) rank below
# every one, whatever range the unbounded head drifts to off-distribution.
_VALUE_CLAMP = 0.99


class SearchPlayer:
    """A `Player` wrapping a JoshuaNet with one-ply lookahead (or the
    terminal-only veto) over `Engine.determinize` copies. See module doc."""

    def __init__(
        self,
        net: JoshuaNet,
        *,
        evaluator: str = "value",
        k: int = 6,
        chance_cap: int = 36,
        # ~0.9 ms per probed node (an engine copy): 300 bounds a fully
        # unprovable decision at ~0.3 s while the DEFCON-gift proof itself
        # needs only tens of nodes.
        probe_budget: int = 300,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        if evaluator not in ("value", "terminal"):
            raise ValueError(f"evaluator must be 'value' or 'terminal', not {evaluator!r}")
        if k < 1:
            raise ValueError("k must be >= 1")
        self._net = net.to(device).eval()
        self._device = torch.device(device)
        self._evaluator = evaluator
        self._k = k
        self._chance_cap = chance_cap
        self._probe_budget = probe_budget
        self._seed = seed
        self._engine: Engine | None = None
        # Sampling of unenumerated chance is the player's own stream, never
        # the engine's (mandate #3) -- and irrelevant to the live dice anyway,
        # since it only ever runs on determinized copies.
        self._rng = torch.Generator().manual_seed(seed)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        evaluator: str = "value",
        k: int = 6,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> "SearchPlayer":
        net, _ = load_checkpoint(path, device=device)
        return cls(net, evaluator=evaluator, k=k, seed=seed, device=device)

    def bind(self, engine: Engine) -> None:
        """Attach the engine this player is seated at. Used solely for
        `engine.determinize(side, seed)`; the player never reads it."""
        self._engine = engine

    # -- Player ------------------------------------------------------------

    def choose_action(self, observation: Observation, history: Sequence[Event]) -> Action:
        decision = observation.pending_decision
        options = decision.options
        if len(options) == 1:
            return options[0]
        if self._engine is None:
            raise RuntimeError("SearchPlayer is unbound: call bind(engine) before play")
        logits = self._policy_logits(observation)
        if self._evaluator == "terminal":
            index = self._choose_veto(observation, decision, logits)
        else:
            index = self._choose_value(observation, decision, logits)
        return options[index]

    # -- shared machinery --------------------------------------------------

    def _policy_logits(self, observation: Observation) -> list[float]:
        with torch.no_grad():
            logits, _ = self._net(to_tensors(F.encode_single(observation), self._device))
        return logits[0, : len(observation.pending_decision.options)].tolist()

    def _det_seed(self, decision: Decision, option_index: int, sample: int) -> int:
        return ((self._seed + 1) * 1_000_003 + decision.id * 9973 + option_index * 101 + sample) & 0x7FFFFFFF

    def _step(self, sim: Engine, action: Action) -> bool:
        """Step `sim`, reporting whether the draw pile changed -- a card
        dealt, drawn, or reshuffled: hidden information realized."""
        before = len(sim.draw_pile)
        sim.step(action)
        return len(sim.draw_pile) != before

    def _sample(self, options: tuple[Action, ...]) -> Action:
        index = int(torch.randint(len(options), (1,), generator=self._rng))
        return options[index]

    def _terminal_value(self, sim: Engine, root_side: Side) -> float:
        if sim.winner is None:
            return 0.0
        return 1.0 if sim.winner is root_side else -1.0

    def _leaf_value(self, sim: Engine, actor: Side, root_side: Side) -> float:
        with torch.no_grad():
            _, value = self._net(to_tensors(F.encode_single(sim.observe(actor)), self._device))
        v = max(-_VALUE_CLAMP, min(_VALUE_CLAMP, float(value[0])))
        return v if actor is root_side else -v

    # -- value evaluator: one-ply search -----------------------------------

    def _choose_value(self, observation: Observation, decision: Decision, logits: list[float]) -> int:
        side = observation.side
        scores: list[float] = []
        for i, option in enumerate(decision.options):
            samples: list[float] = []
            for j in range(self._k):
                sim = self._engine.determinize(side, self._det_seed(decision, i, j))
                try:
                    consumed = self._step(sim, option)
                    value, consumed_more = self._continue(sim, side, self._chance_cap)
                except Exception:
                    break
                samples.append(value)
                if j == 0 and not (consumed or consumed_more):
                    break  # deterministic branch: one simulation is exact
            # An unscoreable branch (a simulation the determinized copy
            # cannot replay, e.g. a physical-only frame) counts as unknown,
            # 0 on the value scale: still below every winning score and --
            # crucially -- above a branch *known* to lose. With every branch
            # unknown this degrades to the raw policy argmax via the
            # tie-break.
            scores.append(sum(samples) / len(samples) if samples else 0.0)
        # Value first, the policy's own logit as the tie-break.
        return max(range(len(scores)), key=lambda i: (scores[i], logits[i]))

    def _continue(self, sim: Engine, root_side: Side, budget: int) -> tuple[float, bool]:
        """Run `sim` to the next non-CHANCE decision (or the end) and score
        it for `root_side`. Multi-outcome chance frames are enumerated to an
        exact expectation while `budget` covers the branch's outcomes, and
        sampled beyond it; returns (value, whether randomness was consumed)."""
        consumed = False
        while True:
            if sim.is_terminal:
                return self._terminal_value(sim, root_side), consumed
            frame = sim.pending_decision
            if frame.actor is not Side.CHANCE:
                return self._leaf_value(sim, frame.actor, root_side), consumed
            options = frame.options
            consumed = True  # every CHANCE frame is realized randomness
            if len(options) == 1:
                self._step(sim, options[0])
                continue
            if budget >= len(options):
                total = 0.0
                for option in options:
                    child = Engine.deserialize(sim.serialize())
                    self._step(child, option)
                    value, _ = self._continue(child, root_side, budget // len(options))
                    total += value
                return total / len(options), True
            self._step(sim, self._sample(options))

    # -- terminal evaluator: the veto --------------------------------------

    def _choose_veto(self, observation: Observation, decision: Decision, logits: list[float]) -> int:
        # Probe in the policy's own preference order and stop at the first
        # option that is not a provable loss -- the same answer as probing
        # everything and taking the surviving argmax, at (usually) one
        # probe per decision instead of one per option.
        side = observation.side
        ranked = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)
        for i in ranked:
            sim = self._engine.determinize(side, self._det_seed(decision, i, 0))
            boundary = self._boundary(sim)
            try:
                self._step(sim, decision.options[i])
                lost = self._probe(sim, side, boundary, [self._probe_budget])
            except Exception:
                lost = False  # unprovable is never a veto
            if not lost:
                return i
        return ranked[0]  # genuinely forced: every option provably loses

    @staticmethod
    def _boundary(sim: Engine) -> tuple:
        # The veto's horizon: the current play. `_ars_played` advances when
        # an action-round play completes, so the opponent's own next card --
        # their turn, not a forced reply -- is beyond it.
        return (sim.turn, sim._ars_played, sim.phase)

    def _probe(self, sim: Engine, root_side: Side, boundary: tuple, budget: list[int], my_depth: int = 0) -> bool:
        """True iff this position provably loses for `root_side` within the
        current play: terminal against them, or -- through any opponent
        choice, every own choice, and every exposed chance outcome -- forced
        to one. Anything unprovable within `budget` nodes, beyond the play
        boundary, or past three of the prober's own choice points (an own
        ops chain fans out by tens per point and cannot itself end the
        game) is False: the veto never fires on a guess."""
        while True:
            if sim.is_terminal:
                return sim.winner is root_side.opponent
            if budget[0] <= 0 or self._boundary(sim) != boundary:
                return False
            budget[0] -= 1
            frame = sim.pending_decision
            options = frame.options
            if len(options) == 1:
                self._step(sim, options[0])
                continue
            # Opponent picks the reply (ANY option may lose us); we and the
            # dice must be *forced* (ALL options / outcomes lose).
            forced = frame.actor is not root_side.opponent
            if frame.actor is root_side:
                if my_depth >= 3:
                    return False
                my_depth += 1
            for option in options:
                child = Engine.deserialize(sim.serialize())
                self._step(child, option)
                lost = self._probe(child, root_side, boundary, budget, my_depth)
                if lost and not forced:
                    return True
                if not lost and forced:
                    return False
            return forced
