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
        k: int = 4,
        # The policy's own pick stands unless another option's searched
        # value beats it by this much. Ordinary per-option value noise is
        # ~0.1; a real blunder (a found loss against an ordinary position)
        # differs by ~1.0 -- the search overrides only where the value head
        # is loudly sure, instead of replacing trained play with argmax
        # over its noise.
        margin: float = 0.3,
        # One die is enumerated exactly; a second nested die (36 outcomes,
        # each a full policy rollout) is sampled instead -- the exact
        # expectation there costs more wall clock than its variance is worth.
        chance_cap: int = 6,
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
        self._margin = margin
        self._chance_cap = chance_cap
        self._probe_budget = probe_budget
        self._seed = seed
        self._engine: Engine | None = None
        self.last_scores: list[float] | None = None  # value mode's per-option scores, for analysis
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
        self.last_scores = scores  # per-option, root-side-signed; for analysis
        # The policy's pick is the default; the search overrides it only
        # when another option's value clears it by the margin. The pick is
        # then probed, and a *provable* loss is refused and re-picked: the
        # rollout's simulated opponent is the policy, which never takes the
        # gifted coup (the original finding), so the value estimate alone
        # reads a gift branch as clean -- the veto's rules-probe, which asks
        # what the opponent CAN do rather than what the policy would, is
        # the floor under the search (search subsumes veto, restored).
        candidates = set(range(len(scores)))
        policy_pick = max(range(len(logits)), key=logits.__getitem__)
        while True:
            default = policy_pick if policy_pick in candidates else max(candidates, key=logits.__getitem__)
            best = max(candidates, key=lambda i: (scores[i], logits[i]))
            pick = best if scores[best] - scores[default] > self._margin else default
            if len(candidates) == 1 or not self._provably_loses(observation.side, decision, pick):
                return pick
            candidates.remove(pick)

    def _continue(
        self,
        sim: Engine,
        root_side: Side,
        budget: int,
        # The caps must outlast one whole action or an end-of-turn terminal
        # (a held scoring card) is never reached and the blind flip estimate
        # stands. A plain ops spend is ~8 multi-option frames (card, mode,
        # order, type, the placements); an event can hand the opponent a
        # long chain *before* their own action -- Marshall Plan's seven
        # placements plus their play is 13+ -- hence the deeper reply cap.
        my_steps: int = 12,
        opp_steps: int = 18,
        flip_value: float | None = None,
    ) -> tuple[float, bool]:
        """Score the branch for `root_side`. The branch is rolled forward --
        through chance, through the mover's *own* subsequent decisions
        (along the policy's argmax, up to `my_steps`: stopping at the
        mover's own next decision would price an action by its first
        atomic step, e.g. an `OPS_TYPE` "coup" before any target is
        picked), and then through the *opponent's* reply the same way (up
        to `opp_steps`) -- and evaluated twice: `flip_value`, the value
        head at the opponent's first real decision (their view: prices the
        threat they now hold, but is blind to the mover's remaining hand),
        and again at the mover's own next decision (their view of their
        hand -- a held scoring card -- with any end-of-turn terminal
        played out for real in between). The branch scores the **minimum**
        of the two: each estimate covers the other's blind spot, and a
        playout terminal is likewise floored against `flip_value` so a
        weak simulated reply cannot make a branch look won. Multi-outcome
        chance frames are enumerated to an exact expectation while
        `budget` covers the branch's outcomes, and sampled beyond it.
        Returns (value, whether randomness was consumed)."""
        consumed = False
        while True:
            if sim.is_terminal:
                value = self._terminal_value(sim, root_side)
                return (min(flip_value, value) if flip_value is not None else value), consumed
            frame = sim.pending_decision
            if frame.actor is Side.CHANCE:
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
                        value, _ = self._continue(
                            child, root_side, budget // len(options), my_steps, opp_steps, flip_value
                        )
                        total += value
                    return total / len(options), True
                self._step(sim, self._sample(options))
                continue
            options = frame.options
            if len(options) == 1:
                self._step(sim, options[0])
                continue
            if frame.actor is root_side:
                if flip_value is not None:  # back at the mover: the second, hand-aware estimate
                    return min(flip_value, self._leaf_value(sim, root_side, root_side)), consumed
                if my_steps > 0:
                    if self._policy_step(sim, frame.actor, options):
                        consumed = True
                    my_steps -= 1
                    continue
                return self._leaf_value(sim, root_side, root_side), consumed
            if flip_value is None:
                flip_value = self._leaf_value(sim, frame.actor, root_side)
            if opp_steps > 0:
                if self._policy_step(sim, frame.actor, options):
                    consumed = True
                opp_steps -= 1
                continue
            return flip_value, consumed

    def _policy_step(self, sim: Engine, actor: Side, options: tuple[Action, ...]) -> bool:
        logits = self._policy_logits(sim.observe(actor))
        return self._step(sim, options[max(range(len(logits)), key=logits.__getitem__)])

    # -- terminal evaluator: the veto --------------------------------------

    def _choose_veto(self, observation: Observation, decision: Decision, logits: list[float]) -> int:
        # Probe in the policy's own preference order and stop at the first
        # option that is not a provable loss -- the same answer as probing
        # everything and taking the surviving argmax, at (usually) one
        # probe per decision instead of one per option.
        side = observation.side
        ranked = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)
        for i in ranked:
            if not self._provably_loses(side, decision, i):
                return i
        return ranked[0]  # genuinely forced: every option provably loses

    def _provably_loses(self, side: Side, decision: Decision, i: int) -> bool:
        sim = self._engine.determinize(side, self._det_seed(decision, i, 7919))
        boundary = self._boundary(sim)
        try:
            self._step(sim, decision.options[i])
            return self._probe(sim, side, boundary, [self._probe_budget])
        except Exception:
            return False  # unprovable is never a veto

    @staticmethod
    def _boundary(sim: Engine) -> tuple:
        # The veto's horizon: the current play. `_ars_played` advances when
        # an action-round play completes, so the opponent's own next card --
        # their turn, not a forced reply -- is beyond it.
        return (sim.turn, sim._ars_played, sim.phase)

    def _probe(self, sim: Engine, root_side: Side, boundary: tuple, budget: list[int]) -> bool:
        """True iff this position loses for `root_side` within the current
        play *on the line they would actually play*: terminal against them,
        or forced to one through any opponent choice and every exposed
        chance outcome, with the prober's own subsequent decisions followed
        along the policy's argmax. A minimax ALL over own options was tried
        and cannot get through a real ops chain (tens of placements wide,
        several deep) inside any budget -- the gift survived it; the policy
        line is what the bot will do if the option stands, so a loss found
        on it is a loss that will be realized. Anything unprovable within
        `budget` nodes or beyond the play boundary is False: the probe
        never fires on a guess."""
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
            if frame.actor is root_side:
                self._policy_step(sim, root_side, options)
                continue
            if frame.actor is Side.CHANCE:
                # The dice must be *forced*: every outcome loses.
                for option in options:
                    child = Engine.deserialize(sim.serialize())
                    self._step(child, option)
                    if not self._probe(child, root_side, boundary, budget):
                        return False
                return True
            # The opponent picks the reply: ANY option may lose us.
            for option in options:
                child = Engine.deserialize(sim.serialize())
                self._step(child, option)
                if self._probe(child, root_side, boundary, budget):
                    return True
            return False
