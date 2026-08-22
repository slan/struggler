"""Two rules engines, one game: Playdek's DLL drives, struggler replays.

A hotseat Playdek game (both seats ours) is played by a policy; every
option it picks becomes, through `translate`, the answers the struggler
engine will ask for -- the engine runs in physical mode, so its dice and
its deals come from the DLL's roll and card-location events too. The
engine is driven on demand: whenever it asks something the queues cannot
yet answer, the DLL is advanced first. At every point where both engines
are waiting on the same player, their states are compared; whenever the
engine asks a decision, the options the DLL offered for the same choice
are compared with the engine's. Every disagreement is a `Divergence`.

    python -m wopr.playdek.lockstep --games 3 --seed 1

One hand is hidden from the engine (physical mode's `physical_side`): its
cards are only learned as they are played, so its card-choice legality is
checked coarsely. Alternate `physical_side` across games to cover both.

The facts, queues and answers live in `bridge.Bridge`, shared with the
match operator (`operator.py`); this module is the prompt loop.
"""

from __future__ import annotations

import argparse
import random
from typing import Callable

from struggler.engine.types import Side
from wopr.playdek import ffi, ids, translate as T
from wopr.playdek.bridge import GRANTED_OPS_PROMPT, UI_ONLY, Bridge, Move, Report
from wopr.playdek.ffi import SelectionHint
from wopr.playdek.game import Option, Playdek, Prompt

__all__ = ["Lockstep", "Policy", "Report", "random_policy"]

Policy = Callable[[Prompt], Option]


def random_policy(rng: random.Random) -> Policy:
    def pick(prompt: Prompt) -> Option:
        playable = [o for o in prompt.visible if T.meaning(o).meaning not in UI_ONLY]
        return rng.choice(playable or list(prompt.visible))

    return pick


class Lockstep(Bridge):
    def __init__(self, pd: Playdek, *, game_no: int, seed: int, physical_side: Side, policy: Policy,
                 max_divergences: int = 40, trace: bool = False) -> None:
        super().__init__(pd, game_no=game_no, seed=seed, local_side=Side.USSR, ai_difficulty=None,
                         physical_side=physical_side, max_divergences=max_divergences, trace=trace)
        self.policy = policy
        self._held_first: Move | None = None
        self._skipped_compares = 0

    # -- the DLL side -----------------------------------------------------

    def advance_playdek(self) -> bool:
        """Let the policy answer one DLL prompt; queue what it implies.
        Returns False when the DLL's game is over."""
        prompt = self.game.pump(idle_limit=30)
        self._absorb_events()
        # Both engines at rest: the DLL has applied the previous choice and
        # waits for the next; give the engine everything that implies, then
        # compare.
        self.drain_engine()
        if prompt is None:
            return False
        boundary = any(o.hint in (SelectionHint.PLAY_CARD, SelectionHint.PLAY_SCORING_CARD, SelectionHint.HEADLINE_CARD) for o in prompt.visible)
        if boundary and not any(self.moves.values()):
            # Both sides between actions: the DLL asks for a card play, the
            # engine too, nothing queued between them. (A discard prompt
            # inside an event is not a rest: a trap's discard precedes the
            # roll and We Will Bury You's VP, which the engine has paid.)
            self.compare_state()
            self._skipped_compares = 0
        elif boundary:
            self._skipped_compares += 1
            if self._skipped_compares == 8:
                self.diverge("harness", f"no state comparison for 8 action boundaries: a move the engine never consumed is queued "
                             f"({ {s.value: [m.option.text for m in q] for s, q in self.moves.items() if q} })")
        if self._held_first is not None:
            # The DLL re-asks the very first hotseat prompt and drops the first
            # answer: only queue it once the next prompt proves it was taken.
            same = (prompt.player_id, prompt.text, tuple(prompt.options)) == \
                (self._held_first.prompt.player_id, self._held_first.prompt.text, tuple(self._held_first.prompt.options))
            if not same:
                self.queue(self._held_first.side, self._held_first)
            self._held_first = None
        self.report.prompts += 1
        side = self.prompt_side(prompt)
        if prompt.text == GRANTED_OPS_PROMPT and any(T.meaning(o).use == T.Use("space_race") for o in prompt.visible):
            # Ops that are not a card play of one's own -- a UN-Intervened
            # card's, Missile Envy's exchanged card's -- may go to the Space
            # Race in the DLL; the engine offers Ops only (`un_intervention`
            # mode, `push_event_operations`): for the UN-Intervened card the
            # same play is spacing the card itself. Counted, never picked.
            what = "UN Intervention: DLL offers the Space Race for the cancelled card" if self._un_ops[side] \
                else "Missile Envy: DLL offers the Space Race for the exchanged card"
            self.known[what] += 1
            prompt = Prompt(prompt.player_id, prompt.text, tuple(o for o in prompt.options if T.meaning(o).use != T.Use("space_race")))
        if (T.uses_offered(prompt) and self.engine.game_effects.get("missile_envy_forced") == side.value
                and self._last_played[side] == "Missile_Envy" and any(T.meaning(o).use == T.Use("space_race") for o in prompt.visible)):
            # "The opponent must use the Missile Envy card for Ops": the DLL
            # lets that play go to the Space Race, the engine asks the Ops
            # type straight away. Counted, never picked.
            self.known["Missile Envy: DLL lets the forced play go to the Space Race"] += 1
            prompt = Prompt(prompt.player_id, prompt.text, tuple(o for o in prompt.options if T.meaning(o).use != T.Use("space_race")))
        if self._un_ops[side] and T.uses_offered(prompt):
            self._un_ops[side] = False
        if prompt.text == "Remove Cuban Missile Crisis?" and not self.engine.turn_effects.get("cuban_missile_crisis"):
            # The DLL's crisis outlives the engine's: seen after the USSR played
            # the card for Ops, which fires no event in either program, yet
            # the DLL asked the US to pay its way out of a coup.
            self.known["Cuban Missile Crisis: the DLL asks the opponent to cancel a crisis the engine has no record of"] += 1
            prompt = Prompt(prompt.player_id, prompt.text, tuple(o for o in prompt.options if T.meaning(o).meaning is T.Meaning.STOP))
        option = self.policy(prompt)
        m = T.meaning(option)
        if (option.hint == SelectionHint.TRAP_PASS and side is not self.engine.physical_side
                and not any(self.engine.cards[c].scoring for c in self.engine.hands[side.value])):
            # "You May Play a Scoring Card" -> "Pass": the DLL lets the trapped
            # seat keep its scoring card for a later round; the engine (which
            # sees this hand) has played it already.
            self.known["trap step: the DLL lets the trapped seat keep its scoring card, the engine plays it"] += 1
            self.diverge("rules", f"{side.value} is trapped with no 2+-Ops card: the DLL offers to keep the scoring card (Pass), "
                         "the engine played it", fatal=True)
        drawn = [o for o in prompt.visible if o.hint == SelectionHint.SWITCH_CARD]
        if drawn and prompt.text.endswith("?") and any(o.hint == SelectionHint.STOP for o in prompt.visible):
            # Grain Sales: "Play <the drawn card>?" / "Return It". The card
            # never leaves the USSR hand when returned, so the engine's
            # RANDOM_DISCARD is answered from here, and its take/return too.
            self._grain = (ids.card_id(drawn[0].selection_id), option.hint == SelectionHint.SWITCH_CARD)
        if m.meaning is T.Meaning.UNKNOWN:
            self.diverge("unknown option", f"{prompt.text!r} -> {option.text!r} hint={option.hint:#x} id={option.selection_id} "
                         f"among {[(o.text, f'{o.hint:#x}', o.selection_id) for o in prompt.visible][:10]}")
        move = Move(side, prompt, option, m)
        if m.meaning in UI_ONLY:
            pass  # UI steps ("continue with this card" after an event-first resolution): nothing for the engine
        elif self.report.prompts == 1:
            self._held_first = move
        else:
            self.queue(side, move)
        if self.trace:
            print(f"  PD  {side.value:4s} {prompt.text!r} -> {option.text!r} [fifo {len(self._replay)}]")
        self.game.choose(option.index)
        return True

    # -- the loop ---------------------------------------------------------

    def drain_engine(self) -> None:
        """Step the engine as far as the queued facts allow."""
        while not self.stop and not self.engine.is_terminal:
            d = self.engine.pending_decision
            action = self._answer(d)
            if action is None:
                return
            if self.trace:
                print(f"  ENG {d.actor.value:6s} {d.kind.value} -> {dict(action.payload)}")
                was = (self.engine.defcon, self.engine.vp)
            self.engine.step(action)
            if self.trace and (self.engine.defcon, self.engine.vp) != was:
                print(f"  ENG DEFCON {self.engine.defcon} VP {self.engine.vp}")
            self.report.engine_steps += 1
            self.report.steps += 1

    def run(self, *, max_steps: int = 5000) -> Report:
        while not self.stop and self.report.steps < max_steps and not self.engine.is_terminal:
            if self.game.result is not None:
                self.drain_engine()
                if not self.engine.is_terminal:
                    r = self.game.result
                    if r.win_type is ffi.GameOverType.HELD_CARDS:
                        loser = self._sides_by_player.get(r.winner_id).opponent
                        if loser is self.engine.physical_side:
                            self.known["held scoring card in the hand the engine cannot see"] += 1
                        else:
                            self.diverge("rules", f"Playdek ends the game at the end of turn {self.engine.turn}: {loser.value} held a "
                                         "scoring card; the engine plays on")
                    else:
                        e = self.engine
                        self.diverge("game over", f"Playdek's game ended ({r.win_type.name}, {self._sides_by_player.get(r.winner_id)} wins, "
                                     f"score {r.score}) while the engine still asks {e.pending_decision.kind.value} "
                                     f"at DEFCON {e.defcon}, VP {e.vp}, turn {e.turn} AR {e.action_round}")
                break
            if not self.advance_playdek():
                continue
            self.report.steps += 1
        # Drain the DLL so its result is known.
        while self.game.result is None and not self.stop and self.report.prompts < max_steps:
            if not self.advance_playdek():
                break
        return self.finish()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--max-divergences", type=int, default=40)
    p.add_argument("--trace", action="store_true", help="print every DLL prompt/choice and engine decision/action")
    p.add_argument("--physical", choices=["us", "ussr"], help="the hand hidden from the engine (default: alternates by game index, US first)")
    args = p.parse_args(argv)
    pd = Playdek()
    for g in range(args.games):
        seed = args.seed + g
        physical = Side(args.physical.upper()) if args.physical else (Side.US if g % 2 == 0 else Side.USSR)
        ls = Lockstep(pd, game_no=g, seed=seed, physical_side=physical,
                      policy=random_policy(random.Random(seed)), max_divergences=args.max_divergences, trace=args.trace)
        r = ls.run(max_steps=args.max_steps)
        print(f"game {g} seed {seed}: {r.prompts} prompts, {r.engine_steps} engine steps; Playdek {r.playdek_result}; engine {r.engine_result}; "
              f"{len(r.divergences)} divergence(s)")
        for dv in r.divergences:
            print("  " + str(dv))


if __name__ == "__main__":
    main()
