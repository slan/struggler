"""Play scripted games against Playdek's AI and report what the DLL said.

    python -m wopr.playdek.smoke --games 2 --policy random --difficulty hard

Every prompt text and `selectionHint` seen is tallied at the end: that
catalogue is how the hint vocabulary in `docs/WOPR.md` was collected.
"""

from __future__ import annotations

import argparse
import collections
import random
import time

from struggler.engine.types import Side
from wopr.playdek.ffi import AIDifficulty
from wopr.playdek.game import Playdek, Prompt


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--side", choices=["ussr", "us"], default="ussr", help="the seat we answer; the AI takes the other")
    p.add_argument("--difficulty", choices=["easy", "hard"], default="hard")
    p.add_argument("--policy", choices=["first", "random"], default="random")
    p.add_argument("--max-prompts", type=int, default=None, help="abandon each game after this many prompts")
    p.add_argument("--show-prompts", type=int, default=0, help="print the first N prompts of each game with their options")
    p.add_argument("--idle-limit", type=float, default=300.0)
    args = p.parse_args(argv)

    pd = Playdek()
    print("install:", pd.root, flush=True)
    rng = random.Random(args.seed)
    prompts: collections.Counter[str] = collections.Counter()
    hints: collections.Counter[tuple[int, str]] = collections.Counter()
    side = Side.USSR if args.side == "ussr" else Side.US
    difficulty = AIDifficulty.HARD if args.difficulty == "hard" else AIDifficulty.EASY

    for g in range(args.games):
        game = pd.new_game(local_side=side, ai_difficulty=difficulty, seed=args.seed + g)
        start = time.monotonic()
        shown = 0
        count = 0
        while (prompt := game.pump(idle_limit=args.idle_limit)) is not None:
            count += 1
            prompts[prompt.text] += 1
            for o in prompt.options:
                hints[(o.hint, o.text.split(" ")[0])] += 1
            if shown < args.show_prompts:
                shown += 1
                print(f"[{g}] player {prompt.player_id}: {prompt.text!r}")
                for o in prompt.options:
                    print(f"      {o.index:3d} id={o.selection_id:4d} hint={o.hint} hidden={int(o.hidden)} {o.text!r}")
            if args.max_prompts is not None and count >= args.max_prompts:
                break
            visible = prompt.visible
            pick = visible[0] if args.policy == "first" else rng.choice(visible)
            game.choose(pick.index)
        elapsed = time.monotonic() - start
        if game.result is None:
            print(f"[{g}] abandoned after {count} prompts, {elapsed:.0f}s; hand={game.hand_count(0)} ai={game.ai_state().isAIPlayer}")
            game.close()
        else:
            r = game.result
            print(f"[{g}] {r.win_type.name} winner={r.winner_id} score={r.score} prompts={count} events={len(game.events)} {elapsed:.0f}s")

    print("--- prompts:")
    for text, n in prompts.most_common():
        print(f"  {n:4d} {text}")
    print("--- hints (hint, first word of label):")
    for (hint, word), n in sorted(hints.items()):
        print(f"  {hint:6d} {word:12s} {n}")


if __name__ == "__main__":
    main()
