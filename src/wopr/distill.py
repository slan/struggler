"""Distill Playdek's AI from the bridge's logged games into a pool opponent.

    python -m wopr.distill harvest --out runs/falken1/corpus runs/playdek/v3-easy-r*
    python -m wopr.distill train --corpus runs/falken1/corpus --out runs/falken1

The relaxing-SELF-PLAY-ONLY entry (docs/JOSHUA.md, 2026-08-30). The DLL
cannot be an arena opponent (one game per process, 15 s per AI decision;
docs/WOPR.md), but every eval batch already wrote the engine's replayable
log of each game, and the AI's decisions are in them. `harvest` replays
those logs on the current engine and records, at every AI-seat decision
with at least two options, the encoded observation and the option the AI
chose; `train` fits a fresh `JoshuaNet` to the rows by cross-entropy —
a clone that `--pool-seed` can put in the training mix like any snapshot.

The one reconstruction in the pipeline: the engine's physical-mode mirror
does not know most of the AI's hand (cards are learned as they are
played), so the hand is determinized in hindsight before encoding — known
cards kept, the chosen card and the AI's later same-turn card plays
forced in, the rest sampled uniformly from the cards the AI's observation
leaves unseen, seeded by (game seed, step index) so the harvest is
reproducible. Option lists are encoded as the physical-mode engine
enumerated them; with the chosen card always in the rebuilt hand, options
outside it are consistent labeled negatives.

Desynced and void games are excluded (their mirror may have drifted
before the fatal); a log that no longer replays on the current rules is
skipped and counted.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import zlib
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from struggler.bots.joshua import features as F
from struggler.engine.cards import load_cards
from struggler.engine.replay import decode_action, make_engine
from struggler.engine.types import Period, Side

#: Decision kinds whose chosen card comes from the mover's own hand: the
#: kinds whose recorded choice is forced into the rebuilt hand.
HAND_PLAY_KINDS = frozenset({"action_round_play", "headline_play", "quagmire_discard", "random_discard"})

_CARDS = load_cards()
_ENTRY_TURN = {cid: {Period.EARLY_WAR: 1, Period.MID_WAR: 4, Period.LATE_WAR: 8}[card.period]
               for cid, card in _CARDS.items()}

SHARD_ROWS = 8192


# -- harvest -------------------------------------------------------------------


def rebuild_hand(observation, forced: Sequence[str], rng: random.Random, *, include_optional: bool) -> list[str]:
    """The observation's hand with its unknowns determinized in hindsight.

    Known cards stay; `forced` (the chosen card first, then the AI's later
    same-turn card plays) fill unknown slots when the observation does not
    already place them elsewhere; the remaining slots are dealt uniformly
    from the cards the observation leaves unseen. The hand keeps its true
    size unless the unseen pool runs dry (endgame), where slots are
    dropped rather than left unknowable."""
    hand = list(observation.hand)
    slots = sum(1 for c in hand if c == "?")
    known = [c for c in hand if c != "?"]
    elsewhere = set(observation.discard_pile) | set(observation.removed_cards)
    for cid in forced:
        if slots == 0:
            break
        if cid in known or cid in elsewhere or cid not in _CARDS or not _CARDS[cid].in_deck:
            continue
        known.append(cid)
        slots -= 1
    unseen = [cid for cid, card in _CARDS.items()
              if card.in_deck and (include_optional or not card.optional)
              and _ENTRY_TURN[cid] <= observation.turn
              and cid not in known and cid not in elsewhere]
    rng.shuffle(unseen)
    return known + unseen[:slots]


def harvest_game(log: dict, *, game_hash: int) -> tuple[dict[str, np.ndarray], Counter]:
    """One log's AI-seat decisions as encoded rows. Returns (arrays, counts);
    the arrays are the layout buffers plus `label`, `n_options` and
    `game_hash` per row."""
    ai = Side(log["physical_side"])
    include_optional = bool(log.get("include_optional", True))
    steps = log["actions"]
    engine = make_engine(log)
    counts: Counter = Counter()
    buffers = F.allocate(len(steps))
    labels: list[int] = []
    n_options: list[int] = []
    row = 0
    for index, step in enumerate(steps):
        decision = engine.pending_decision
        if decision is None:
            counts["truncated"] += 1
            break
        action = decode_action(step)
        if decision.actor is ai and len(decision.options) > 1:
            try:
                label = decision.options.index(action)
            except ValueError:
                label = -1
                counts["option_mismatch"] += 1
            if 0 <= label and len(decision.options) <= F.K_MAX:
                forced = []
                if step["kind"] in HAND_PLAY_KINDS and isinstance(step["payload"].get("card"), str):
                    forced.append(step["payload"]["card"])
                turn = step.get("turn")  # eval logs annotate turn/actor; goldens may not
                if turn is not None:
                    forced += [later["payload"]["card"] for later in steps[index + 1:]
                               if later.get("turn") == turn and later.get("actor") == ai.value
                               and later["kind"] in HAND_PLAY_KINDS
                               and isinstance(later["payload"].get("card"), str)]
                rng = random.Random(zlib.crc32(f"{log['seed']}:{index}".encode()))
                observation = engine.observe(ai)
                observation = replace(
                    observation, hand=tuple(rebuild_hand(observation, forced, rng, include_optional=include_optional))
                )
                F.encode_into(observation, buffers, row)
                labels.append(label)
                n_options.append(len(decision.options))
                row += 1
            elif 0 <= label:
                counts["over_k_max"] += 1
        engine.step(action)
    counts["rows"] += row
    arrays = {name: array[:row].copy() for name, array in buffers.items()}
    arrays["label"] = np.asarray(labels, dtype=np.int64)
    arrays["n_options"] = np.asarray(n_options, dtype=np.int64)
    arrays["game_hash"] = np.full(row, game_hash, dtype=np.uint32)
    return arrays, counts


def clean_games(source: Path) -> Iterator[Path]:
    """The game logs of one eval batch that finished without desync or void."""
    flags: dict[int, bool] = {}
    results = source / "results.jsonl"
    if results.exists():
        for line in results.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            flags[r["index"]] = bool(r.get("desync")) or bool(r.get("void"))
    for path in sorted((source / "games").glob("*.json")):
        index = int(path.name.split("_", 1)[0])
        if not flags.get(index, False):
            yield path


def _concat(shard: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.concatenate([arrays[name] for arrays in shard]) for name in shard[0]}


def harvest(argv: Sequence[str]) -> None:
    p = argparse.ArgumentParser(prog="wopr.distill harvest",
                                description="Replay eval-batch game logs and record the AI seat's decisions.")
    p.add_argument("sources", nargs="+", help="eval batch directories (each with games/ and results.jsonl)")
    p.add_argument("--out", required=True, help="corpus directory for the .npz shards and manifest.json")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    totals: Counter = Counter()
    pending: list[dict[str, np.ndarray]] = []
    pending_rows = 0
    shards = 0

    def flush() -> None:
        nonlocal pending, pending_rows, shards
        if not pending:
            return
        np.savez_compressed(out / f"shard{shards:04d}.npz", **_concat(pending))
        shards += 1
        pending, pending_rows = [], 0

    for source in map(Path, args.sources):
        if not (source / "games").is_dir():
            raise SystemExit(f"{source}: no games/ directory")
        for path in clean_games(source):
            game_hash = zlib.crc32(f"{source.name}/{path.name}".encode())
            log = json.loads(path.read_text(encoding="utf-8"))
            try:
                arrays, counts = harvest_game(log, game_hash=game_hash)
            except Exception as e:
                totals["replay_failed"] += 1
                totals[f"replay_failed:{type(e).__name__}"] += 1
                continue
            totals["games"] += 1
            totals.update(counts)
            if len(arrays["label"]):
                pending.append(arrays)
                pending_rows += len(arrays["label"])
            if pending_rows >= SHARD_ROWS:
                flush()
        print(f"[harvest] {source.name}: rows so far {totals['rows']}, games {totals['games']}, "
              f"failed {totals['replay_failed']}", flush=True)
    flush()
    manifest = {"sources": [str(s) for s in args.sources], "layout_version": F.LAYOUT_VERSION,
                "shards": shards, **{k: totals[k] for k in sorted(totals)}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[harvest] {totals['rows']} rows from {totals['games']} games in {shards} shards -> {out}; "
          f"failed {totals['replay_failed']}, option mismatches {totals['option_mismatch']}, "
          f"over K_MAX {totals['over_k_max']}", flush=True)


# -- train ---------------------------------------------------------------------


VAL_FOLDS = 10  # game_hash % VAL_FOLDS == 0 is the held-out split


def _split(arrays: dict[str, np.ndarray], val: bool) -> dict[str, np.ndarray]:
    keep = (arrays["game_hash"] % VAL_FOLDS == 0) == val
    return {name: array[keep] for name, array in arrays.items()}


def _batches(arrays: dict[str, np.ndarray], batch_size: int, rng: np.random.Generator | None) -> Iterator[dict[str, np.ndarray]]:
    n = len(arrays["label"])
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, batch_size):
        index = order[start:start + batch_size]
        yield {name: array[index] for name, array in arrays.items()}


def train(argv: Sequence[str]) -> None:
    import torch

    from struggler.bots.joshua.model import JoshuaConfig, JoshuaNet, save_checkpoint, to_tensors
    from wopr.repo import git_commit
    from wopr.train import resolve_device

    p = argparse.ArgumentParser(prog="wopr.distill train",
                                description="Fit a fresh JoshuaNet to a harvested corpus by cross-entropy.")
    p.add_argument("--corpus", required=True, help="directory of harvest shards")
    p.add_argument("--out", required=True, help="run directory; writes joshua.pt and distill.json")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20, help="the cap; early-stopped by held-out top-1")
    p.add_argument("--patience", type=int, default=2, help="epochs without a new best held-out top-1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-threads", type=int, default=None)
    args = p.parse_args(argv)

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    shard_paths = sorted(Path(args.corpus).glob("shard*.npz"))
    if not shard_paths:
        raise SystemExit(f"{args.corpus}: no shards")
    net = JoshuaNet(JoshuaConfig(hidden=args.hidden)).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    def evaluate() -> tuple[float, float, int]:
        """Held-out (top-1, legal-uniform floor, rows)."""
        net.eval()
        hits = floor = rows = 0
        with torch.no_grad():
            for path in shard_paths:
                arrays = _split(dict(np.load(path)), val=True)
                if not len(arrays["label"]):
                    continue
                for batch in _batches(arrays, args.batch_size, None):
                    logits, _ = net(to_tensors({k: v for k, v in batch.items() if k in F.LAYOUT}, device))
                    labels = torch.as_tensor(batch["label"], device=device)
                    hits += int((logits.argmax(-1) == labels).sum())
                    floor += float((1.0 / batch["n_options"]).sum())
                    rows += len(batch["label"])
        return (hits / rows if rows else 0.0, floor / rows if rows else 0.0, rows)

    best = (-1.0, -1)  # (top-1, epoch)
    best_state: dict | None = None
    history: list[dict] = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        net.train()
        losses: list[float] = []
        for path in rng.permutation(shard_paths):
            arrays = _split(dict(np.load(path)), val=False)
            if not len(arrays["label"]):
                continue
            for batch in _batches(arrays, args.batch_size, rng):
                logits, _ = net(to_tensors({k: v for k, v in batch.items() if k in F.LAYOUT}, device))
                loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(batch["label"], device=device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        top1, floor, val_rows = evaluate()
        history.append({"epoch": epoch, "loss": sum(losses) / len(losses), "val_top1": top1})
        print(f"[distill] epoch {epoch}: loss {history[-1]['loss']:.4f}, held-out top-1 {top1:.4f} "
              f"(uniform floor {floor:.4f}, {val_rows} rows)", flush=True)
        if top1 > best[0]:
            best = (top1, epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        elif epoch - best[1] >= args.patience:
            print(f"[distill] no new best for {args.patience} epochs; stopping", flush=True)
            break
    assert best_state is not None
    net.load_state_dict(best_state)
    out = Path(args.out)
    top1 = best[0]  # floor and val_rows are properties of the split, constant across epochs
    save_checkpoint(net, out / "joshua.pt",
                    extra={"distilled_from": args.corpus, "val_top1": top1, "commit": git_commit()})
    (out / "distill.json").write_text(json.dumps({
        "corpus": args.corpus, "commit": git_commit(), "device": device, "seed": args.seed,
        "hidden": args.hidden, "lr": args.lr, "batch_size": args.batch_size,
        "best_epoch": best[1], "val_top1": top1, "uniform_floor": floor, "val_rows": val_rows,
        "train_s": round(time.perf_counter() - started, 1), "history": history,
    }, indent=2), encoding="utf-8")
    print(f"[distill] best epoch {best[1]}: held-out top-1 {top1:.4f} vs uniform {floor:.4f}; "
          f"saved {out / 'joshua.pt'}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("harvest", "train"):
        raise SystemExit("usage: python -m wopr.distill {harvest|train} ...")
    (harvest if argv[0] == "harvest" else train)(argv[1:])


if __name__ == "__main__":
    main()
