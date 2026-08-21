"""Train Joshua with PPO in the WOPR arena.

    python -m wopr.train --run first --games 2000
    python -m wopr.train --run first --games 5000        # resumes `first` to 5000 games

A run directory (`runs/<run>/`) holds the SB3 model (`ppo.zip`), the latest
plain checkpoint (`joshua.pt`, what `--us joshua` loads), the checkpoint
pool (`pool/`), `metrics.csv`, and `config.json`.

Opponent mix per game (`--self-play`, `--vs-pool`, remainder vs `--anchor`):
- self-play: both seats are the learner; the alternating buffer trains
  both perspectives from one game.
- vs pool: the learner takes a random seat against a PFSP-sampled past
  snapshot (falls back to self-play while the pool is empty).
- vs anchor: against `random` or `greedy`, a fixed yardstick.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from stable_baselines3 import PPO

from struggler.engine import Side
from wopr.arena import Arena
from wopr.backend import ArenaSpec, Backend, InProcessBackend, SharedMemoryBackend
from wopr.buffer import AlternatingRolloutBuffer
from wopr.callback import StopAtGames, WoprCallback
from wopr.opponents import StandardOpponents
from wopr.policy import PRECISIONS, JoshuaPolicy
from wopr.pool import AnchorSchedule, CheckpointPool
from wopr.vec_env import LEARNER, WoprVecEnv

RUNS_DIR = Path("runs")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Joshua in the WOPR arena.")
    p.add_argument("--run", required=True, help="run name under runs/")
    p.add_argument("--games", type=int, required=True, help="stop once this many games have been played")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=64, help="games in flight")
    p.add_argument("--n-steps", type=int, default=128, help="learner decisions per env per update")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.03)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--gnn-layers", type=int, default=2)
    p.add_argument("--card-dim", type=int, default=32)
    p.add_argument("--self-play", type=float, default=0.5, help="fraction of games learner vs learner")
    p.add_argument("--vs-pool", type=float, default=0.4, help="fraction of games learner vs a pool snapshot")
    p.add_argument(
        "--anchor", default="random",
        help="opponent for the remaining games: random, greedy, or a schedule such as random,greedy "
        "(promoted in order once the learner's win rate over --anchor-window anchor games reaches --anchor-promote)",
    )
    p.add_argument("--anchor-promote", type=float, default=0.75, help="win rate that promotes a scheduled anchor to the next")
    p.add_argument("--anchor-window", type=int, default=100, help="anchor games the promotion win rate is measured over")
    p.add_argument("--snapshot-every", type=int, default=10, help="updates between pool snapshots (0: never)")
    p.add_argument("--pool-window", type=int, default=None, help="sample only the newest N snapshots")
    p.add_argument("--no-events", action="store_true", help="Ops-only curriculum: Engine.new_game(events=False)")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--precision", choices=list(PRECISIONS), default="bf16", help="bf16 autocast for the network (default; halves the update) or plain fp32")
    p.add_argument("--torch-threads", type=int, default=None)
    p.add_argument("--workers", type=int, default=1, help="collector processes stepping the games (1: in this process)")
    p.add_argument("--worker-threads", type=int, default=2, help="torch threads per collector (pool-net inference)")
    return p.parse_args(argv)


ANCHORS = ("random", "greedy", "first")


def make_anchor_schedule(args: argparse.Namespace) -> AnchorSchedule:
    anchors = tuple(name.strip() for name in args.anchor.split(","))
    unknown = [name for name in anchors if name not in ANCHORS]
    if unknown:
        raise ValueError(f"--anchor: unknown opponent(s) {unknown}; choose from {ANCHORS}")
    return AnchorSchedule(anchors, promote_at=args.anchor_promote, window=args.anchor_window)


def make_seat_assigner(self_play: float, vs_pool: float, anchor: AnchorSchedule, pool: CheckpointPool):
    if not 0.0 <= self_play <= 1.0 or not 0.0 <= vs_pool <= 1.0 or self_play + vs_pool > 1.0:
        raise ValueError("--self-play and --vs-pool must be fractions summing to at most 1")

    def assign(slot: int, episode: int, rng: random.Random) -> dict[Side, str]:
        draw = rng.random()
        if draw < self_play or (draw < self_play + vs_pool and len(pool) == 0):
            return {Side.US: LEARNER, Side.USSR: LEARNER}
        opponent = pool.sample(rng) if draw < self_play + vs_pool else anchor.current
        learner_side = rng.choice((Side.US, Side.USSR))
        return {learner_side: LEARNER, learner_side.opponent: opponent}

    return assign


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def build_env(args: argparse.Namespace, pool: CheckpointPool, anchor: AnchorSchedule, device: str) -> WoprVecEnv:
    seats = make_seat_assigner(args.self_play, args.vs_pool, anchor, pool)
    opponents = StandardOpponents(str(pool.directory), args.seed, device)
    if args.workers > 1:
        spec = ArenaSpec(args.n_envs, args.seed, events=not args.no_events)
        backend: Backend = SharedMemoryBackend(spec, seats, opponents, workers=args.workers, worker_threads=args.worker_threads)
    else:
        backend = InProcessBackend(Arena(args.n_envs, seed=args.seed, seat_assigner=seats, events=not args.no_events), opponents)
    return WoprVecEnv(backend)


def build_model(args: argparse.Namespace, env: WoprVecEnv, device: str) -> PPO:
    return PPO(
        JoshuaPolicy,
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        target_kl=args.target_kl,
        rollout_buffer_class=AlternatingRolloutBuffer,
        policy_kwargs={
            "joshua_config": {"hidden": args.hidden, "gnn_layers": args.gnn_layers, "card_dim": args.card_dim},
            "precision": args.precision,
        },
        seed=args.seed,
        device=device,
        verbose=0,
    )


def last_update(metrics_path: Path) -> int:
    """The update counter a resumed run continues from, read off the last
    row of `metrics.csv`: pool snapshots are named by update number, so a
    counter restarting at 0 would overwrite the run's earlier snapshots."""
    if not metrics_path.exists():
        return 0
    with metrics_path.open(newline="") as f:
        updates = [int(row["update"]) for row in csv.DictReader(f) if row.get("update")]
    return max(updates, default=0)


def wire_buffer(model: PPO, env: WoprVecEnv) -> None:
    buffer = model.rollout_buffer
    if not isinstance(buffer, AlternatingRolloutBuffer):
        raise TypeError("model.rollout_buffer must be an AlternatingRolloutBuffer")
    buffer.next_mover_source = env.current_am_us


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    run_dir = RUNS_DIR / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    model_path = run_dir / "ppo.zip"
    pool = CheckpointPool(run_dir / "pool", window=args.pool_window)

    games_done = 0
    updates_done = 0
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        games_done = int(previous.get("games_done", 0))
        if args.games <= games_done:
            print(f"[wopr] run {args.run!r} already at {games_done} games (target {args.games}); nothing to do")
            return
        updates_done = last_update(run_dir / "metrics.csv")
        print(f"[wopr] resuming {args.run!r} from {games_done} games ({updates_done} updates) to {args.games}")

    anchor = make_anchor_schedule(args)
    env = build_env(args, pool, anchor, device)
    if model_path.exists():
        model = PPO.load(model_path, env=env, device=device)
        # A resumed run takes its precision from the flag, not from the
        # policy saved in the zip (the weights are float32 either way).
        model.policy.precision = args.precision
    else:
        model = build_model(args, env, device)
    wire_buffer(model, env)

    tracker = WoprCallback(
        run_dir=run_dir,
        env=env,
        pool=pool,
        target_games=args.games,
        snapshot_every=args.snapshot_every,
        games_done=games_done,
        updates_done=updates_done,
        anchor_schedule=anchor,
    )
    config: dict[str, Any] = {**vars(args), "device": device, "games_done": games_done}
    config_path.write_text(json.dumps(config, indent=2))
    print(f"[wopr] device={device} n_envs={args.n_envs} n_steps={args.n_steps} params={sum(p.numel() for p in model.policy.parameters())}")
    try:
        model.learn(total_timesteps=2**62, callback=[tracker, StopAtGames(tracker)], reset_num_timesteps=not model_path.exists())
    finally:
        model.save(model_path)
        config["games_done"] = tracker.games
        config_path.write_text(json.dumps(config, indent=2))
        print(f"[wopr] saved {model_path} after {tracker.games} games")


if __name__ == "__main__":
    main()
