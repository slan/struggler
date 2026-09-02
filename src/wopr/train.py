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
  `--pool-seed name=ckpt.pt ...` pre-loads the pool with frozen
  checkpoints -- the exploiter/counter-run wiring (docs/WOPR.md).
- vs anchor: against `random`, `greedy`, or a frozen checkpoint
  (`name=ckpt.pt`) -- a fixed yardstick whose share PFSP never touches.

`--eval-every N` plays the latest checkpoint against Greedy (`--eval-games`,
argmax, half on each seat, a fresh deck seed per tick) every N training
games, on the collectors while the PPO update runs, and records it in
`metrics.csv` (`eval_*`): the run's curve against the yardstick, free of
the opponent mix. `wopr.bootstrap` reads it to decide when to stop.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from struggler.bots.joshua import features as F
from struggler.bots.joshua.model import load_checkpoint
from struggler.engine import RULES_VERSION, Side
from wopr.arena import Arena
from wopr.backend import ArenaSpec, Backend, InProcessBackend, SharedMemoryBackend
from wopr.buffer import AlternatingRolloutBuffer
from wopr.callback import StopAtGames, WoprCallback
from wopr.opponents import CKPT_PREFIX, StandardOpponents
from wopr.policy import PRECISIONS, JoshuaPolicy
from wopr.pool import AnchorSchedule, CheckpointPool
from wopr.repo import git_commit
from wopr.vec_env import LEARNER, WoprVecEnv

RUNS_DIR = Path("runs")

#: Named recipes: the learning settings of a frozen version, so a clean run
#: is one flag (`--recipe v11`) and `config.json` says which. Machine
#: settings (workers, threads, device) are not part of a recipe. A flag
#: given explicitly beside `--recipe` wins over the recipe's value.
RECIPES: dict[str, dict[str, Any]] = {
    # v5's mix -- no anchor, 50% self-play, 50% pool, a snapshot every 5
    # updates -- at hidden 256 and 4 PPO epochs: the capacity A/B winner.
    "v11": {"hidden": 256, "n_epochs": 4, "self_play": 0.5, "vs_pool": 0.5, "snapshot_every": 5},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.recipe:
        explicit = vars(explicit_parser().parse_args(argv))
        for key, value in RECIPES[args.recipe].items():
            if key not in explicit:
                setattr(args, key, value)
    return args


def explicit_parser() -> argparse.ArgumentParser:
    """The same parser with every default suppressed: its namespace holds
    only the flags actually given, which is what a recipe must not override."""
    p = build_parser()
    for action in p._actions:
        action.default = argparse.SUPPRESS
    return p


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Joshua in the WOPR arena.")
    p.add_argument("--recipe", choices=sorted(RECIPES), default=None, help="a frozen version's learning settings (see RECIPES); explicit flags override it")
    p.add_argument("--init", default=None, help="start a new run from this joshua.pt checkpoint (weights only: fresh optimizer and pool); the network size comes from the checkpoint")
    p.add_argument("--run", required=True, help="run name under runs/")
    p.add_argument("--games", type=int, required=True, help="stop once this many games have been played")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=64, help="games in flight")
    p.add_argument("--n-steps", type=int, default=128, help="learner decisions per env per update")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--n-epochs", type=int, default=2, help="PPO epochs per update (2: as strong as 4 when continuing a trained run, at half the update cost; a run from scratch wants 4)")
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
        help="opponent for the remaining games: random, greedy, first, a frozen checkpoint as name=ckpt.pt "
        "(unlike --pool-seed its share is fixed: the anchor slot is never PFSP-reweighted), or a schedule "
        "such as random,greedy (promoted in order once the learner's win rate over --anchor-window anchor "
        "games reaches --anchor-promote)",
    )
    p.add_argument("--anchor-promote", type=float, default=0.75, help="win rate that promotes a scheduled anchor to the next")
    p.add_argument("--anchor-window", type=int, default=100, help="anchor games the promotion win rate is measured over")
    p.add_argument("--snapshot-every", type=int, default=10, help="updates between pool snapshots (0: never)")
    p.add_argument("--pool-window", type=int, default=None, help="sample only the newest N snapshots")
    p.add_argument(
        "--pool-seed", nargs="+", default=None, metavar="NAME=CKPT",
        help="checkpoints copied into the run's pool before training: frozen opponents "
        "the seat assigner samples like any snapshot, PFSP-weighted by the learner's "
        "record. Seeded first, they are the pool's oldest entries -- a --pool-window "
        "can age them out. Idempotent on resume (a name already in the pool is skipped).",
    )
    p.add_argument("--no-events", action="store_true", help="Ops-only curriculum: Engine.new_game(events=False)")
    p.add_argument("--margin", type=float, default=0.0, help="weight of the final VP margin in the terminal reward: (1-m)*outcome + m*clip(vp/20); 0 is the outcome alone")
    p.add_argument("--handicap", type=int, default=0, help="training games open with the US this many VP ahead (a tournament bid for the USSR seat); evaluation stays at 0")
    p.add_argument("--bid", type=int, default=0, help="the tournament bid (11.1.4): this much extra US influence placed after setup, in every training game and in --eval-every's evaluations")
    p.add_argument("--scenarios", default=None, help="a scenario bank (wopr.scenarios): start --scenario-frac of training games from its states; evaluation stays at the printed game")
    p.add_argument("--scenario-frac", type=float, default=0.25, help="fraction of training games started from --scenarios (ignored without it)")
    p.add_argument("--scenario-vs-anchor", action="store_true", help="seat every scenario-started game as the bank entry's mover (the learner) against the --anchor opponent, overriding the seat mix for those games (needs --scenarios and a single fixed --anchor)")
    p.add_argument("--veto-train", action="store_true", help="train under the veto: options of the learner's rows that are provable DEFCON deaths within the play are struck from its mask before the policy samples (the self-kill coup, the granted-coup gift; docs/JOSHUA.md kick7); opponent seats untouched, `vetoes_per_game` counts the strikes")
    p.add_argument("--kickstart", default=None, help="a harvested corpus (wopr.distill): after every PPO update, pull the policy toward the teacher's choices with --kickstart-batches cross-entropy minibatches (kickstarting, docs/JOSHUA.md)")
    p.add_argument("--kickstart-coef", type=float, default=1.0, help="weight on the kickstart cross-entropy")
    p.add_argument("--kickstart-batches", type=int, default=4, help="corpus minibatches per PPO update")
    p.add_argument("--kickstart-batch-size", type=int, default=512)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--precision", choices=list(PRECISIONS), default="bf16", help="bf16 autocast for the network (default; halves the update) or plain fp32")
    p.add_argument("--torch-threads", type=int, default=None)
    p.add_argument("--workers", type=int, default=1, help="collector processes stepping the games (1: in this process)")
    p.add_argument("--worker-threads", type=int, default=2, help="torch threads per collector (pool-net inference)")
    p.add_argument("--eval-every", type=int, default=0, help="evaluate the latest checkpoint against --eval-opponent every N training games, on the collectors (0: never)")
    p.add_argument("--eval-games", type=int, default=200, help="games per evaluation (half on each seat)")
    p.add_argument("--eval-seed", type=int, default=1000, help="deck seed of the first evaluation; each later one adds its tick number")
    p.add_argument("--eval-opponent", choices=["random", "greedy", "first"], default="greedy")
    return p


ANCHORS = ("random", "greedy", "first")


def resolve_anchor(spec: str) -> str:
    """One `--anchor` schedule element to its policy id: a built-in name
    as itself, `name=ckpt.pt` as `ckpt:<path>` (`StandardOpponents` seats
    it as a frozen sampling `NetOpponent`). Loading the checkpoint now
    surfaces a bad path or layout before any training happens."""
    if spec in ANCHORS:
        return spec
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise ValueError(f"--anchor: unknown opponent {spec!r}; choose from {ANCHORS} or give name=checkpoint.pt")
    load_checkpoint(path, device="cpu")
    return CKPT_PREFIX + path


def make_anchor_schedule(args: argparse.Namespace) -> AnchorSchedule:
    anchors = tuple(resolve_anchor(name.strip()) for name in args.anchor.split(","))
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


def seed_pool(pool: CheckpointPool, specs: Sequence[str]) -> None:
    """Copy `name=checkpoint.pt` specs into the pool as ordinary snapshots.

    Loading (not file-copying) validates the checkpoint's layout version up
    front and re-saves it self-describing. A name already in the pool is
    skipped, so a resumed run can pass the same flags again."""
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--pool-seed {spec!r}: expected name=checkpoint.pt")
        if name in pool.stats:
            continue
        net, _ = load_checkpoint(path, device="cpu")
        pool.add(name, net, extra={"seeded_from": str(path)})


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def build_env(args: argparse.Namespace, pool: CheckpointPool, anchor: AnchorSchedule, device: str) -> WoprVecEnv:
    seats = make_seat_assigner(args.self_play, args.vs_pool, anchor, pool)
    opponents = StandardOpponents(str(pool.directory), args.seed, device)
    scenario_path = getattr(args, "scenarios", None)
    scenario_frac = getattr(args, "scenario_frac", 0.0) if scenario_path else 0.0
    scenario_seats = None
    if getattr(args, "scenario_vs_anchor", False):
        if not scenario_path:
            raise ValueError("--scenario-vs-anchor needs --scenarios")
        if len(anchor.anchors) != 1:
            raise ValueError("--scenario-vs-anchor needs a single fixed --anchor, not a schedule")
        scenario_seats = (LEARNER, anchor.current)
    veto_train = bool(getattr(args, "veto_train", False))
    if args.workers > 1:
        spec = ArenaSpec(args.n_envs, args.seed, events=not args.no_events, starting_vp=args.handicap,
                         us_bid=args.bid, margin=args.margin,
                         scenario_path=scenario_path, scenario_frac=scenario_frac, scenario_seats=scenario_seats,
                         veto_train=veto_train)
        backend: Backend = SharedMemoryBackend(spec, seats, opponents, workers=args.workers, worker_threads=args.worker_threads)
    else:
        bank = None
        if scenario_path:
            from wopr.scenarios import ScenarioBank

            bank = ScenarioBank.load(scenario_path)
        backend = InProcessBackend(
            Arena(args.n_envs, seed=args.seed, seat_assigner=seats, events=not args.no_events,
                  starting_vp=args.handicap, us_bid=args.bid,
                  scenario_bank=bank, scenario_frac=scenario_frac, scenario_seats=scenario_seats),
            opponents,
            margin=args.margin,
            veto_train=veto_train,
        )
    return WoprVecEnv(backend)


def build_model(args: argparse.Namespace, env: WoprVecEnv, device: str, joshua_config: Mapping[str, Any] | None = None) -> PPO:
    if joshua_config is None:
        joshua_config = {"hidden": args.hidden, "gnn_layers": args.gnn_layers, "card_dim": args.card_dim}
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
            "joshua_config": dict(joshua_config),
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


def resume_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """The saved values `PPO.load` must replace with this segment's flags."""
    return {
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "target_kl": args.target_kl,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
    }


def init_from_checkpoint(args: argparse.Namespace, env: WoprVecEnv, device: str) -> PPO:
    """A new model with `--init`'s weights: the checkpoint's own network
    config, its weights copied in, everything else (optimizer, pool,
    metrics) fresh. A frozen baseline keeps only `joshua.pt`, so this is
    how a line continues from a version after a failed experiment."""
    net, _ = load_checkpoint(args.init, device=device)
    config = net.config.to_dict()
    model = build_model(args, env, device, joshua_config=config)
    model.policy.net.load_state_dict(net.state_dict())
    for key in ("hidden", "gnn_layers", "card_dim"):
        setattr(args, key, config[key])  # config.json records the size actually built
    return model


def check_layout(previous: Mapping[str, Any], run: str) -> None:
    """A run records the layout version its rows were encoded with; later
    code that encodes another cannot continue it (the checkpoint would
    read features that mean something else)."""
    recorded = previous.get("layout_version")
    if recorded is not None and recorded != F.LAYOUT_VERSION:
        raise SystemExit(f"run {run!r} was trained against layout v{recorded}; this code encodes v{F.LAYOUT_VERSION} -- start a new run")


def wire_buffer(model: PPO, env: WoprVecEnv) -> None:
    buffer = model.rollout_buffer
    if not isinstance(buffer, AlternatingRolloutBuffer):
        raise TypeError("model.rollout_buffer must be an AlternatingRolloutBuffer")
    buffer.next_mover_source = env.current_am_us


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


def run(
    args: argparse.Namespace, *, callbacks: Callable[[WoprCallback], Sequence[BaseCallback]] | None = None
) -> WoprCallback | None:
    """Train (or resume) `args.run` to `args.games` games. `callbacks(tracker)`
    builds extra SB3 callbacks that run after the tracker each step
    (`wopr.bootstrap` adds its stop rule); returns the tracker, with the
    game count and the evaluations -- None when the run was already there."""
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    run_dir = RUNS_DIR / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    model_path = run_dir / "ppo.zip"
    pool = CheckpointPool(run_dir / "pool", window=args.pool_window)
    if args.pool_seed:
        seed_pool(pool, args.pool_seed)

    games_done = 0
    updates_done = 0
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        check_layout(previous, args.run)
        games_done = int(previous.get("games_done", 0))
        updates_done = last_update(run_dir / "metrics.csv")
        if args.games <= games_done:
            print(f"[wopr] run {args.run!r} already at {games_done} games (target {args.games}); nothing to do")
            return None
        print(f"[wopr] resuming {args.run!r} from {games_done} games ({updates_done} updates) to {args.games}")

    anchor = make_anchor_schedule(args)
    env = build_env(args, pool, anchor, device)
    if model_path.exists():
        # A resumed run takes its PPO hyperparameters and its precision from
        # the flags, not from the zip: the flags are this segment's spec and
        # `config.json` records them. (`n_steps` stays: it sizes the buffer.)
        if args.init:
            raise SystemExit(f"--init starts a new run; {args.run!r} already has a model")
        model = PPO.load(model_path, env=env, device=device, custom_objects=resume_overrides(args))
        model.policy.precision = args.precision
    elif args.init:
        model = init_from_checkpoint(args, env, device)
        print(f"[wopr] initialised from {args.init}")
    else:
        model = build_model(args, env, device)
    wire_buffer(model, env)

    config: dict[str, Any] = {
        **vars(args), "device": device, "games_done": games_done,
        "commit": git_commit(), "layout_version": F.LAYOUT_VERSION, "rules_version": RULES_VERSION,
    }
    config_path.write_text(json.dumps(config, indent=2))

    def save_model(games: int) -> None:
        # With every pool snapshot, so a killed run resumes from its last
        # snapshot (a `ppo.zip` only written at exit was lost with the process).
        model.save(model_path)
        config["games_done"] = games
        config_path.write_text(json.dumps(config, indent=2))

    tracker = WoprCallback(
        run_dir=run_dir,
        env=env,
        pool=pool,
        target_games=args.games,
        snapshot_every=args.snapshot_every,
        games_done=games_done,
        updates_done=updates_done,
        anchor_schedule=anchor,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        eval_seed=args.eval_seed,
        eval_opponent=args.eval_opponent,
        eval_bid=args.bid,
        on_snapshot=save_model,
    )
    print(f"[wopr] device={device} n_envs={args.n_envs} n_steps={args.n_steps} params={sum(p.numel() for p in model.policy.parameters())}")
    try:
        extra = [] if callbacks is None else list(callbacks(tracker))
        if getattr(args, "kickstart", None):
            from wopr.callback import KickstartCallback

            extra.insert(0, KickstartCallback(
                args.kickstart, coef=args.kickstart_coef, batches_per_update=args.kickstart_batches,
                batch_size=args.kickstart_batch_size, seed=args.seed,
            ))
        model.learn(total_timesteps=2**62, callback=[tracker, StopAtGames(tracker), *extra], reset_num_timesteps=not model_path.exists())
    finally:
        env.close()  # collector processes, if any
        save_model(tracker.games)
        print(f"[wopr] saved {model_path} after {tracker.games} games")
    return tracker


if __name__ == "__main__":
    main()
