"""Tests for the WOPR arena: decision routing, the VecEnv's reward/perspective
semantics, the alternating-perspective GAE, the pool, and the ladder.
Skipped without stable-baselines3."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import csv

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from gymnasium import spaces  # noqa: E402

from struggler.bots.joshua import features as F  # noqa: E402
from struggler.bots.joshua.model import JoshuaConfig, JoshuaNet  # noqa: E402
from struggler.bots.naive import FirstLegalPlayer  # noqa: E402
from struggler.engine import Side  # noqa: E402
from wopr.arena import Arena, play_out, self_play  # noqa: E402
from wopr.backend import ArenaSpec, InProcessBackend, SharedMemoryBackend  # noqa: E402
from wopr.buffer import AlternatingRolloutBuffer  # noqa: E402
from wopr.ladder import Match, elo_ratings, summarize  # noqa: E402
from wopr.opponents import NetOpponent, PlayerOpponent, RandomOpponent, StandardOpponents  # noqa: E402
from wopr.pool import POOL_PREFIX, AnchorSchedule, CheckpointPool  # noqa: E402
from wopr.callback import CSV_COLUMNS, ensure_columns  # noqa: E402
from wopr.eval import PairJob, play_pairs, run_pair  # noqa: E402
from wopr.policy import JoshuaPolicy  # noqa: E402
from wopr.vec_env import LEARNER, WoprVecEnv, observation_space  # noqa: E402

SMALL = JoshuaConfig(hidden=32, gnn_layers=1, card_dim=8, option_hidden=32)


def _seats(us: str, ussr: str):
    return lambda slot, episode, rng: {Side.US: us, Side.USSR: ussr}


# -- arena -------------------------------------------------------------------------


def test_arena_groups_pending_rows_by_policy_and_never_exposes_chance():
    arena = Arena(3, seed=1, seat_assigner=_seats("a", "b"))
    pending = arena.pending()
    assert set(pending) == {"a", "b"} or set(pending) == {"b"}  # setup: USSR places first
    for rows in pending.values():
        for row in rows:
            assert row.side is not Side.CHANCE
            assert row.observation.side is row.side
            assert row.observation.pending_decision.actor is row.side


def test_play_out_runs_every_game_to_a_result_with_distinct_seeds():
    arena = Arena(4, seed=2, seat_assigner=_seats("a", "b"))
    results = play_out(arena, {"a": RandomOpponent(1), "b": PlayerOpponent(FirstLegalPlayer())})
    assert [r.slot for r in results] == [0, 1, 2, 3]
    assert len({r.seed for r in results}) == 4
    assert all(arena.is_terminal(i) for i in range(4))
    assert all(r.decisions[Side.US] > 0 and r.decisions[Side.USSR] > 0 for r in results)


def test_arena_rejects_out_of_range_options():
    arena = Arena(1, seed=3)
    with pytest.raises(IndexError):
        arena.apply(0, F.K_MAX)


# -- vec env -------------------------------------------------------------------------


def _first_legal(obs):
    return np.array([np.flatnonzero(mask)[0] for mask in obs["opt_mask"]])


def test_vec_env_self_play_rows_carry_the_mover_and_reward_only_at_game_end():
    env = WoprVecEnv(Arena(4, seed=4, seat_assigner=self_play), lambda pid: RandomOpponent(0))
    obs = env.reset()
    assert obs["globals"].shape == (4, F.G)
    movers_seen = set()
    finished = 0
    for _ in range(400):
        movers_seen.update(obs["globals"][:, F.AM_US_INDEX].tolist())
        obs, rewards, dones, infos = env.step(_first_legal(obs))
        assert np.all((rewards == 0) | dones), "reward only on the terminal row"
        for slot, info in enumerate(infos):
            if dones[slot]:
                finished += 1
                episode = info["episode"]
                assert episode["seats"] == {"US": LEARNER, "USSR": LEARNER}
                expected = 0.0 if episode["winner"] is None else (1.0 if episode["winner"] == episode["mover"] else -1.0)
                assert rewards[slot] == expected
                assert "terminal_observation" not in info  # games always end for real: nothing to bootstrap
    assert movers_seen == {0.0, 1.0}, "both seats must reach the learner in self-play"
    assert finished > 0, "first-legal self-play games end quickly (DEFCON) -- none finished?"
    assert env.current_am_us().shape == (4,)


def test_vec_env_vs_opponent_only_ever_shows_the_learner_seat():
    env = WoprVecEnv(Arena(3, seed=5, seat_assigner=_seats(LEARNER, "random")), lambda pid: RandomOpponent(1))
    obs = env.reset()
    for _ in range(60):
        assert np.all(obs["globals"][:, F.AM_US_INDEX] == 1.0)
        obs, rewards, dones, infos = env.step(_first_legal(obs))
        for slot in np.flatnonzero(dones):
            assert infos[slot]["episode"]["mover"] == "US"
            assert infos[slot]["episode"]["seats"]["USSR"] == "random"


# -- buffer ----------------------------------------------------------------------------


def _buffer(n_steps: int, gamma: float, lam: float) -> AlternatingRolloutBuffer:
    buffer = AlternatingRolloutBuffer(
        n_steps, observation_space(), spaces.Discrete(F.K_MAX), device="cpu", gamma=gamma, gae_lambda=lam, n_envs=1
    )
    buffer.next_mover_source = lambda: np.array([1.0], dtype=np.float32)
    return buffer


def _fill(buffer: AlternatingRolloutBuffer, movers_us: list[float], values: list[float], rewards: list[float]) -> None:
    template = F.allocate(1)
    for t, (mover, value, reward) in enumerate(zip(movers_us, values, rewards)):
        obs = {name: array.copy() for name, array in template.items()}
        obs["globals"][0, F.AM_US_INDEX] = mover
        buffer.add(obs, np.zeros(1), np.array([reward], dtype=np.float32), np.array([t == 0]),
                   torch.tensor([value]), torch.tensor([0.0]))


def test_alternating_gae_flips_sign_when_the_mover_changes():
    # US, US, USSR, US; US wins on the last row. Undiscounted Monte Carlo:
    # every US row is worth +1, the USSR row -1.
    buffer = _buffer(4, gamma=1.0, lam=1.0)
    _fill(buffer, movers_us=[1, 1, 0, 1], values=[0, 0, 0, 0], rewards=[0, 0, 0, 1])
    buffer.compute_returns_and_advantage(torch.zeros(1), np.array([True]))
    np.testing.assert_allclose(buffer.advantages[:, 0], [1.0, 1.0, -1.0, 1.0])
    np.testing.assert_allclose(buffer.returns[:, 0], [1.0, 1.0, -1.0, 1.0])


def test_alternating_gae_with_discount_and_bootstraps_matches_hand_computation():
    buffer = _buffer(4, gamma=0.9, lam=1.0)
    _fill(buffer, movers_us=[1, 1, 0, 1], values=[0.2, 0.0, 0.5, 0.0], rewards=[0, 0, 0, 1])
    buffer.compute_returns_and_advantage(torch.zeros(1), np.array([True]))
    # step 3: A = 1 - 0 = 1
    # step 2 (USSR -> US, flip): delta = 0 + 0.9 * (-1) * 0.0 - 0.5 = -0.5; A = -0.5 + 0.9 * (-1) * 1 = -1.4
    # step 1 (US -> USSR, flip): delta = 0 + 0.9 * (-1) * 0.5 - 0.0 = -0.45; A = -0.45 + 0.9 * (-1) * (-1.4) = 0.81
    # step 0 (US -> US): delta = 0 + 0.9 * 0.0 - 0.2 = -0.2; A = -0.2 + 0.9 * 0.81 = 0.529
    np.testing.assert_allclose(buffer.advantages[:, 0], [0.529, 0.81, -1.4, 1.0], rtol=1e-5)


def test_alternating_gae_reduces_to_plain_gae_with_a_fixed_seat():
    buffer = _buffer(3, gamma=0.9, lam=0.95)
    _fill(buffer, movers_us=[1, 1, 1], values=[0.1, 0.2, 0.3], rewards=[0, 0, 0])
    buffer.compute_returns_and_advantage(torch.tensor([0.4]), np.array([False]))  # bootstrap, next mover US
    d2 = 0 + 0.9 * 0.4 - 0.3
    d1 = 0 + 0.9 * 0.3 - 0.2
    d0 = 0 + 0.9 * 0.2 - 0.1
    a2 = d2
    a1 = d1 + 0.9 * 0.95 * a2
    a0 = d0 + 0.9 * 0.95 * a1
    np.testing.assert_allclose(buffer.advantages[:, 0], [a0, a1, a2], rtol=1e-5)


# -- pool & ladder ----------------------------------------------------------------------


def test_pool_prioritises_snapshots_the_learner_loses_to(tmp_path):
    pool = CheckpointPool(tmp_path, hardness=2.0, floor=0.0)
    torch.manual_seed(0)
    weak = pool.add("weak", JoshuaNet(SMALL))
    strong = pool.add("strong", JoshuaNet(SMALL))
    for _ in range(10):
        pool.record(weak, learner_won=True)
        pool.record(strong, learner_won=False)
    draws = [pool.sample(random.Random(i)) for i in range(50)]
    assert set(draws) == {strong}
    assert pool.learner_win_rate("weak") == 1.0 and pool.learner_win_rate("strong") == 0.0
    pool.save()  # `record` is in-memory; the training callback saves once per game batch
    reloaded = CheckpointPool(tmp_path)
    assert reloaded.names == ["weak", "strong"] and reloaded.stats == pool.stats
    assert (tmp_path / "strong.pt").exists() and strong == POOL_PREFIX + "strong"


def test_pool_save_retries_a_transient_rename_failure(tmp_path, monkeypatch):
    # Windows: a rename over a file another process has open for a moment
    # raises PermissionError; the stats are saved after every pool game.
    pool = CheckpointPool(tmp_path)
    pool.names = ["only"]
    pool.stats = {"only": {"games": 1, "wins": 1}}
    real_replace, failures = Path.replace, []

    def flaky_replace(self, target):
        if not failures:
            failures.append(target)
            raise PermissionError(5, "Access is denied")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr("wopr.pool.time.sleep", lambda _: None)
    pool.save()
    assert failures and CheckpointPool(tmp_path).stats == pool.stats

    def stuck_replace(self, target):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(Path, "replace", stuck_replace)
    with pytest.raises(PermissionError):
        pool.save()


def test_elo_orders_policies_and_respects_anchors():
    matches = [Match("a", "b", 1.0)] * 8 + [Match("b", "a", 0.0)] * 8 + [Match("b", "random", 1.0)] * 8
    ratings = elo_ratings(matches, anchors={"random": 0.0})
    assert ratings["random"] == 0.0
    assert ratings["a"] > ratings["b"] > ratings["random"]
    assert summarize(matches, "a", "b") == {"games": 16, "wins": 16, "draws": 0, "losses": 0, "win_rate": 1.0}


def test_net_opponent_answers_every_row_with_a_legal_index():
    arena = Arena(2, seed=6, seat_assigner=_seats("net", "net"))
    torch.manual_seed(0)
    opponent = NetOpponent(JoshuaNet(SMALL), seed=1)
    rows = arena.pending()["net"]
    choices = opponent.choose(rows)
    assert len(choices) == len(rows)
    for row, choice in zip(rows, choices):
        assert 0 <= choice < len(row.observation.pending_decision.options)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_policy_precision_keeps_the_ppo_interface_in_float32(precision):
    """Under `bf16` only the network's matmuls change dtype: actions are
    legal, log-probs and values come back float32 (the loss stays float32),
    and the setting survives `_get_constructor_parameters` so `PPO.load`
    rebuilds the same policy."""
    torch.manual_seed(0)
    policy = JoshuaPolicy(
        observation_space(), spaces.Discrete(F.K_MAX), lambda _: 3e-4, joshua_config=SMALL, precision=precision
    )
    arena = Arena(4, seed=3, seat_assigner=lambda slot, episode, rng: {Side.US: LEARNER, Side.USSR: LEARNER})
    env = WoprVecEnv(arena, lambda policy_id: None)
    obs = {name: torch.as_tensor(array) for name, array in env.reset().items()}

    actions, values, log_prob = policy(obs)
    mask = obs["opt_mask"].bool()
    assert mask[torch.arange(4), actions].all()
    assert values.dtype is torch.float32 and log_prob.dtype is torch.float32
    assert torch.isfinite(log_prob).all()
    evaluated_values, evaluated_log_prob, entropy = policy.evaluate_actions(obs, actions)
    assert torch.allclose(evaluated_log_prob, log_prob) and entropy.dtype is torch.float32
    assert policy._get_constructor_parameters()["precision"] == precision

    with pytest.raises(ValueError):
        JoshuaPolicy(observation_space(), spaces.Discrete(F.K_MAX), lambda _: 3e-4, joshua_config=SMALL, precision="fp16")


def test_metrics_csv_from_an_older_run_is_rewritten_under_the_current_columns(tmp_path):
    """Rows are appended in `CSV_COLUMNS` order, so a resumed run whose
    file predates a column must have its header (and old rows) migrated,
    not its new values filed under the wrong names."""
    path = tmp_path / "metrics.csv"
    path.write_text("update,games,steps_per_s,win_rate\n1,30,2000.0,0.5\n2,65,1900.0,0.6\n")

    ensure_columns(path)

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == CSV_COLUMNS
        rows = list(reader)
    assert [r["update"] for r in rows] == ["1", "2"]
    assert rows[1]["win_rate"] == "0.6" and rows[1]["steps_per_s"] == "1900.0"
    assert rows[1]["update_s"] == ""  # a column the old run never had

    ensure_columns(path)  # idempotent on a current file
    assert path.read_text().count("\n") == 3


def test_anchor_schedule_promotes_on_a_windowed_win_rate_and_keeps_the_last():
    schedule = AnchorSchedule(["random", "greedy"], promote_at=0.75, window=4)
    assert schedule.current == "random" and schedule.is_anchor("greedy") and not schedule.is_anchor("pool:u00005")
    # Three wins in three games: above the threshold, but the window is not full yet.
    assert [schedule.record(True) for _ in range(3)] == [None, None, None]
    assert schedule.current == "random"
    # The fourth game fills the window at exactly 3/4 = promote_at: promoted.
    assert schedule.record(False) == "greedy"
    assert schedule.current == "greedy"
    # The last anchor is kept no matter how often the learner wins.
    for _ in range(10):
        assert schedule.record(True) is None
    assert schedule.current == "greedy"

    with pytest.raises(ValueError):
        AnchorSchedule([], promote_at=0.75)


def test_eval_pairs_are_independent_jobs_and_run_in_a_process_pool():
    """A pair's result depends only on its own job -- the same job played
    alone, in-process, or in a pool beside other pairs gives the same
    matches -- which is what lets the ladder fan pairs out to processes."""
    jobs = [PairJob("random", "first", 4, seed=7), PairJob("first", "random", 4, seed=8), PairJob("random", "first", 4, seed=7)]
    alone = run_pair(jobs[0])
    serial = play_pairs(jobs, workers=1)
    pooled = play_pairs(jobs, workers=2)

    assert len(alone) == 4 and all(isinstance(m, Match) for m in alone)
    assert {m.a for m in alone} == {"random", "first"}  # half the games on each seat
    assert serial[0] == alone and serial[2] == alone
    assert pooled == serial


def test_shared_memory_backend_plays_the_same_games_as_the_in_process_one(tmp_path):
    """k collectors over shared memory must be indistinguishable from one
    process: same rows, rewards, dones and episode records step for step.
    Self-play against a fixed first-option policy is fully deterministic
    (no opponent RNG), so the comparison is exact."""
    n_slots, steps = 6, 300
    opponents = StandardOpponents(str(tmp_path), seed=1)
    local = InProcessBackend(Arena(n_slots, seed=9, seat_assigner=self_play), opponents)
    shared = SharedMemoryBackend(ArenaSpec(n_slots, 9), self_play, opponents, workers=3)
    try:
        local.reset()
        shared.reset()
        for name in F.LAYOUT:
            assert np.array_equal(local.buffers[name], shared.buffers[name]), name
        finished = 0
        for _ in range(steps):
            actions = np.zeros(n_slots, dtype=np.int64)  # the first legal option, always
            r1, d1, rec1 = local.step(actions)
            r2, d2, rec2 = shared.step(actions)
            assert np.array_equal(r1, r2) and np.array_equal(d1, d2)
            assert np.array_equal(local.am_us(), shared.am_us())
            assert rec1 == rec2
            finished += int(d1.sum())
            for name in F.LAYOUT:
                assert np.array_equal(local.buffers[name], shared.buffers[name]), name
        assert finished > 0  # first-option self-play games end fast, so resets were exercised too
    finally:
        shared.close()


def test_loop_gate_needs_every_seed_and_versions_follow_the_chain(tmp_path, monkeypatch):
    from wopr import baseline, loop

    report = {"vs_greedy": {"win_rate": 0.7, "as_us": 0.7, "as_ussr": 0.7},
              "vs_champion": {"win_rate": 0.6, "as_us": 0.6, "as_ussr": 0.6, "min_seed": 0.5, "per_seed": []}}
    assert not loop.gate_passes(report, 0.55)  # a mean of 0.6 does not carry a seed at 0.5
    report["vs_champion"]["min_seed"] = 0.56
    assert loop.gate_passes(report, 0.55)
    del report["vs_champion"]
    assert loop.gate_passes(report, 0.55)  # no champion yet: the same bar against Greedy
    assert not loop.gate_passes(report, 0.75)

    monkeypatch.setattr(baseline, "BASELINES_DIR", tmp_path)
    assert loop.latest_version() is None and loop.next_version() == "v1"
    for name in ("v1", "v3", "v10", "v2-draft"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "joshua.pt").write_bytes(b"")
    assert loop.latest_version() == "v10" and loop.next_version() == "v11"  # numeric, not lexical; drafts ignored

    summary = {"commit": "abcdef0123", "run": "pure", "games_trained": 12000, "protocol": {"seeds": [0, 1, 2]},
               "elo": {"v11": {"mean": 1300.4, "std": 12.0}},
               "win_rate": {"greedy": {"win_rate": {"mean": 0.8}, "as_us": {"mean": 0.75}, "as_ussr": {"mean": 0.85}}}}
    entry = loop.readme_entry("v11", summary, "note")
    assert entry.startswith("## v11\n") and "`abcdef0`" in entry and "12,000 games" in entry
    assert "+1300 ± 12" in entry and "vs greedy: 0.800 (US 0.750 / USSR 0.850)" in entry


def test_resumed_run_takes_its_ppo_hyperparameters_from_the_flags(tmp_path):
    """`PPO.load` restores what the zip saved; a resumed segment must run on
    the flags it was given (the config records them), or `-- --n-epochs 2`
    through the loop would silently train with the old value."""
    from stable_baselines3 import PPO
    from wopr import train

    first = train.parse_args(["--run", "x", "--games", "1", "--n-envs", "2", "--n-steps", "4", "--batch-size", "8", "--n-epochs", "4"])
    env = WoprVecEnv(Arena(2, seed=1, seat_assigner=self_play), lambda policy_id: None)
    model = train.build_model(first, env, "cpu")
    assert model.n_epochs == 4
    model.save(tmp_path / "ppo.zip")

    second = train.parse_args(["--run", "x", "--games", "1", "--n-epochs", "2", "--lr", "1e-4", "--clip-range", "0.1", "--ent-coef", "0.05"])
    resumed = PPO.load(tmp_path / "ppo.zip", env=env, device="cpu", custom_objects=train.resume_overrides(second))
    assert resumed.n_epochs == 2 and resumed.ent_coef == 0.05
    assert resumed.lr_schedule(1.0) == 1e-4 and resumed.clip_range(1.0) == 0.1
    assert resumed.n_steps == 4  # sized the buffer; not a flag a resume can change


def test_arena_and_eval_open_games_at_the_handicap():
    arena = Arena(2, seed=0, starting_vp=5)
    assert arena.engine(0).vp == 5 and arena.engine(1).vp == 5
    arena.reset(0)
    assert arena.engine(0).vp == 5
    assert Arena(1, seed=0).engine(0).vp == 0
    job = PairJob("random", "first", games=2, seed=0, starting_vp=19)
    assert ArenaSpec(4, 0, starting_vp=job.starting_vp).starting_vp == 19
    # With the US one VP short of winning, the first VP the US gains ends
    # the game: every match still resolves to a Match with a US-first record.
    matches = run_pair(job)
    assert len(matches) == 2 and {m.a for m in matches} == {"random", "first"}


def test_margin_reward_grades_the_final_vp_and_stays_zero_sum():
    from wopr.backend import EpisodeRecord

    def rec(winner, mover, vp, margin):
        return EpisodeRecord(winner=winner, mover=mover, seats={}, turn=10, vp=vp, seed=0, length=1, margin=margin)

    # Margin 0: the outcome alone.
    assert rec(Side.USSR, Side.US, -20, 0.0).reward() == -1.0
    assert rec(None, Side.US, 5, 0.0).reward() == 0.0
    # Margin on: a loss held to -4 beats a loss at -20; a win on VP is still +1.
    assert rec(Side.USSR, Side.US, -4, 0.5).reward() == pytest.approx(-0.6)
    assert rec(Side.USSR, Side.US, -20, 0.5).reward() == pytest.approx(-1.0)
    assert rec(Side.USSR, Side.USSR, -20, 0.5).reward() == pytest.approx(1.0)
    # The two seats' rewards sum to zero for every final state.
    for vp in (-20, -4, 0, 7):
        us, ussr = rec(Side.USSR, Side.US, vp, 0.5), rec(Side.USSR, Side.USSR, vp, 0.5)
        assert us.reward() + ussr.reward() == pytest.approx(0.0)
    assert ArenaSpec(4, 0, margin=0.5).margin == 0.5
