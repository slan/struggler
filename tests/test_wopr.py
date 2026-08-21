"""Tests for the WOPR arena: decision routing, the VecEnv's reward/perspective
semantics, the alternating-perspective GAE, the pool, and the ladder.
Skipped without stable-baselines3."""

from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from gymnasium import spaces  # noqa: E402

from struggler.bots.joshua import features as F  # noqa: E402
from struggler.bots.joshua.model import JoshuaConfig, JoshuaNet  # noqa: E402
from struggler.bots.naive import FirstLegalPlayer  # noqa: E402
from struggler.engine import Side  # noqa: E402
from wopr.arena import Arena, play_out, self_play  # noqa: E402
from wopr.buffer import AlternatingRolloutBuffer  # noqa: E402
from wopr.ladder import Match, elo_ratings, summarize  # noqa: E402
from wopr.opponents import NetOpponent, PlayerOpponent, RandomOpponent  # noqa: E402
from wopr.pool import POOL_PREFIX, CheckpointPool  # noqa: E402
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
                assert set(info["terminal_observation"]) == set(F.LAYOUT)
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
