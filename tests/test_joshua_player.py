"""Tests for JoshuaNet and JoshuaPlayer: legal play through the public API,
determinism, and checkpoint round-trips. Skipped without torch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from struggler.bots.joshua import features as F  # noqa: E402
from struggler.bots.joshua.model import JoshuaConfig, JoshuaNet, load_checkpoint, save_checkpoint, to_tensors  # noqa: E402
from struggler.bots.joshua.player import JoshuaPlayer  # noqa: E402
from struggler.bots.naive import RandomPlayer  # noqa: E402
from struggler.engine import Engine, Side  # noqa: E402
from struggler.runner import play_game  # noqa: E402

SMALL = JoshuaConfig(hidden=32, gnn_layers=1, card_dim=8, option_hidden=32)


def _net(seed: int = 0) -> JoshuaNet:
    torch.manual_seed(seed)
    return JoshuaNet(SMALL)


def test_forward_masks_illegal_options_and_returns_one_value_per_row():
    engine = Engine.new_game(seed=5)
    observation = engine.observe(Side.USSR)
    buffers = F.encode_single(observation)
    logits, value = _net()(to_tensors(buffers))
    n_options = len(observation.pending_decision.options)
    assert logits.shape == (1, F.K_MAX) and value.shape == (1,)
    assert torch.isfinite(logits[0, :n_options]).all()
    assert (logits[0, n_options:] == torch.finfo(logits.dtype).min).all()


def test_option_head_on_legal_rows_matches_scoring_every_padded_slot():
    """`score_options` runs the shared head only on the `(row, slot)` pairs
    the mask selects. It must produce the same logits as scoring every
    padded slot and masking afterwards -- the straightforward definition --
    on a real batch whose rows have different numbers of legal options."""
    net = _net()
    buffers = F.allocate(3)
    for i, seed in enumerate((5, 6, 7)):
        engine = Engine.new_game(seed=seed)
        for _ in range(i * 4):  # a few opening placements, so the rows differ
            engine.step(engine.pending_decision.options[0])
        F.encode_into(engine.observe(engine.pending_decision.actor), buffers, i)
    obs = to_tensors(buffers)
    mask = obs["opt_mask"].bool()
    assert len(set(mask.sum(1).tolist())) > 1  # rows genuinely differ in legal-option count

    with torch.no_grad():
        nodes, latent = net.encode(obs)
        h = nodes.shape[-1]
        padded = torch.cat([nodes, nodes.new_zeros(3, 1, h)], dim=1)
        option_nodes = torch.gather(padded, 1, obs["opt_country"].unsqueeze(-1).expand(-1, -1, h))
        dense_in = torch.cat(
            [latent.unsqueeze(1).expand(-1, F.K_MAX, -1), obs["opt_feats"], option_nodes, net.card_embedding(obs["opt_card"])],
            dim=-1,
        )
        dense = net.option(dense_in).squeeze(-1).masked_fill(~mask, torch.finfo(torch.float32).min)
        sparse = net.score_options(obs, nodes, latent)

    assert torch.allclose(sparse[mask], dense[mask], atol=1e-6)
    assert (sparse[~mask] == torch.finfo(torch.float32).min).all()
    assert torch.equal(net(obs)[0], sparse)


def test_joshua_plays_a_full_game_legally_and_deterministically():
    def run() -> Engine:
        engine = Engine.new_game(seed=7)
        play_game(engine, {Side.US: JoshuaPlayer(_net(1)), Side.USSR: RandomPlayer(seed=2)})
        return engine

    first, second = run(), run()
    assert first.is_terminal  # play_game would have raised on any illegal action
    assert first.serialize() == second.serialize()
    assert first.turn >= 1 and first.winner in (Side.US, Side.USSR, None)


def test_sampling_player_uses_its_own_rng_not_the_engines():
    engine = Engine.new_game(seed=7)
    before = engine.serialize()["rng_state"]
    player = JoshuaPlayer(_net(1), deterministic=False, seed=3)
    action = player.choose_action(engine.observe(Side.USSR), ())
    assert action in engine.pending_decision.options
    assert engine.serialize()["rng_state"] == before


def test_checkpoint_round_trip_preserves_outputs(tmp_path):
    net = _net(4)
    path = tmp_path / "joshua.pt"
    save_checkpoint(net, path, extra={"games": 12})
    loaded, extra = load_checkpoint(path)
    assert extra == {"games": 12}
    assert loaded.config == SMALL
    obs = to_tensors(F.encode_single(Engine.new_game(seed=9).observe(Side.USSR)))
    with torch.no_grad():
        torch.testing.assert_close(net(obs)[0], loaded(obs)[0])


def test_checkpoint_from_another_layout_version_is_refused(tmp_path):
    path = tmp_path / "joshua.pt"
    save_checkpoint(_net(), path)
    payload = torch.load(path, weights_only=False)
    payload["layout_version"] = F.LAYOUT_VERSION + 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="layout"):
        load_checkpoint(path)
