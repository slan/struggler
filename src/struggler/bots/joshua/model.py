"""JoshuaNet: the policy/value network over the WOPR layout (`features.py`).

Pure torch -- no training-framework imports -- so the in-game `JoshuaPlayer`
only needs torch, and the WOPR training code can wrap the same module.

Shape of the network, input by input:

- `board` [N_COUNTRIES, F_BOARD] runs through a small graph network over the
  fixed country adjacency (a dense, row-normalised matrix with self-loops,
  registered as a buffer): each layer is `relu(W_self h + W_nb (A h))`.
  The board is a graph and adjacency decides reachability, coups'
  neighbourhoods, and realignment bonuses, so the node latents learn it.
- `card_loc` [N_CARDS] becomes, per card, `card_embedding + location_embedding`,
  mean-pooled per location into one vector per location (my hand, discard,
  unseen, ...) -- a compact "what is where" summary.
- `globals` [G] through a linear layer; also the query for an attention pool
  over the node latents ("which countries matter right now").
- `focus` [N_FOCUS]: embeddings of the card the decision is about.
- The state latent is an MLP over all of the above; the value head reads it.
- **Options are scored one at a time by a shared head** over
  `[state latent, option features, node latent of the option's country,
  embedding of the option's card]`, then masked. The same head answers
  every decision kind, because the kind is in `globals` and the option's
  meaning is in its own features.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from struggler.bots.joshua import features as F


@dataclass(frozen=True)
class JoshuaConfig:
    hidden: int = 128
    gnn_layers: int = 2
    card_dim: int = 32
    option_hidden: int = 128

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JoshuaConfig":
        return cls(**data)


def _normalised_adjacency() -> Tensor:
    adjacency = torch.tensor(F.ADJACENCY) + torch.eye(F.N_COUNTRIES)
    return adjacency / adjacency.sum(dim=1, keepdim=True)


class JoshuaNet(nn.Module):
    def __init__(self, config: JoshuaConfig | None = None) -> None:
        super().__init__()
        self.config = config or JoshuaConfig()
        h, d, layers = self.config.hidden, self.config.card_dim, self.config.gnn_layers
        if layers < 1:
            raise ValueError("gnn_layers must be >= 1")

        self.register_buffer("adjacency", _normalised_adjacency())
        self.node_in = nn.Linear(F.F_BOARD, h)
        self.node_self = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.node_neighbours = nn.ModuleList(nn.Linear(h, h, bias=False) for _ in range(layers))

        # Index N_CARDS is the "no card" sentinel used by `focus` and `opt_card`.
        self.card_embedding = nn.Embedding(F.N_CARDS + 1, d, padding_idx=F.N_CARDS)
        self.location_embedding = nn.Embedding(F.N_CARD_LOCATIONS, d)

        self.globals_in = nn.Linear(F.G, h)
        self.pool_query = nn.Linear(h, h)
        self.pool_key = nn.Linear(h, h)

        state_in = h + h + F.N_CARD_LOCATIONS * d + F.N_FOCUS * d
        self.state = nn.Sequential(nn.Linear(state_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())

        option_in = h + F.F_OPTION + h + d
        self.option = nn.Sequential(
            nn.Linear(option_in, self.config.option_hidden),
            nn.ReLU(),
            nn.Linear(self.config.option_hidden, 1),
        )
        self.value = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

        # A near-uniform initial policy: early rollouts should explore every
        # legal option, not commit to whatever the random init happened to favour.
        nn.init.orthogonal_(self.option[-1].weight, gain=0.01)
        nn.init.zeros_(self.option[-1].bias)
        nn.init.orthogonal_(self.value[-1].weight, gain=1.0)
        nn.init.zeros_(self.value[-1].bias)

    def forward(self, obs: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Returns `(masked option logits [B, K_MAX], value [B])`."""
        board = obs["board"]
        batch = board.shape[0]
        h = self.config.hidden

        nodes = torch.relu(self.node_in(board))
        for self_layer, neighbour_layer in zip(self.node_self, self.node_neighbours):
            aggregated = torch.einsum("nm,bmh->bnh", self.adjacency, nodes)
            nodes = torch.relu(self_layer(nodes) + neighbour_layer(aggregated))

        globals_latent = torch.relu(self.globals_in(obs["globals"]))
        query = self.pool_query(globals_latent)
        keys = self.pool_key(nodes)
        scores = (keys * query.unsqueeze(1)).sum(-1) / math.sqrt(h)
        weights = torch.softmax(scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * nodes).sum(1)

        card_ids = torch.arange(F.N_CARDS, device=board.device)
        per_card = self.card_embedding(card_ids).unsqueeze(0) + self.location_embedding(obs["card_loc"])
        location_one_hot = torch.nn.functional.one_hot(obs["card_loc"], F.N_CARD_LOCATIONS).to(per_card.dtype)
        location_sums = torch.einsum("bkl,bkd->bld", location_one_hot, per_card)
        location_counts = location_one_hot.sum(1).unsqueeze(-1)
        cards = (location_sums / (location_counts + 1.0)).reshape(batch, -1)

        focus = self.card_embedding(obs["focus"]).reshape(batch, -1)

        latent = self.state(torch.cat([pooled, globals_latent, cards, focus], dim=-1))

        option_feats = obs["opt_feats"]
        k = option_feats.shape[1]
        padded_nodes = torch.cat([nodes, nodes.new_zeros(batch, 1, h)], dim=1)
        option_nodes = torch.gather(padded_nodes, 1, obs["opt_country"].unsqueeze(-1).expand(-1, -1, h))
        option_cards = self.card_embedding(obs["opt_card"])
        option_in = torch.cat(
            [latent.unsqueeze(1).expand(-1, k, -1), option_feats, option_nodes, option_cards], dim=-1
        )
        logits = self.option(option_in).squeeze(-1)
        mask = obs["opt_mask"].to(torch.bool)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits, self.value(latent).squeeze(-1)


def to_tensors(buffers: Mapping[str, np.ndarray], device: torch.device | str = "cpu") -> dict[str, Tensor]:
    return {name: torch.as_tensor(array, device=device) for name, array in buffers.items()}


CHECKPOINT_FORMAT = 1


def save_checkpoint(net: JoshuaNet, path: str | Path, *, extra: Mapping[str, Any] | None = None) -> None:
    """Write a self-describing checkpoint: config, weights, and the layout
    version the weights were trained against (refused on load if it differs)."""
    payload = {
        "format": CHECKPOINT_FORMAT,
        "layout_version": F.LAYOUT_VERSION,
        "config": net.config.to_dict(),
        "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "extra": dict(extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, *, device: torch.device | str = "cpu") -> tuple[JoshuaNet, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: unsupported checkpoint format {payload.get('format')!r}")
    if payload["layout_version"] != F.LAYOUT_VERSION:
        raise ValueError(
            f"{path}: trained against layout v{payload['layout_version']}, "
            f"this code encodes v{F.LAYOUT_VERSION}"
        )
    net = JoshuaNet(JoshuaConfig.from_dict(payload["config"]))
    net.load_state_dict(payload["state_dict"])
    net.to(device)
    net.eval()
    return net, payload["extra"]
