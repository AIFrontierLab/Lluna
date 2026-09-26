"""Fixed-graph expert collaboration with common/private residual prediction."""

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    feature_dim: int


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ExpertAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.network = MLP(input_dim, hidden_dim, hidden_dim, dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.output_norm(self.network(x))


class SiteAdapter(nn.Module):
    """Shared content projection, site embeddings and ordered masked fusion."""

    def __init__(self, input_dim, hidden_dim, site_count, projection_dim, dropout):
        super().__init__()
        self.site_projection = nn.Linear(input_dim, projection_dim)
        self.site_norm = nn.LayerNorm(projection_dim)
        self.site_embedding = nn.Parameter(
            torch.randn(site_count, projection_dim) * 0.02
        )
        width = site_count * projection_dim + site_count
        self.ordered_fusion = MLP(width, max(hidden_dim, width), hidden_dim, dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):
        x = x * mask.unsqueeze(-1).to(x.dtype)
        tokens = self.site_norm(
            self.site_projection(x)
        ) + self.site_embedding.unsqueeze(0)
        tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
        return self.output_norm(
            self.ordered_fusion(
                torch.cat([tokens.flatten(1), mask.to(tokens.dtype)], -1)
            )
        )


class Lluna(nn.Module):
    """Adjacency indexes source, target. Missing endpoints are masked, not relearned.

    Private = concatenated adapted nodes plus availability.
    Common = shared alignment, gated graph propagation, attention pooling.
    Prediction = private prediction + tanh(scale) * common prediction.
    """

    def __init__(
        self,
        experts,
        adjacency,
        *,
        output_dim,
        hidden_dim,
        common_dim=None,
        dropout=0.1,
        adapter_dropout=None,
        private_head_dropout=None,
        common_dropout=None,
        gnn_layers=1,
        residual_scale_init=0.1,
        site_experts=(),
        site_count=0,
        site_projection_dim=64
    ):
        super().__init__()
        self.experts = list(experts)
        experts = self.experts
        self.expert_names = tuple(e.name for e in experts)
        n, h, c = (
            len(experts),
            hidden_dim,
            2 * hidden_dim if common_dim is None else common_dim,
        )
        dimensions = (h, c, output_dim, gnn_layers, *(e.feature_dim for e in experts))
        if (
            not n
            or len(set(self.expert_names)) != n
            or any(
                not isinstance(d, int) or isinstance(d, bool) or d <= 0
                for d in dimensions
            )
            or any(not e.name or "." in e.name for e in experts)
        ):
            raise ValueError("Nonempty unique experts and positive dimensions required")
        self.site_experts = frozenset(site_experts)
        self.site_count = site_count
        if not self.site_experts <= set(self.expert_names):
            raise ValueError("Unknown site expert")
        if self.site_experts and (site_count < 2 or site_projection_dim < 1):
            raise ValueError("Invalid site adapter dimensions")
        if not self.site_experts and site_count:
            raise ValueError("Site count requires site experts")
        ds = [
            dropout if x is None else x
            for x in (adapter_dropout, private_head_dropout, common_dropout)
        ]
        if any(not 0 <= x < 1 for x in ds) or not -1 < residual_scale_init < 1:
            raise ValueError("Invalid dropout or residual scale")
        graph = torch.as_tensor(adjacency, dtype=torch.float32)
        if (
            graph.shape != (n, n)
            or not torch.isfinite(graph).all()
            or (graph < 0).any()
            or (graph.sum(0) <= 0).any()
        ):
            raise ValueError("Invalid fixed adjacency")
        self.register_buffer("fixed_adjacency", graph.detach().clone())
        self.gnn_layers = gnn_layers
        self.adapters = nn.ModuleDict(
            {
                e.name: (
                    SiteAdapter(
                        e.feature_dim, h, site_count, site_projection_dim, ds[0]
                    )
                    if e.name in self.site_experts
                    else ExpertAdapter(e.feature_dim, h, ds[0])
                )
                for e in experts
            }
        )
        self.private_head = MLP(
            n * h + n + site_count, max(2 * h, output_dim), output_dim, ds[1]
        )
        self.common_aligner = MLP(h, c, c, ds[2])
        self.common_message = MLP(c, c, c, ds[2])
        self.common_gate = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, 1), nn.Sigmoid())
        self.common_norm = nn.LayerNorm(c)
        self.common_score = nn.Linear(c, 1)
        self.common_head = MLP(
            c + n + site_count, max(c, h, output_dim), output_dim, ds[2]
        )
        self.residual_scale_parameter = nn.Parameter(
            torch.tensor(math.atanh(residual_scale_init), dtype=torch.float32)
        )

    def forward(self, batch):
        features, available = batch["features"], batch["available"]
        if set(features) != set(self.expert_names) or set(available) != set(
            self.expert_names
        ):
            raise ValueError("Batch expert keys differ")
        availability = torch.stack([available[e.name] for e in self.experts], 1)
        if availability.dtype != torch.bool or availability.ndim != 2:
            raise ValueError("Availability must be bool [batch] per expert")
        if not availability.any(-1).all():
            raise ValueError("Every sample must retain at least one expert")
        site_mask = batch.get("site_mask")
        if self.site_experts and (
            site_mask is None
            or site_mask.dtype != torch.bool
            or site_mask.shape != (len(availability), self.site_count)
        ):
            raise ValueError("Expected bool [batch, sites] mask")
        nodes = []
        for i, e in enumerate(self.experts):
            x = features[e.name]
            shape = (
                (len(availability), self.site_count, e.feature_dim)
                if e.name in self.site_experts
                else (len(availability), e.feature_dim)
            )
            if x.shape != shape or not x.is_floating_point():
                raise ValueError("Feature shape/type mismatch")
            mask = availability[:, i, None].to(x.dtype)
            state = (
                self.adapters[e.name](x, site_mask & availability[:, i, None])
                if e.name in self.site_experts
                else self.adapters[e.name](x * mask)
            )
            nodes.append(state * mask)
        adapted = torch.stack(nodes, 1)
        extra = [site_mask.to(adapted.dtype)] if self.site_experts else []
        private = self.private_head(
            torch.cat([adapted.flatten(1), availability.to(adapted.dtype)] + extra, -1)
        )
        common_pre = self.common_aligner(adapted) * availability.unsqueeze(-1).to(
            adapted.dtype
        )
        allowed = availability.unsqueeze(2) & availability.unsqueeze(1)
        edges = self.fixed_adjacency.to(common_pre.dtype).unsqueeze(0) * allowed.to(
            common_pre.dtype
        )
        edges = edges / edges.sum(1, keepdim=True).clamp_min(1e-6)
        common = common_pre
        for _ in range(self.gnn_layers):
            incoming = torch.einsum("bij,bid->bjd", edges, self.common_message(common))
            common = self.common_norm(common + self.common_gate(common) * incoming)
            common = common * availability.unsqueeze(-1).to(common.dtype)
        score = self.common_score(common).squeeze(-1)
        readout = torch.softmax(
            score.masked_fill(~availability, -1e4), -1
        ) * availability.to(score.dtype)
        readout = readout / readout.sum(-1, keepdim=True).clamp_min(1.0)
        consensus = torch.einsum("be,bed->bd", readout, common)
        common_logits = self.common_head(
            torch.cat([consensus, availability.to(consensus.dtype)] + extra, -1)
        )
        scale = torch.tanh(self.residual_scale_parameter)
        return dict(
            logits=private + scale * common_logits,
            private_logits=private,
            common_residual_logits=common_logits,
            common_aux_logits=common_logits,
            common_post_consensus=consensus,
            common_pre_nodes=common_pre,
            common_post_nodes=common,
            adapted_nodes=adapted,
            private_nodes=adapted,
            availability_matrix=availability,
            common_readout=readout,
            edges=edges,
            residual_scale=scale,
        )
