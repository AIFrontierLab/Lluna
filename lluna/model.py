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
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)


class ExpertAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.network = MLP(input_dim, hidden_dim, hidden_dim, dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.output_norm(self.network(x))


class Lluna(nn.Module):
    """Adjacency indexes source, target. Missing endpoints are masked, not relearned.

    Private = concatenated adapted nodes plus availability.
    Common = shared alignment, gated graph propagation, attention pooling.
    Prediction = private prediction + tanh(scale) * common prediction.
    """
    def __init__(self, experts, adjacency, *, output_dim, hidden_dim,
                 common_dim=None, dropout=0.1, adapter_dropout=None,
                 private_head_dropout=None, common_dropout=None,
                 gnn_layers=1, residual_scale_init=0.1):
        super().__init__()
        self.experts = list(experts)
        self.expert_names = tuple(e.name for e in experts)
        n, h, c = len(experts), hidden_dim, common_dim or 2 * hidden_dim
        if not n or len(set(self.expert_names)) != n or min(h,c,output_dim,gnn_layers) <= 0:
            raise ValueError('Nonempty unique experts and positive dimensions required')
        ds = [dropout if x is None else x for x in (adapter_dropout,private_head_dropout,common_dropout)]
        if any(not 0 <= x < 1 for x in ds) or not -1 < residual_scale_init < 1:
            raise ValueError('Invalid dropout or residual scale')
        graph = torch.as_tensor(adjacency,dtype=torch.float32)
        if graph.shape != (n,n) or not torch.isfinite(graph).all() or (graph<0).any() or (graph.sum(0)<=0).any():
            raise ValueError('Invalid fixed adjacency')
        self.register_buffer('fixed_adjacency',graph.detach().clone())
        self.gnn_layers = gnn_layers
        self.adapters = nn.ModuleDict({e.name:ExpertAdapter(e.feature_dim,h,ds[0]) for e in experts})
        self.private_head = MLP(n*h+n,max(2*h,output_dim),output_dim,ds[1])
        self.common_aligner = MLP(h,c,c,ds[2])
        self.common_message = MLP(c,c,c,ds[2])
        self.common_gate = nn.Sequential(nn.LayerNorm(c),nn.Linear(c,1),nn.Sigmoid())
        self.common_norm = nn.LayerNorm(c)
        self.common_score = nn.Linear(c,1)
        self.common_head = MLP(c+n,max(c,h,output_dim),output_dim,ds[2])
        self.residual_scale_parameter = nn.Parameter(torch.tensor(math.atanh(residual_scale_init),dtype=torch.float32))

    def forward(self, batch):
        features, available = batch['features'], batch['available']
        if set(features) != set(self.expert_names) or set(available) != set(self.expert_names):
            raise ValueError('Batch expert keys differ')
        availability = torch.stack([available[e.name] for e in self.experts],1)
        if availability.dtype != torch.bool or availability.ndim != 2:
            raise ValueError('Availability must be bool [batch] per expert')
        nodes=[]
        for i,e in enumerate(self.experts):
            x=features[e.name]
            if x.shape != (len(availability),e.feature_dim) or not x.is_floating_point():
                raise ValueError('Feature shape/type mismatch')
            mask=availability[:,i,None].to(x.dtype)
            nodes.append(self.adapters[e.name](x*mask)*mask)
        adapted=torch.stack(nodes,1)
        private=self.private_head(torch.cat([adapted.flatten(1),availability.to(adapted.dtype)],-1))
        common_pre=self.common_aligner(adapted)*availability.unsqueeze(-1).to(adapted.dtype)
        allowed=availability.unsqueeze(2)&availability.unsqueeze(1)
        edges=self.fixed_adjacency.to(common_pre.dtype).unsqueeze(0)*allowed.to(common_pre.dtype)
        edges=edges/edges.sum(1,keepdim=True).clamp_min(1e-6)
        common=common_pre
        for _ in range(self.gnn_layers):
            incoming=torch.einsum('bij,bid->bjd',edges,self.common_message(common))
            common=self.common_norm(common+self.common_gate(common)*incoming)
            common=common*availability.unsqueeze(-1).to(common.dtype)
        score=self.common_score(common).squeeze(-1)
        readout=torch.softmax(score.masked_fill(~availability,-1e4),-1)*availability.to(score.dtype)
        readout=readout/readout.sum(-1,keepdim=True).clamp_min(1.0)
        consensus=torch.einsum('be,bed->bd',readout,common)
        common_logits=self.common_head(torch.cat([consensus,availability.to(consensus.dtype)],-1))
        scale=torch.tanh(self.residual_scale_parameter)
        return dict(logits=private+scale*common_logits,private_logits=private,
                    common_residual_logits=common_logits,common_aux_logits=common_logits,
                    common_post_consensus=consensus,common_pre_nodes=common_pre,
                    common_post_nodes=common,adapted_nodes=adapted,private_nodes=adapted,
                    availability_matrix=availability,common_readout=readout,edges=edges,residual_scale=scale)
