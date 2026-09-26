"""Full-lattice or unbiased uniform maximal-chain view sampling."""

from collections import Counter
from itertools import permutations
import random
import math
from .pcc import cover_edges, edge_weights


class ViewSampler:
    def __init__(
        self,
        views,
        mode="full",
        chains_per_step=1,
        level_weights=None,
        task_weights=None,
    ):
        if (
            mode not in ("full", "chain")
            or not isinstance(chains_per_step, int)
            or chains_per_step < 1
        ):
            raise ValueError("Invalid view sampler")
        self.views = tuple(views)
        self.edges = cover_edges(views)
        self.mode, self.draws = mode, chains_per_step
        self.task = (
            {v: 1 / len(views) for v in views}
            if task_weights is None
            else dict(task_weights)
        )
        if (
            set(self.task) != set(views)
            or any(not math.isfinite(w) or w < 0 for w in self.task.values())
            or sum(self.task.values()) <= 0
        ):
            raise ValueError("Invalid task-view weights")
        total = sum(self.task.values())
        self.task = {v: w / total for v, w in self.task.items()}
        self.pcc = edge_weights(self.edges, level_weights)
        modalities = tuple(sorted(views[0]))
        self.chains = [
            tuple(frozenset(p[:k]) for k in range(len(p), 0, -1))
            for p in permutations(modalities)
        ]
        self.view_probability = {
            v: sum(v in c for c in self.chains) / len(self.chains) for v in views
        }
        self.edge_probability = {
            e: sum(e in tuple(zip(c, c[1:])) for c in self.chains) / len(self.chains)
            for e in self.edges
        }

    def sample(self, seed):
        if self.mode == "full":
            return self.views, self.edges, dict(self.task), dict(self.pcc)
        chosen = random.Random(seed).choices(self.chains, k=self.draws)
        vc = Counter(v for c in chosen for v in c)
        ec = Counter(e for c in chosen for e in zip(c, c[1:]))
        views = tuple(v for v in self.views if v in vc)
        edges = tuple(e for e in self.edges if e in ec)
        # Draw multiplicity / expected multiplicity, not renormalization over sampled views.
        vw = {
            v: self.task[v] * vc[v] / (self.draws * self.view_probability[v])
            for v in views
        }
        ew = {
            e: self.pcc[e] * ec[e] / (self.draws * self.edge_probability[e])
            for e in edges
        }
        return views, edges, vw, ew
