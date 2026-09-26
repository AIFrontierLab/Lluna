"""One-way adjacent-view consistency of pooled common representations."""

from collections import defaultdict
from itertools import combinations
import math
import torch
import torch.nn.functional as F


def modality_views(modalities):
    """All nonempty modality subsets, richest first."""
    modalities = tuple(modalities)
    if (
        len(set(modalities)) != len(modalities)
        or not 2 <= len(modalities) <= 6
        or any(not isinstance(m, str) or not m or "+" in m for m in modalities)
    ):
        raise ValueError("At least two unique modalities required")
    return tuple(
        frozenset(c)
        for k in range(len(modalities), 0, -1)
        for c in combinations(modalities, k)
    )


def cover_edges(views):
    return tuple((p, c) for p in views for c in views if c < p and len(p) == len(c) + 1)


def edge_weights(edges, level_weights=None):
    """Balance child level, then child, then incoming teacher."""
    groups = defaultdict(lambda: defaultdict(list))
    if not edges or len(set(edges)) != len(edges):
        raise ValueError("Nonempty unique edges required")
    for p, c in edges:
        if not c < p or len(p) != len(c) + 1:
            raise ValueError("PCC requires adjacent rich-to-poor edges")
        groups[len(c)][c].append(p)
    masses = {k: 1.0 for k in groups} if level_weights is None else dict(level_weights)
    if (
        set(masses) != set(groups)
        or any(not math.isfinite(v) or v < 0 for v in masses.values())
        or sum(masses.values()) <= 0
    ):
        raise ValueError("Invalid child-level weights")
    total = sum(masses.values())
    return {
        (p, c): masses[k] / total / len(children) / len(parents)
        for k, children in groups.items()
        for c, parents in children.items()
        for p in parents
    }


def pcc_loss(
    common_states,
    edges,
    *,
    level_weights=None,
    deadzone=0.0,
    power=1.0,
    weights=None,
    candidate_mask=None
):
    if (
        not math.isfinite(deadzone)
        or deadzone < 0
        or not math.isfinite(power)
        or power <= 0
    ):
        raise ValueError("Invalid distance shaping")
    weights = edge_weights(edges, level_weights) if weights is None else weights
    if set(weights) != set(edges) or any(
        not math.isfinite(v) or v < 0 for v in weights.values()
    ):
        raise ValueError("Invalid edge coefficients")
    terms = []
    for parent, child in edges:
        teacher, student = common_states[parent], common_states[child]
        expected_ndim = 2 if candidate_mask is None else 3
        if teacher.ndim != expected_ndim or teacher.shape != student.shape:
            raise ValueError("Expected matching [batch, common_dim] states")
        distance = 1 - F.cosine_similarity(teacher.detach(), student, dim=-1, eps=1e-8)
        if deadzone != 0 or power != 1:
            distance = (distance.clamp_min(0) - deadzone).clamp_min(0).pow(power)
        if candidate_mask is not None:
            if (
                candidate_mask.dtype != torch.bool
                or candidate_mask.shape != distance.shape
                or not candidate_mask.any(-1).all()
            ):
                raise ValueError("Invalid candidate mask")
            distance = distance.clamp_min(0)
            distance = (distance * candidate_mask.to(distance.dtype)).sum(
                -1
            ) / candidate_mask.sum(-1)
        terms.append(weights[parent, child] * distance.mean())
    return torch.stack(terms).sum()


def masked_view(features, dependencies, view):
    """An expert is present only when all its required modalities survive.

    Empty dependency sets represent always-visible context. Features must
    already be finite; external encoders are frozen and run outside this code.
    """
    if set(features) != set(dependencies):
        raise ValueError("Dependency keys differ from features")
    return {
        "features": features,
        "available": {
            name: torch.full(
                (len(x),),
                set(dependencies[name]) <= set(view),
                dtype=torch.bool,
                device=x.device,
            )
            for name, x in features.items()
        },
    }
