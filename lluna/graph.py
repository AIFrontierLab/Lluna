"""Training-only prediction agreement followed by temperature / union-kNN."""

import torch
import math


@torch.no_grad()
def agreement_graph(
    train_predictions,
    *,
    task="classification",
    floor=0.0,
    temperature=1.0,
    top_k=None,
    self_loop=1.0
):
    """Input [train rows, experts, outputs] in shared output coordinates.

    Classification consumes logits. Regression consumes train-standardized
    predictions. No labels, validation rows, or test rows belong here.
    Expert task heads must themselves be fitted on training data only.
    """
    x = train_predictions.detach().cpu()
    if x.ndim != 3 or min(x.shape) <= 0 or not torch.isfinite(x).all():
        raise ValueError("Expected finite [train, experts, outputs]")
    n = x.shape[1]
    k = n - 1 if top_k is None else top_k
    if (
        n < 2
        or not isinstance(k, int)
        or isinstance(k, bool)
        or not 1 <= k < n
        or not all(math.isfinite(v) for v in (temperature, self_loop, floor))
        or temperature <= 0
        or self_loop <= 0
        or not 0 <= floor <= 1
    ):
        raise ValueError("Invalid graph controls")
    x = x.double() if task == "classification" and x.shape[-1] == 2 else x.float()
    if task == "classification":
        x = x.softmax(-1)
        score = (
            (1 - 0.5 * (x[:, :, None, :] - x[:, None, :, :]).abs().sum(-1))
            .clamp(0, 1)
            .mean(0)
        )
    elif task == "regression":
        score = (-(x[:, :, None, :] - x[:, None, :, :]).abs().mean(-1)).exp().mean(0)
    else:
        raise ValueError("Unknown task")
    score = score.float().clamp(min=floor, max=1)
    score = (score + score.T) / 2
    graph = ((score - 1) / temperature).exp()
    graph.fill_diagonal_(self_loop)
    keep = torch.zeros_like(graph, dtype=torch.bool)
    for target in range(n):
        candidates = [i for i in range(n) if i != target]
        order = torch.argsort(graph[candidates, target], descending=True, stable=True)
        keep[[candidates[int(i)] for i in order[:k]], target] = True
    keep = keep | keep.T
    keep.fill_diagonal_(True)
    return graph * keep
