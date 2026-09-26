"""Generate a synthetic feature bundle and run the standard training pipeline."""

import argparse
import json
from pathlib import Path
import torch
from lluna.data import FeatureDataset
from lluna.runner import train, test


def make_data(root, task="classification", sites=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    latent = torch.randn(96, 6)
    target = latent[:, 0] + latent[:, 1]
    labels = (target > 0).long() if task in ("classification", "candidate") else target
    dims = {f"expert_{i}": 12 + i for i in range(9)}
    features = {
        k: latent @ torch.randn(6, d) + 0.1 * torch.randn(96, d)
        for k, d in dims.items()
    }
    metadata = dict(
        task=task,
        num_classes=2,
        modalities=list("TAV"),
        feature_dims=dims,
        dependencies={k: ["TAV"[i // 3]] for i, k in enumerate(dims)},
    )
    candidate_mask = None
    if task == "candidate":
        candidate_mask = torch.ones(96, 4, dtype=torch.bool)
        candidate_mask[::2, 3] = False
        features = {
            k: v[:, None, :].expand(-1, 4, -1).clone() + torch.randn(96, 4, v.shape[-1])
            for k, v in features.items()
        }
    if sites:
        if task == "candidate":
            raise ValueError("Candidate and site examples are separate tasks")
        metadata["modalities"] = ["AV", "PV", "TV", "MV"]
        metadata["site_experts"] = list(dims)[:6]
        metadata["dependencies"] = {k: [] for k in dims}
        for k in metadata["site_experts"]:
            features[k] = features[k][:, None, :].expand(-1, 4, -1).clone()
    metadata["require_group_disjoint"] = True
    torch.save(metadata, root / "metadata.pt")
    for name, start, stop in [
        ("train", 0, 64),
        ("validation", 64, 80),
        ("test", 80, 96),
    ]:
        rows = dict(
            ids=[f"example_{i}" for i in range(start, stop)],
            group_ids=[f"group_{i}" for i in range(start, stop)],
            labels=labels[start:stop],
            features={k: v[start:stop] for k, v in features.items()},
        )
        if candidate_mask is not None:
            rows["candidate_mask"] = candidate_mask[start:stop]
        torch.save(rows, root / f"{name}.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("demo"))
    parser.add_argument(
        "--task",
        choices=("classification", "regression", "candidate"),
        default="classification",
    )
    parser.add_argument("--sites", action="store_true")
    parser.add_argument("--sampling", choices=("full", "chain"), default="full")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    make_data(args.output / "data", args.task, args.sites)
    config = json.loads((Path(__file__).parent / "configs/default.json").read_text())
    config["model"].update(hidden_dim=16, common_dim=32)
    config["training"].update(epochs=5, patience=5, view_sampling=args.sampling)
    if args.task == "regression":
        config["training"]["selection_metric"] = "mae"
    data = FeatureDataset(args.output / "data")
    train(data, config, args.output / "run", torch.device("cpu"))
    test(
        data,
        args.output / "run/checkpoint.pt",
        args.output / "evaluation",
        torch.device("cpu"),
    )


if __name__ == "__main__":
    main()
