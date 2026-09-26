"""Validated feature bundles with explicit, disjoint dataset partitions."""

from pathlib import Path
import torch
from .model import ExpertSpec
from .pcc import modality_views, masked_view


class FeatureDataset:
    def __init__(self, path):
        self.root = Path(path)
        meta = torch.load(
            self.root / "metadata.pt", map_location="cpu", weights_only=True
        )
        self.task = meta["task"]
        if self.task not in ("classification", "regression", "candidate"):
            raise ValueError("Supported tasks: classification, regression, candidate")
        self.site_experts = tuple(meta.get("site_experts", []))
        self.require_groups = bool(meta.get("require_group_disjoint", False))
        self.dependencies = {k: tuple(v) for k, v in meta["dependencies"].items()}
        self.experts = [ExpertSpec(k, int(v)) for k, v in meta["feature_dims"].items()]
        if set(self.dependencies) != {e.name for e in self.experts}:
            raise ValueError("Expert metadata differs")
        self.modalities = tuple(meta["modalities"])
        if any(not set(v) <= set(self.modalities) for v in self.dependencies.values()):
            raise ValueError("Unknown modality dependency")
        self.views = modality_views(self.modalities)
        if self.site_experts and (
            self.task == "candidate"
            or not set(self.site_experts) <= set(self.dependencies)
        ):
            raise ValueError("Invalid site expert declaration")
        if any(self.dependencies[name] for name in self.site_experts):
            raise ValueError(
                "Site experts use per-site masking, not all-modality dependencies"
            )
        for view in self.views:
            if not any(
                e.name in self.site_experts or set(self.dependencies[e.name]) <= view
                for e in self.experts
            ):
                raise ValueError("Every view must retain at least one expert")
        self.output_dim = (
            int(meta["num_classes"]) if self.task == "classification" else 1
        )
        if self.task == "classification" and self.output_dim < 2:
            raise ValueError("Classification needs at least two classes")
        self.splits = {}
        # Test data is deliberately not loaded during training or selection.
        for name in ("train", "validation"):
            self.load_split(name)
        self.label_mean = 0.0
        self.label_std = 1.0
        if self.task == "regression":
            y = self.splits["train"]["labels"].float()
            self.label_mean = float(y.mean())
            self.label_std = max(float(y.std(unbiased=False)), 1e-8)

    def load_split(self, name):
        if name in self.splits:
            return self.splits[name]
        if name not in ("train", "validation", "test"):
            raise ValueError("Unknown partition")
        data = torch.load(
            self.root / f"{name}.pt", map_location="cpu", weights_only=True
        )
        ids, y, features = data["ids"], data["labels"], data["features"]
        if (
            not ids
            or len(set(ids)) != len(ids)
            or not all(isinstance(i, str) for i in ids)
        ):
            raise ValueError("IDs must be unique nonempty strings")
        if any(set(ids) & set(s["ids"]) for s in self.splits.values()):
            raise ValueError("Partitions overlap; use patient/group-disjoint IDs")
        groups = data.get("group_ids")
        if self.require_groups and groups is None:
            raise ValueError("group_ids required by dataset metadata")
        if groups is not None:
            if len(groups) != len(ids) or not all(
                isinstance(g, str) and g for g in groups
            ):
                raise ValueError("Invalid group_ids")
            if any(
                set(groups) & set(s.get("group_ids", [])) for s in self.splits.values()
            ):
                raise ValueError("Groups overlap across partitions")
        if self.splits and any(
            ("group_ids" in s) != (groups is not None) for s in self.splits.values()
        ):
            raise ValueError(
                "group_ids must be supplied consistently across partitions"
            )
        if y.shape != (len(ids),) or not torch.isfinite(y).all():
            raise ValueError("Labels must be finite [N]")
        if self.task == "classification" and (
            y.dtype != torch.long or y.min() < 0 or y.max() >= self.output_dim
        ):
            raise ValueError("Class labels must be int64 in [0, num_classes)")
        mask = data.get("candidate_mask")
        if self.task == "candidate":
            if (
                mask is None
                or mask.ndim != 2
                or mask.shape[0] != len(ids)
                or mask.dtype != torch.bool
                or not (mask.sum(-1) >= 2).all()
            ):
                raise ValueError(
                    "Candidate tasks require bool [N,K] mask with at least two valid choices"
                )
            if (
                y.dtype != torch.long
                or y.min() < 0
                or y.max() >= mask.shape[1]
                or not mask[torch.arange(len(y)), y].all()
            ):
                raise ValueError("Candidate label must identify a valid choice")
        if self.task == "regression" and not y.is_floating_point():
            raise ValueError("Regression targets must be floating point")
        if set(features) != {e.name for e in self.experts}:
            raise ValueError("Feature keys differ")
        for e in self.experts:
            x = features[e.name]
            prefix = (
                (len(ids), mask.shape[1])
                if self.task == "candidate"
                else (
                    (len(ids), len(self.modalities))
                    if e.name in self.site_experts
                    else (len(ids),)
                )
            )
            if (
                x.shape != (*prefix, e.feature_dim)
                or not x.is_floating_point()
                or not torch.isfinite(x).all()
            ):
                raise ValueError(f"Invalid features for {e.name}")
        self.splits[name] = data
        return data

    def batch(self, split, indexes, view, device):
        data = self.splits[split]
        features = {
            k: v[indexes].float().to(device) for k, v in data["features"].items()
        }
        labels = data["labels"][indexes].to(device)
        if self.task == "regression":
            labels = (labels.float() - self.label_mean) / self.label_std
        candidate_mask = None
        if self.task == "candidate":
            candidate_mask = data["candidate_mask"][indexes].to(device)
            features = {k: v[candidate_mask] for k, v in features.items()}
        batch = masked_view(features, self.dependencies, view)
        if candidate_mask is not None:
            batch["candidate_mask"] = candidate_mask
        if self.site_experts:
            mask = torch.tensor(
                [m in view for m in self.modalities], dtype=torch.bool, device=device
            ).expand(len(indexes), -1)
            batch["site_mask"] = mask
            for name in self.site_experts:
                batch["features"][name] = features[name] * mask.unsqueeze(-1).to(
                    features[name].dtype
                )
        return batch, labels

    def model(self, adjacency, config):
        from .model import Lluna

        return Lluna(
            self.experts,
            adjacency,
            output_dim=self.output_dim,
            site_experts=self.site_experts,
            site_count=len(self.modalities) if self.site_experts else 0,
            **config,
        )

    def head_features(self, batch, name):
        x = batch["features"][name]
        return x.flatten(1) if name in self.site_experts else x

    def predictions(self, output, batch):
        """Scatter only valid candidates; invalid choices never enter the encoder."""
        if self.task != "candidate":
            return output
        mask = batch["candidate_mask"]
        result = dict(output)
        for key in ("logits", "common_aux_logits"):
            flat = output[key].flatten()
            result[key] = flat.new_full(
                mask.shape, torch.finfo(flat.dtype).min
            ).masked_scatter(mask, flat)
        if "common_post_consensus" in output:
            flat = output["common_post_consensus"]
            state = flat.new_zeros((*mask.shape, flat.shape[-1]))
            state[mask] = flat
            result["common_post_consensus"] = state
        return result


def view_name(view):
    return "+".join(sorted(view))
