"""Train on cached features, select on validation, evaluate a frozen checkpoint."""

import argparse
import csv
import hashlib
import json
import random
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from .data import FeatureDataset, view_name
from .graph import agreement_graph
from .metrics import metrics
from .pcc import pcc_loss
from .sampling import ViewSampler


def batches(n, size, seed=None):
    if size <= 0:
        raise ValueError("Batch size must be positive")
    ids = list(range(n))
    if seed is not None:
        random.Random(seed).shuffle(ids)
    return [ids[i : i + size] for i in range(0, n, size)]


def objective(prediction, labels, task, weights=None):
    return (
        F.mse_loss(prediction.flatten(), labels.float())
        if task == "regression"
        else F.cross_entropy(prediction, labels, weight=weights)
    )


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def evaluate(data, model, split, batch_size, device):
    model.eval()
    labels = data.splits[split]["labels"]
    result, predictions = {}, {}
    for view in data.views:
        parts = []
        for ids in batches(len(labels), batch_size):
            batch, _ = data.batch(split, ids, view, device)
            parts.append(data.predictions(model(batch), batch)["logits"].detach().cpu())
        prediction = torch.cat(parts)
        if data.task == "regression":
            prediction = prediction.flatten() * data.label_std + data.label_mean
        name = view_name(view)
        result[name] = metrics(prediction, labels, data.task)
        predictions[name] = prediction
    return result, predictions


def fit_heads(data, config, device):
    cfg = config["heads"]
    heads = nn.ModuleList(
        [
            nn.Linear(
                e.feature_dim
                * (len(data.modalities) if e.name in data.site_experts else 1),
                data.output_dim,
            )
            for e in data.experts
        ]
    ).to(device)

    def predict(head, e, batch):
        z = head(data.head_features(batch, e.name))
        return data.predictions({"logits": z, "common_aux_logits": z}, batch)["logits"]

    optimizer = torch.optim.AdamW(
        heads.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    for epoch in range(cfg["epochs"]):
        for ids in batches(
            len(data.splits["train"]["ids"]), cfg["batch_size"], config["seed"] + epoch
        ):
            batch, y = data.batch("train", ids, data.views[0], device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack(
                [
                    objective(predict(head, e, batch), y, data.task)
                    for head, e in zip(heads, data.experts)
                ]
            ).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite expert-head loss")
            loss.backward()
            optimizer.step()
    heads.eval()
    with torch.no_grad():
        rows = []
        for ids in batches(len(data.splits["train"]["ids"]), cfg["batch_size"]):
            batch, _ = data.batch("train", ids, data.views[0], device)
            rows.append(
                torch.stack(
                    [predict(head, e, batch) for head, e in zip(heads, data.experts)], 1
                ).cpu()
            )
    return torch.cat(rows), {k: v.cpu() for k, v in heads.state_dict().items()}


def train(data, config, output, device):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {output}")
    output.mkdir(parents=True)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    training = config["training"]
    if (
        training["epochs"] < 1
        or training["patience"] < 1
        or config["heads"]["epochs"] < 1
    ):
        raise ValueError("Epochs and patience must be positive")
    for section, keys in (
        (
            training,
            (
                "learning_rate",
                "weight_decay",
                "gradient_clip",
                "aux_weight",
                "pcc_weight",
                "pcc_deadzone",
                "pcc_power",
            ),
        ),
        (config["heads"], ("learning_rate", "weight_decay")),
    ):
        if any(not math.isfinite(section[k]) or section[k] < 0 for k in keys):
            raise ValueError("Training coefficients must be finite and nonnegative")
    if (
        training["learning_rate"] <= 0
        or config["heads"]["learning_rate"] <= 0
        or training["gradient_clip"] <= 0
        or training["pcc_power"] <= 0
    ):
        raise ValueError("Learning rates, gradient clip and PCC power must be positive")
    metric = training["selection_metric"]
    supported = (
        ("mae", "rmse")
        if data.task == "regression"
        else (
            ("accuracy",)
            if data.task == "candidate"
            else ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
        )
    )
    if metric not in supported:
        raise ValueError(f"Selection metric must be one of {supported}")
    save_json(output / "config.json", config)
    # Fingerprints cover only inputs used by fitting and selection.
    provenance = {
        name: fingerprint(data.root / name)
        for name in ("metadata.pt", "train.pt", "validation.pt")
    }
    predictions, heads = fit_heads(data, config, device)
    graph = agreement_graph(
        predictions,
        task="classification" if data.task == "candidate" else data.task,
        **config["graph"],
    )
    torch.save({"state_dict": heads, "graph": graph}, output / "expert_heads.pt")
    model = data.model(graph, config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    weights = None
    if training.get("class_balance", False) and data.task == "classification":
        counts = torch.bincount(
            data.splits["train"]["labels"], minlength=data.output_dim
        ).float()
        if (counts == 0).any():
            raise ValueError("Class balancing requires every class in training")
        weights = (counts.sum() / (len(counts) * counts)).to(device)
    level_weights = training.get("pcc_level_weights")
    if level_weights is not None:
        level_weights = {int(k): v for k, v in level_weights.items()}
    task_weights = training.get("task_view_weights")
    if task_weights is not None:
        if set(task_weights) != {view_name(v) for v in data.views}:
            raise ValueError("Task-view weights must specify every canonical view name")
        task_weights = {v: task_weights[view_name(v)] for v in data.views}
    sampler = ViewSampler(
        data.views,
        mode=training.get("view_sampling", "full"),
        chains_per_step=training.get("chains_per_step", 1),
        level_weights=level_weights,
        task_weights=task_weights,
    )
    history, best, stale = [], None, 0
    for epoch in range(1, training["epochs"] + 1):
        model.train()
        total, seen = 0.0, 0
        for step, ids in enumerate(
            batches(
                len(data.splits["train"]["ids"]), training["batch_size"], seed + epoch
            )
        ):
            optimizer.zero_grad(set_to_none=True)
            outputs = {}
            views, edges, task_weights, pcc_weights = sampler.sample(
                seed + 100003 * epoch + step
            )
            for view in views:
                batch, y = data.batch("train", ids, view, device)
                devices = [device.index or 0] if device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed + 100003 * epoch + step)
                    outputs[view] = data.predictions(model(batch), batch)
            task = torch.stack(
                [
                    task_weights[v] * objective(o["logits"], y, data.task, weights)
                    for v, o in outputs.items()
                ]
            ).sum()
            auxiliary = torch.stack(
                [
                    task_weights[v]
                    * objective(o["common_aux_logits"], y, data.task, weights)
                    for v, o in outputs.items()
                ]
            ).sum()
            consistency = pcc_loss(
                {v: o["common_post_consensus"] for v, o in outputs.items()},
                edges,
                deadzone=training["pcc_deadzone"],
                power=training["pcc_power"],
                weights=pcc_weights,
                candidate_mask=batch.get("candidate_mask"),
            )
            loss = (
                task
                + training["aux_weight"] * auxiliary
                + training["pcc_weight"] * consistency
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), training["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            seen += len(ids)
        validation, _ = evaluate(
            data, model, "validation", training["eval_batch_size"], device
        )
        score = sum(validation[view_name(v)][metric] for v in data.views[1:]) / (
            len(data.views) - 1
        )
        improved = best is None or (
            score < best if data.task == "regression" else score > best
        )
        if improved:
            best, stale = score, 0
            torch.save(
                {
                    "model": {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                    "config": config,
                    "epoch": epoch,
                    "validation_score": score,
                    "input_sha256": provenance,
                    "label_mean": data.label_mean,
                    "label_std": data.label_std,
                },
                output / "checkpoint.pt",
            )
        else:
            stale += 1
        row = dict(
            epoch=epoch,
            loss=total / seen,
            validation_mean_missing=score,
            selected=improved,
            views=validation,
        )
        history.append(row)
        save_json(output / "history.json", history)
        print(json.dumps({k: v for k, v in row.items() if k != "views"}), flush=True)
        if stale >= training["patience"]:
            break
    save_json(
        output / "training_summary.json",
        {
            "selected_metric": metric,
            "validation_mean_missing": best,
            "epochs": len(history),
        },
    )


def test(data, checkpoint, output, device):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation directory: {output}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for name, expected in saved["input_sha256"].items():
        if fingerprint(data.root / name) != expected:
            raise ValueError(f"Checkpoint/data mismatch: {name}")
    config = saved["config"]
    model = data.model(saved["model"]["fixed_adjacency"], config["model"]).to(device)
    model.load_state_dict(saved["model"], strict=True)
    data.label_mean, data.label_std = saved["label_mean"], saved["label_std"]
    data.load_split("test")
    results, predictions = evaluate(
        data, model, "test", config["training"]["eval_batch_size"], device
    )
    output.mkdir(parents=True)
    torch.save(
        {"ids": data.splits["test"]["ids"], "predictions": predictions},
        output / "predictions.pt",
    )
    keys = list(next(iter(results.values())))
    missing = [results[view_name(v)] for v in data.views[1:]]
    averages = {
        k: (
            sum(r[k] for r in missing) / len(missing)
            if all(r[k] is not None for r in missing)
            else None
        )
        for k in keys
    }
    save_json(
        output / "metrics.json",
        {
            "views": results,
            "mean_missing": averages,
            "checkpoint_sha256": fingerprint(checkpoint),
            "test_sha256": fingerprint(data.root / "test.pt"),
        },
    )
    with (output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["view"] + keys)
        writer.writeheader()
        for name, row in results.items():
            writer.writerow({"view": name, **row})
        writer.writerow({"view": "Mean-Missing", **averages})
    print(json.dumps(averages), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("train", "evaluate"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    data, device = FeatureDataset(args.data), torch.device(args.device)
    if args.command == "train":
        if args.config is None:
            p.error("--config is required for training")
        train(data, json.loads(args.config.read_text()), args.output, device)
    else:
        if args.checkpoint is None:
            p.error("--checkpoint is required for evaluation")
        test(data, args.checkpoint, args.output, device)


if __name__ == "__main__":
    main()
