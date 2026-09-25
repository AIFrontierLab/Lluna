# Feature data

Place four PyTorch files in a dataset directory. Use tensors and ordinary Python
containers only; files are loaded with `weights_only=True`.

`metadata.pt`:

```python
{
    "task": "classification",  # or "regression"
    "num_classes": 2,          # classification only
    "modalities": ["T", "A", "V"],
    "feature_dims": {"text_encoder": 768, "audio_encoder": 1024, "visual_encoder": 768},
    "dependencies": {"text_encoder": ["T"], "audio_encoder": ["A"], "visual_encoder": ["V"]}
}
```

Each of `train.pt`, `validation.pt`, and `test.pt` contains:

```python
{
    "ids": ["sample_001", "sample_002"],
    "features": {
        "text_encoder": torch.randn(2, 768),
        "audio_encoder": torch.randn(2, 1024),
        "visual_encoder": torch.randn(2, 768)
    },
    "labels": torch.tensor([0, 1], dtype=torch.long)
}
```

Use finite floating-point vectors and matching sample order across experts.
Classification labels are contiguous int64 class indexes; regression labels are
floating-point scalars in original units. Keep subject/group partitions disjoint
when examples share a patient, speaker, or source recording. The loader rejects
duplicate IDs and ID overlap; it cannot infer group relationships from opaque IDs.

Frozen encoders must use the same preprocessing across partitions. Any learned
feature normalization or projection must be fitted on training rows only. The
runner fits independent linear expert task heads on training features to construct
the agreement graph. These heads supply graph predictions, not fusion inputs.
Regression targets are standardized using training statistics and restored before
reporting metrics. The model receives features and availability, never labels.

Training enumerates every nonempty modality subset. An expert is masked when any
of its required modalities is absent. An empty dependency list denotes context
that is always available. This interface assumes one vector per expert and sample;
ordered-site inputs and variable-length answer candidates require their own data
adapters and are not handled by flattening them into this format without adjustment.

`configs/default.json` supplies runnable defaults, not dataset-specific benchmark
recipes. For regression set `training.selection_metric` to `mae` or `rmse`.
Classification supports `accuracy`, `balanced_accuracy`, `macro_f1`, and `weighted_f1`.

Training writes only one selected checkpoint, expert heads/graph, configuration,
and epoch history. Evaluation writes per-view predictions, JSON metrics and a CSV
table with an equal-view Mean-Missing row. Undefined correlation is reported as
null. Existing run/evaluation directories are never silently overwritten.
