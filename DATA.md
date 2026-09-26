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
duplicate IDs and ID overlap. Supply a parallel list of `group_ids` in every split
to also check patient/speaker/source-group overlap. Set
`require_group_disjoint: true` in metadata to require this field. Group identity
must be supplied by the data preparer; it cannot be inferred from sample IDs.

Frozen encoders must use the same preprocessing across partitions. Any learned
feature normalization or projection must be fitted on training rows only. The
runner fits independent linear expert task heads on training features to construct
the agreement graph. These heads supply graph predictions, not fusion inputs.
Regression targets are standardized using training statistics and restored before
reporting metrics. The model receives features and availability, never labels.

Evaluation enumerates every nonempty modality subset. An expert is masked when any
of its required modalities is absent. An empty dependency list denotes context
that is always available. Every view must retain at least one usable expert.
Use 2–6 unique modality names, without `+`, and stable expert order in metadata.

## Candidate-answer tasks

Set `task: "candidate"`. Store each expert's features as `[N,K,D]`, labels as
int64 answer indexes `[N]`, and `candidate_mask` as bool `[N,K]`. Each question must
have at least two valid choices and its label must point to a valid choice. Pad to
a common K within each partition. Padding values must be finite.

Only valid candidate rows are forwarded through the model. A shared scalar head
scores candidates, then predictions are regrouped by question. Invalid choices
are masked out of cross entropy, graph probabilities and accuracy. PCC averages
distances over valid choices first, then over questions, so questions with more
choices receive no extra weight. Graph agreement compares each expert's answer
probabilities on the same training questions. This adapter expects already
computed candidate-conditioned features; it does not generate or encode answers.

## Ordered multi-site tasks

List the ordered site names in `modalities`, and the signal expert names in
`site_experts`. Signal features have shape `[N,S,D]`; `feature_dims` stores D, not
S times D. Non-site context experts retain `[N,D]` vectors. Site experts declare
empty `dependencies`: their internal site mask controls availability, rather than
requiring every site to survive. Signal vectors from absent sites are masked before
projection and their tokens again after site conditioning. Shared content projection,
site embeddings and ordered concatenation retain site identity. Site masks are also
provided to the private and common prediction heads. Context dependencies can be
empty when context is always available.

The graph's expert task heads use full-site training features; the collaboration
model uses learned site adapters for all views. Patients should have disjoint
`group_ids`. Site features and candidate-answer features are separate input types
and cannot be combined in one dataset bundle.

`configs/default.json` supplies runnable defaults, not dataset-specific benchmark
recipes. For regression set `training.selection_metric` to `mae` or `rmse`.
Classification supports `accuracy`, `balanced_accuracy`, `macro_f1`, and `weighted_f1`.

Training writes only one selected checkpoint, expert heads/graph, configuration,
and epoch history. Evaluation writes per-view predictions, JSON metrics and a CSV
table with an equal-view Mean-Missing row. Undefined correlation is reported as
null. Existing run/evaluation directories are never silently overwritten.
