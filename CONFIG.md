# Configuration

Start with `configs/default.json`. Model dimensions, dropout, graph calibration,
expert-head fitting, and optimizer controls are explicit and saved with each run.
Different datasets may supply different configurations.

## Model and graph

- `model.hidden_dim`: adapted private node width.
- `model.common_dim`: common propagation width; omitted means twice hidden width.
- `model.gnn_layers`: number of gated propagation steps with shared parameters.
- `model.residual_scale_init`: initial bounded residual coefficient, strictly between -1 and 1.
- `model.dropout`: default dropout; optional `adapter_dropout`, `private_head_dropout`
  and `common_dropout` override individual paths.
- `model.site_projection_dim`: optional ordered-site token width, default 64.
- `graph.temperature`, `graph.top_k`, `graph.self_loop`, `graph.floor`: fixed graph
  calibration. A null top-k retains all off-diagonal neighbors. Union-kNN can
  produce degrees larger than k because either direction retains an edge.

## Training objective

The loss is task supervision plus `aux_weight` times common-head supervision and
`pcc_weight` times PCC. PCC uses the detached richer-view common representation as
teacher. `pcc_deadzone` and `pcc_power` shape cosine distance; defaults 0 and 1 leave
it unchanged. Each view uses the same samples and synchronized dropout seed.

Task terms default to equal mass per view. Optional `task_view_weights` must name
every view using alphabetically sorted modality names joined by `+`, and are
normalized to sum to one. PCC defaults to equal mass per child level, then child,
then parent. Optional `pcc_level_weights` maps child modality counts (JSON string
keys such as `"1"` and `"2"`) to relative masses and must cover every child level.

## View sampling

- `view_sampling: "full"` evaluates every view per training step.
- `view_sampling: "chain"` samples uniform random maximal chains with replacement.
  `chains_per_step` controls the number of draws; duplicate views are forwarded once.

Chain coefficients use draw multiplicity divided by expected multiplicity for
both task terms and PCC edges. They preserve the configured full-lattice objective
in expectation and must not be renormalized over only the sampled views. Full
validation and test evaluation are unchanged. Lattices support 2–6 modalities.

## Selection and outputs

`selection_metric` is minimized for regression (`mae` or `rmse`) and maximized for
classification. Candidate tasks use `accuracy`. `patience` counts non-improving
epochs; ties retain the earliest checkpoint. `class_balance` enables inverse
training-frequency weights for ordinary classification, not candidate indexes.

The training command saves one selected checkpoint, head parameters/graph,
configuration, input fingerprints, epoch metrics, and a training summary. Evaluation
is a separate command and produces per-view predictions plus JSON/CSV metrics.
The reported Mean-Missing is an equal average over all non-full views, regardless
of training weights. These are configurable reference runs; exact benchmark
preprocessing and configurations are not bundled here.
