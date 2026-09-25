# Lluna

A PyTorch implementation of multi-expert collaboration with a fixed
agreement graph, common/private residual prediction, and progressive common
consistency (PCC).

## Quick start

Requires Python 3.10+ and PyTorch 2.5+.

```bash
python -m pip install -r requirements.txt
python example.py
python -m unittest discover -s tests -v
```

The example creates synthetic features and runs training and evaluation through
the same pipeline used for supplied feature datasets. For a regression example,
run `python example.py --task regression --output demo-regression`.

## Train and evaluate

Prepare frozen expert features as described in [DATA.md](DATA.md), then run:

```bash
python -m lluna.runner train --data data/my_dataset --config configs/default.json --output runs/my_run
python -m lluna.runner evaluate --data data/my_dataset --checkpoint runs/my_run/checkpoint.pt --output runs/my_run/evaluation
```

Use `--device cuda:0` for GPU execution. Expert heads and the graph use training
data only. Checkpoints are selected using validation metrics averaged over missing
views. Evaluation reports every test view and Mean-Missing in JSON and CSV.

## Implementation

- `lluna/graph.py`: prediction agreement and sparse graph construction.
- `lluna/model.py`: expert adapters, common message passing, and residual prediction.
- `lluna/pcc.py`: modality views, availability masks, and one-way PCC.
- `lluna/data.py`: feature bundles and partition validation.
- `lluna/runner.py`: expert heads, graph, training, checkpoint selection and evaluation.
- `lluna/metrics.py`: classification and regression metrics.
- `example.py`: a runnable example with nine experts.

Inputs are frozen expert features and boolean availability masks. PCC aligns
the poorer view's pooled common representation with a detached richer-view
representation. Predictions combine the private output and a learned, bounded
common residual.

The pipeline starts from cached vectors and supports classification and scalar
regression. It enumerates all nonempty modality subsets; sparse view sampling,
raw-data extraction, candidate-answer adapters, and ordered-site medical adapters
are not included. Default settings are examples, not benchmark reproduction recipes.

Dataset preprocessing, additional task adapters, and benchmark experiment
configurations will be released upon publication.
