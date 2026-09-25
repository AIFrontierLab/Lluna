"""Generate a synthetic feature bundle and run the standard training pipeline."""
import argparse
import json
from pathlib import Path
import torch
from lluna.data import FeatureDataset
from lluna.runner import train, test


def make_data(root, task='classification'):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    latent = torch.randn(96, 6)
    target = latent[:, 0] + latent[:, 1]
    labels = (target > 0).long() if task == 'classification' else target
    dims = {f'expert_{i}': 12+i for i in range(9)}
    features = {k: latent @ torch.randn(6, d) + .1*torch.randn(96, d) for k,d in dims.items()}
    metadata = dict(task=task, num_classes=2, modalities=list('TAV'), feature_dims=dims,
                    dependencies={k: ['TAV'[i//3]] for i,k in enumerate(dims)})
    torch.save(metadata, root/'metadata.pt')
    for name, start, stop in [('train',0,64), ('validation',64,80), ('test',80,96)]:
        torch.save(dict(ids=[f'example_{i}' for i in range(start,stop)],
                        labels=labels[start:stop], features={k:v[start:stop] for k,v in features.items()}),
                   root/f'{name}.pt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('demo'))
    parser.add_argument('--task', choices=('classification','regression'), default='classification')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    make_data(args.output/'data', args.task)
    config = json.loads((Path(__file__).parent/'configs/default.json').read_text())
    config['model'].update(hidden_dim=16, common_dim=32)
    config['training'].update(epochs=5, patience=5)
    if args.task == 'regression':
        config['training']['selection_metric'] = 'mae'
    data = FeatureDataset(args.output/'data')
    train(data, config, args.output/'run', torch.device('cpu'))
    test(data, args.output/'run/checkpoint.pt', args.output/'evaluation', torch.device('cpu'))


if __name__ == '__main__':
    main()
