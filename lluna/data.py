"""Validated feature bundles with explicit, disjoint dataset partitions."""
from pathlib import Path
import torch
from .model import ExpertSpec
from .pcc import modality_views, masked_view


class FeatureDataset:
    def __init__(self, path):
        self.root = Path(path)
        meta = torch.load(self.root / 'metadata.pt', map_location='cpu', weights_only=True)
        self.task = meta['task']
        if self.task not in ('classification', 'regression'):
            raise ValueError('Supported tasks: classification, regression')
        self.dependencies = {k: tuple(v) for k, v in meta['dependencies'].items()}
        self.experts = [ExpertSpec(k, int(v)) for k, v in meta['feature_dims'].items()]
        if set(self.dependencies) != {e.name for e in self.experts}:
            raise ValueError('Expert metadata differs')
        self.modalities = tuple(meta['modalities'])
        if any(not set(v) <= set(self.modalities) for v in self.dependencies.values()):
            raise ValueError('Unknown modality dependency')
        self.views = modality_views(self.modalities)
        self.output_dim = int(meta['num_classes']) if self.task == 'classification' else 1
        if self.task == 'classification' and self.output_dim < 2:
            raise ValueError('Classification needs at least two classes')
        self.splits = {}
        # Test data is deliberately not loaded during training or selection.
        for name in ('train', 'validation'):
            self.load_split(name)
        self.label_mean = 0.0
        self.label_std = 1.0
        if self.task == 'regression':
            y = self.splits['train']['labels'].float()
            self.label_mean = float(y.mean())
            self.label_std = max(float(y.std(unbiased=False)), 1e-8)

    def load_split(self, name):
        if name in self.splits:
            return self.splits[name]
        if name not in ('train', 'validation', 'test'):
            raise ValueError('Unknown partition')
        data = torch.load(self.root / f'{name}.pt', map_location='cpu', weights_only=True)
        ids, y, features = data['ids'], data['labels'], data['features']
        if not ids or len(set(ids)) != len(ids) or not all(isinstance(i, str) for i in ids):
            raise ValueError('IDs must be unique nonempty strings')
        if any(set(ids) & set(s['ids']) for s in self.splits.values()):
            raise ValueError('Partitions overlap; use patient/group-disjoint IDs')
        if y.shape != (len(ids),) or not torch.isfinite(y).all():
            raise ValueError('Labels must be finite [N]')
        if self.task == 'classification' and (y.dtype != torch.long or y.min() < 0 or y.max() >= self.output_dim):
            raise ValueError('Class labels must be int64 in [0, num_classes)')
        if set(features) != {e.name for e in self.experts}:
            raise ValueError('Feature keys differ')
        for e in self.experts:
            x = features[e.name]
            if x.shape != (len(ids), e.feature_dim) or not x.is_floating_point() or not torch.isfinite(x).all():
                raise ValueError(f'Invalid features for {e.name}')
        self.splits[name] = data
        return data

    def batch(self, split, indexes, view, device):
        data = self.splits[split]
        features = {k: v[indexes].float().to(device) for k, v in data['features'].items()}
        labels = data['labels'][indexes].to(device)
        if self.task == 'regression':
            labels = (labels.float() - self.label_mean) / self.label_std
        return masked_view(features, self.dependencies, view), labels


def view_name(view):
    return '+'.join(sorted(view))
