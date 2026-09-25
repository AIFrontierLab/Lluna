"""Task metrics in original target units."""
import torch


def metrics(predictions, labels, task):
    if task == 'regression':
        p, y = predictions.flatten().double(), labels.double()
        centered_p, centered_y = p-p.mean(), y-y.mean()
        denom = centered_p.norm()*centered_y.norm()
        return {'mae': float((p-y).abs().mean()),
                'rmse': float((p-y).square().mean().sqrt()),
                'pearson': float((centered_p*centered_y).sum()/denom) if denom > 0 else None}
    p, y = predictions.argmax(-1), labels.long()
    n = predictions.shape[-1]
    matrix = torch.bincount(y*n+p, minlength=n*n).reshape(n,n).double()
    support, predicted, tp = matrix.sum(1), matrix.sum(0), matrix.diag()
    recall = tp/support.clamp_min(1)
    f1 = 2*tp/(support+predicted).clamp_min(1)
    return {'accuracy': float((p==y).double().mean()),
            'balanced_accuracy': float(recall[support>0].mean()),
            'macro_f1': float(f1[(support+predicted)>0].mean()),
            'weighted_f1': float((f1*support).sum()/support.sum())}
