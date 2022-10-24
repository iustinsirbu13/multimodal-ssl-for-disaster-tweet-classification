'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
'''
import logging
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter', 'binary_metrics', 'all_metrics']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def all_metrics(output, target, n_decimals=None):
    output = output.argmax(dim=1).cpu()
    target = target.cpu()
    all_metrics = {}
    eval_methods = {
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
    }
    
    for average in [None, 'micro', 'macro', 'weighted']:
        for name, func in eval_methods.items():
            score = func(target, output, average=average)
            if isinstance(score, np.ndarray):
                score = score.tolist()
            all_metrics[f'{average}/{name}'] = score
    all_metrics['accuracy'] = accuracy_score(target, output)
    if n_decimals is not None:
        all_metrics = round_metrics(all_metrics, n_decimals)
    return all_metrics

def binary_metrics(output, target):
    output = output.argmax(dim=1).cpu()
    target = target.cpu()
    binary_metrics = {
#         'label_0/precision': precision_score(target, output, pos_label=0),
#         'label_0/recall': recall_score(target, output, pos_label=0),
#         'label_0/f1': f1_score(target, output, pos_label=0),
#         'label_1/precision': precision_score(target, output, pos_label=1),
#         'label_1/recall': recall_score(target, output, pos_label=1),
#         'label_1/f1': f1_score(target, output, pos_label=1),
        'micro/precision': precision_score(target, output, average='micro', zero_division=0),
        'micro/recall': recall_score(target, output, average='micro', zero_division=0),
        'micro/f1': f1_score(target, output, average='micro', zero_division=0),
        'macro/precision': precision_score(target, output, average='macro', zero_division=0),
        'macro/recall': recall_score(target, output, average='macro', zero_division=0),
        'macro/f1': f1_score(target, output, average='macro', zero_division=0),
#         'accuracy': accuracy_score(target, output),
    }
    return binary_metrics


def round_metrics(metrics, n_decimals):
    r_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, list):
            v = [round(e, n_decimals) for e in v]
        else:
            v = round(v, n_decimals)
        r_metrics[k] = v
    return r_metrics


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
