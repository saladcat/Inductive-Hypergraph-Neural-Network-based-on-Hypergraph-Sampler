import numpy as np


class MetricItem:

    def __init__(self, res) -> None:
        for k, v in res.items():
            self.__setattr__(k, v)

    def __str__(self) -> str:
        return '\n'.join([f'{k}: {v:.5f}' for k, v in self.__dict__.items()])


def cal_metric_all(y_true, y_pred):
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1)).astype(np.float)
    tn = np.sum(np.logical_and(y_true == 0, y_pred == 0)).astype(np.float)
    fp = np.sum(np.logical_and(y_true == 0, y_pred == 1)).astype(np.float)
    fn = np.sum(np.logical_and(y_true == 1, y_pred == 0)).astype(np.float)

    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = tp / (tp + fn)
    spec = tn / (tn + fp)
    bac = (sen + spec) / 2.0
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    res = {
        'ACC': acc,
        'SEN': sen,
        'SPEC': spec,
        'BAC': bac,
        'PPV': ppv,
        'NPV': npv,
    }
    return MetricItem(res)
