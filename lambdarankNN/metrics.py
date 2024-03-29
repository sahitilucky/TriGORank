"""
Metrics:

NDCG, DGC, PRECISION@K, RECALL@K
https://en.wikipedia.ororgg/wiki/Discounted_cumulative_gain
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
"""

import numpy as np
from sklearn import metrics

class DCG(object):

    def __init__(self, k=10, gain_type='exp2'):
        """
        :param k: int DCG@k
        :param gain_type: 'exp2' or 'identity'
        """
        self.k = k
        self.discount = self._make_discount(256)
        if gain_type in ['exp2', 'identity']:
            self.gain_type = gain_type
        else:
            raise ValueError('gain type not equal to exp2 or identity')

    def evaluate(self, targets):
        """
        :param targets: ranked list with relevance
        :return: float
        """
        gain = self._get_gain(targets)
        discount = self._get_discount(min(self.k, len(gain)))
        return np.sum(np.divide(gain, discount))

    def _get_gain(self, targets):
        t = targets[:self.k]
        if self.gain_type == 'exp2':
            return np.power(2.0, t) - 1.0
        else:
            return t

    def _get_discount(self, k):
        if k > len(self.discount):
            self.discount = self._make_discount(2 * len(self.discount))
        return self.discount[:k]

    @staticmethod
    def _make_discount(n):
        x = np.arange(1, n+1, 1)
        discount = np.log2(x + 1)
        return discount


class NDCG(DCG):

    def __init__(self, k=10, gain_type='exp2'):
        """
        :param k: int NDCG@k
        :param gain_type: 'exp2' or 'identity'
        """
        super(NDCG, self).__init__(k, gain_type)

    def evaluate(self, targets):
        """
        :param targets: ranked list with relevance
        :return: float
        """
        dcg = super(NDCG, self).evaluate(targets)
        ideal = np.sort(targets)[::-1]
        idcg = super(NDCG, self).evaluate(ideal)
        return dcg / idcg

    def maxDCG(self, targets):
        """
        :param targets: ranked list with relevance
        :return:
        """
        ideal = np.sort(targets)[::-1]
        return super(NDCG, self).evaluate(ideal)



def precision(target, pred, k):
    y_true = set(target)
    y_pred = set(pred[:k])
    result = len(y_true & y_pred) / float(k)
    return metrics.precision_score(y_true, y_pred)

def recall(target, pred, k):
    y_true = set(target)
    y_pred = set(pred[:k])
    result = len(y_true & y_pred) / float(len(y_true))
    return metrics.recall_score(y_true, y_pred)


# Define the evaluation metric: NDCG
def calc_ndcg(y_true, y_pred, k):
    import torch
    y_pred = torch.squeeze(y_pred, 1)
    dcg = torch.tensor([0.])
    ideal_dcg = torch.tensor([0.])
    y_true_sorted, index_y_true = torch.sort(y_true, descending=True)
    y_pred_sorted, index_y_pred = torch.sort(y_pred, descending=True)
    for i in range(k):
        ideal_dcg += (torch.tensor(2.) ** y_true_sorted[i] - torch.tensor(1.)) / torch.log2(torch.tensor(i) + torch.tensor(2.))
    for i in range(k):
        dcg += (torch.tensor(2.) ** y_true[index_y_pred[i]] - torch.tensor(1.)) / torch.log2(torch.tensor(i) + torch.tensor(2.))
    ndcg = dcg / ideal_dcg
    return ndcg

if __name__ == "__main__":
    targets = [3, 2, 3, 0, 1, 2, 3, 2]
    dcg6 = DCG(6, 'identity')
    ndcg6 = NDCG(6, 'identity')
    assert 6.861 < dcg6.evaluate(targets) < 6.862
    assert 0.785 < ndcg6.evaluate(targets) < 0.786
    ndcg10 = NDCG(10)
    assert 0 < ndcg10.evaluate(targets) < 1.0
    assert 0 < ndcg10.evaluate([1, 2, 3]) < 1.0
