from sklearn.base import BaseEstimator
import numpy as np

class Baselines(BaseEstimator):
    def __init__(self, baseline='combo'):
        super(Baselines, self).__init__()
        features = ['query1_score', 'query2_score', 'array_score', 'query1_array_score', 'query2_array_score', 'query1_query2_score']
        self.baseline = baseline
        switcher = {
            'w12': [features.index('query1_query2_score')],
            'w13': [features.index('query1_array_score')],
            'w23': [features.index('query2_array_score')],
        }
        self.index = switcher.get(self.baseline, 'combo')
        if self.baseline == 'combo':
            self.index = [features.index('query1_array_score'), features.index('query2_array_score'), features.index('query1_query2_score')]


    def fit(self, X, y):
        return self


    def predict(self, X, y=None):
        return np.mean(X[:,self.index], axis=1)


    def get_params(self, deep=True):
        return {'baseline': self.baseline}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

