"""
Implementation of LambdaMART.
Edits to be able to use sklearn learning curves
"""

import numpy as np
import sklearn.ensemble
import sklearn.externals
import sklearn.utils
import sklearn.tree
from pyltr.models import LambdaMART
from pyltr import metrics


class Lambda(LambdaMART):
    def __init__(self, metric=None, learning_rate=0.1, n_estimators=100,
                 query_subsample=1.0, subsample=1.0, min_samples_split=2,
                 min_samples_leaf=1, max_depth=3, random_state=None,
                 max_features=None, verbose=0, max_leaf_nodes=None,
                 warm_start=True):
        super(LambdaMART, self).__init__()
        self.metric = metrics.dcg.NDCG() if metric is None else metric
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.query_subsample = query_subsample
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self.max_features = max_features
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start

    def fit(self, X, y, qids=None, monitor=None):
        """Fit lambdamart onto a dataset.

        Parameters
        ----------
        X : array_like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array_like, shape = [n_samples]
            Target values (integers in classification, real numbers in
            regression)
            For classification, labels must correspond to classes.
        qids : array_like, shape = [n_samples]
            Query ids for each sample. Samples must be grouped by query such
            that all queries with the same qid appear in one contiguous block.
        monitor : callable, optional
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator and the local variables of
            ``_fit_stages`` as keyword arguments ``callable(i, self,
            locals())``. If the callable returns ``True`` the fitting procedure
            is stopped. The monitor can be used for various things such as
            computing held-out estimates, early stopping, model introspecting,
            and snapshoting.

        """
        if not self.warm_start:
            self._clear_state()

        if qids is None: qids = [1]*len(y)
        X, y = sklearn.utils.check_X_y(X, y, dtype=sklearn.tree._tree.DTYPE)
        n_samples, self.n_features = X.shape

        sklearn.utils.check_consistent_length(X, y, qids)
        if y.dtype.kind == 'O':
            y = y.astype(np.float64)

        random_state = sklearn.utils.check_random_state(self.random_state)
        self._check_params()

        if not self._is_initialized():
            self._init_state()
            print ('coming here begin_at_stage 0')
            begin_at_stage = 0
            y_pred = np.zeros(y.shape[0])
        else:
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError('n_estimators=%d must be larger or equal to '
                                 'estimators_.shape[0]=%d when '
                                 'warm_start==True'
                                 % (self.n_estimators,
                                    self.estimators_.shape[0]))
            begin_at_stage = self.estimators_.shape[0]
            if (begin_at_stage == self.n_estimators):
                self._init_state()
                print ('coming here begin_at_stage 0')
                begin_at_stage = 0
                y_pred = np.zeros(y.shape[0])
            else:
                print ('coming here begin_at_stage ', begin_at_stage)
                self.estimators_fitted_ = begin_at_stage
                self.estimators_.resize((self.n_estimators, 1))
                self.train_score_.resize(self.n_estimators)
                if self.query_subsample < 1.0:
                    self.oob_improvement_.resize(self.n_estimators)
                y_pred = self.predict(X)

        n_stages = self._fit_stages(X, y, qids, y_pred,
                                    random_state, begin_at_stage, monitor)

        if n_stages < self.estimators_.shape[0]:
            self.trim(n_stages)

        return self


    def get_params(self, deep=True):
        return {'metric': self.metric ,
                'learning_rate': self.learning_rate,
                'n_estimators': self.n_estimators,
                'query_subsample': self.query_subsample ,
                'subsample': self.subsample,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_depth': self.max_depth,
                'random_state': self.random_state,
                'max_features': self.max_features,
                'verbose': self.verbose,
                'max_leaf_nodes': self.max_leaf_nodes,
                'warm_start': self.warm_start}

    def set_params(self, **params):
        """Sets the parameters of this estimator.
                # Arguments
                    **params: Dictionary of parameter names mapped to their values.
                # Returns
                    self
        """
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

