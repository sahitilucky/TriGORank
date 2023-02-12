import numpy as np

class VotingRegression():
    """Prediction voting regressor for unfitted estimators.
    .. versionadded:: 0.21
    A voting regressor is an ensemble meta-estimator that fits base
    regressors each on the whole dataset. It, then, averages the individual
    predictions to form a final prediction.

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import VotingRegressor
    r1 = LinearRegression()
    r2 = RandomForestRegressor(n_estimators=10, random_state=1)
    X = np.array([[1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36]])
    y = np.array([2, 6, 12, 20, 30, 42])
    er = VotingRegressor([('lr', r1), ('rf', r2)])
    print(er.predict(X))
    [ 3.3  5.7 11.8 19.7 28.  40.3]
    """

    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights
        if self.estimators is None or len(self.estimators) == 0:
            raise AttributeError('Invalid `estimators` attribute, `estimators`'
                                 ' should be a list of (string, estimator)'
                                 ' tuples')

        if (self.weights is not None and
                len(self.weights) != len(self.estimators)):
            raise ValueError('Number of `estimators` and weights must be equal'
                             '; got %d weights, %d estimators'
                             % (len(self.weights), len(self.estimators)))

        self.names, self.clfs = zip(*self.estimators)

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.clfs]).T


    @property
    def _weights_not_none(self):
        """Get the weights of not `None` estimators"""
        if self.weights is None:
            return None
        return [w for est, w in zip(self.clfs, self.weights)
                if est[1] not in (None, 'drop')]


    def predict(self, X):
        """Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : array of shape (n_samples,)
            The predicted values.
        """
        return np.average(self._predict(X), axis=1,
                          weights=self._weights_not_none)


    def transform(self, X):
        """Return predictions for X for each estimator.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        predictions
            array-like of shape (n_samples, n_classifiers), being
            values predicted by each regressor.
        """
        return self._predict(X)