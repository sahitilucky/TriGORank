import pandas as pd
import argparse
import random
import numpy as np
from format_data import get_XY
from sklearn.model_selection import learning_curve
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
from skorch import NeuralNetRegressor
import torch
import torch.nn as nn
pd.set_option('display.max_colwidth', -1)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn') #or plt.xkcd()

#TODO: paralellism for NN learning curve:
# https://github.com/skorch-dev/skorch/blob/12842307309765b8a0caefa00441df88ec53ce0d/docs/user/parallelism.rst
def getNN():
    model = torch.nn.Sequential(
        nn.Linear(X_train.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    net = NeuralNetRegressor(
        model,
        max_epochs=5,
        lr=0.001,
        batch_size=64,
        optimizer=torch.optim.SGD,
        iterator_train__shuffle=True,
        device=torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    )
    return net


def mse_norm(y_test, y_pred):
    return mean_squared_error(y_test, y_pred)/np.mean(np.power(y_test,2))

parser = argparse.ArgumentParser("Genetic triple mutation regression models")
# Dataset options
parser.add_argument('-dataset', default='data/triple_fitness.tsv', type=str, help="path to dataset")
parser.add_argument('-dataset_double', default='data/double_fitness.tsv', type=str, help="path to dataset")
parser.add_argument('-model', default="nn", type=str, help="model choice", choices=["sgd", "linear", "rf", "svr", "nn"])
parser.add_argument('-seed', type=int, default=3435, help="seed")
parser.add_argument('-normalize', action='store_true', default=False, help='normalize by squared length')
args = parser.parse_args()

# Set seed
random.seed(args.seed)
np.random.seed(args.seed)

(ids_train, X_train, y_train, train_dict), (ids_test, X_test, y_test, test_dict) = get_XY(args.dataset, args.dataset_double)
if args.model == "nn": y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

switcher = {
    'sgd': linear_model.SGDRegressor(loss='huber', max_iter=1000, tol=1e-3),
    'linear': linear_model.LinearRegression(),
    'rf': RandomForestRegressor(max_leaf_nodes=350),
    'svr': SVR(kernel='linear'),
    'nn': getNN()
}

model = switcher.get(args.model, "Invalid model choice")
model.fit(X_train, y=y_train)
y_pred = model.predict(X_test)
#ids_test, y_pred, y_test = (list(x) for x in zip(*sorted(zip(ids_test, y_pred, y_test), key=lambda pair: pair[-1])))
#for i, (id, pred, target) in enumerate(zip(ids_test, y_test, y_pred)):
#    print('{}: item:{} prediction:{} target:{}'.format(i, id, pred, target))

from scipy import stats
tau, _ = stats.kendalltau(y_pred, y_test)
print('Kendall correlation coefficient: %.5f' % (tau))

# Evaluation metrics: https://www.dataquest.io/blog/understanding-regression-error-metrics
# Because values are very close to zero, thus MSE and MAE will be incorrectly close to zero, normalize with mean (squared) values of y_test
# MAE and MSE scores: 0 is perfect prediction (typically)
print("Mean absolute error: %.2f" % (mean_absolute_error(y_test, y_pred)/np.mean(y_test) if args.normalize else mean_absolute_error(y_test, y_pred)))
print("Mean squared error: %.2f" % (mean_squared_error(y_test, y_pred)/np.mean(np.power(y_test,2)) if args.normalize else mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
# A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0
print('Variance score: %.2f' % r2_score(y_test, y_pred))
print('\n', '-' * 20) # separator


# Learning curves: https://www.dataquest.io/blog/learning-curves-machine-learning
n_experiments = 20
max_size = 50000 #int(X_train.shape[0]*0.8) #due to cv=5 ##
increments = round(int(max_size/n_experiments), 1) #round up to next hundred
train_sizes = list(range(10, max_size, increments-1)) #SOS, starting with 1 example does not work for NN so we start with 10
model = switcher.get(args.model, "Invalid model choice")


neg_mean_squared_error_scorer = make_scorer(mse_norm if args.normalize else mean_squared_error, greater_is_better=False)
train_sizes, train_scores, validation_scores = learning_curve( estimator = model, scoring = neg_mean_squared_error_scorer,
                                                               X = X_train, y = y_train, train_sizes = train_sizes, cv=5)
train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)
print('Mean training scores\n', pd.Series(train_scores_mean, index = train_sizes))
print('\nMean validation scores\n',pd.Series(validation_scores_mean, index = train_sizes))
plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
title = 'Learning curve for ' + str(model).split('(')[0] + ' model'
plt.title(title, fontsize = 14, y = 1.03)
plt.legend()
plt.savefig('.'.join([args.model, 'norm.png' if args.normalize else 'png']) , facecolor='white',edgecolor='none', bbox_inches="tight")
#plt.show()

