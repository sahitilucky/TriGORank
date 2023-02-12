import pandas as pd
import time
import copy
import os
import csv
import datetime
import logging
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer
import pyltr
from torch import cuda
import torch
import torch.nn as nn
from skorch import NeuralNetRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel
from sklearn.neural_network import MLPRegressor
from lamdamart import Lambda
from baselines import Baselines
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import math

plt.style.use('seaborn') #or plt.xkcd()
pd.set_option('display.max_colwidth', -1)
logger = logging.getLogger(__name__)

def get_trigenic_data(fname_triple = '../../../Data/work/DeepMutationsData/data/triple_fitness.tsv', ids = ['query1', 'query2','arr']):
    target = 'triple_score'
    #ids = ['query1', 'query2','arr']
    triple = pd.read_csv(fname_triple, sep='\t')
    triple.drop(['Unnamed: 0'], axis=1, inplace=True)
    triple.drop_duplicates(subset=ids, keep=False, inplace=True)
    triple.dropna(inplace=True)
    triple = shuffle(triple)
    
    print ('Mean: %.3f' % np.mean(triple['triple_score']))
    print ('Variance: %.3f' % np.var(triple['triple_score']))
    print ('Standard Deviation: %.3f' % np.std(triple['triple_score']))
    print ('====================================')

    triple["rank"] = triple[target].rank(ascending=1, method='first')
    triple["reverserank"] = triple[target].rank(ascending=0, method='first')
    triple['ids'] = triple[ids].apply(lambda x: '_'.join(x), axis=1)
    return triple

def name_go_terms_list(go_terms_list, go):
    pop_go_terms = []
    for go_term in go_terms_list:
        try:
            pop_go_terms += [(go_term, go[go_term]["name"])]
        except KeyError:
            pop_go_terms += [(go_term, "NA")]
    return pop_go_terms

def get_model(model_str, ndcg_train):
    switcher = {
        'combo_baseline': Baselines('combo'),
        'w12_ baseline': Baselines('w12'),
        'w13_baseline': Baselines('w13'),
        'w23_baseline': Baselines('w23'),
        'sgd': SGDRegressor(loss='huber', max_iter=1000, tol=1e-3),
        'linear': LinearRegression(),
        'ridge': Ridge(),
        'polynomial': getPolynomialRegressor(bagging=False),
        'polynomial_bagging': getPolynomialRegressor(bagging=True),
        'gaussian': GaussianProcessRegressor(normalize_y=True),
        'rf': RandomForestRegressor(max_leaf_nodes=350, criterion = 'mae',bootstrap = True, n_jobs=1,max_depth = 100,max_features = 'auto',min_impurity_decrease = 0.0, min_samples_leaf = 4, min_samples_split = 10,min_weight_fraction_leaf = 0.0, n_estimators = 1000,oob_score = False, random_state = None, verbose = 0, warm_start = False),
        'svr': SVR(kernel='linear'),
        'mlp': MLPRegressor(hidden_layer_sizes=(5, 5, 5), learning_rate_init=0.1, learning_rate='adaptive',
                            solver='sgd', activation='tanh', max_iter=1000),
        'nn': getNN(input_size=6),  # X_train.shape[1]
        'lambda': Lambda(metric=ndcg_train, n_estimators=1000, learning_rate=0.01, max_features='auto',
                         query_subsample=1.0, max_leaf_nodes=350, min_samples_leaf=4, verbose=0)
    }
    model = switcher.get(model_str.lower().strip(), "Invalid model choice")
    return model


def parameter_tuning(model_str):
    #alpha: 0.1 to 10^-7
    switcher = {
        'sgd': {'loss': ['squared_loss', 'huber'], 'penalty': ['none', 'l2', 'l1', 'elasticnet'],'alpha': 10.0 ** -np.arange(1, 5), 'l1_ratio': [.05, .5, .7, .95, 1],'learning_rate': ['optimal', 'adaptive', 'invscaling'], 'early_stopping': [True]},
        'ridge': {'alpha' : 10.0**-np.arange(1,11)},
        'gaussian': {'kernel': [ConstantKernel(),  RBF(), Matern(), WhiteKernel()]},
        'rf': {'n_estimators':[50,100,500,1000], 'criterion':['mse', 'mae'], 'max_depth':[None, 100, 350, 500, 1000],
               'min_samples_leaf': [1,2,4], 'min_samples_split': [2, 5, 10]},      
        'svr': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma':['auto', 'scale']},
        'mlp': {'hidden_layer_sizes': [(5, 5), (32, 32), (16, 16), (64, 32)],'activation': ['logistic', 'tanh', 'relu'],'solver': ['sgd', 'adam'], 'batch_size': ['auto', 32, 64], 'early_stopping': [True], 'max_iter': [1000],'learning_rate': ['constant', 'adaptive', 'invscaling']},
        #'lambda': {'n_estimators': [50, 100, 500, 1000], 'learning_rate': [0.1, 0.01, 0.001],'max_leaf_nodes': [None, 100, 350, 500, 1000], 'min_samples_leaf': [1, 2, 4]}
        'lambda': {'learning_rate': [0.1, 0.01, 0.001]}
    }
    # 'rf': {'n_estimators':[10,50,100,500,1000], 'criterion':['mse', 'mae'], 'max_depth':[None, 100, 350, 500, 1000],
    #       'min_samples_leaf': [1, 2, 4], 'min_samples_split': [2, 5, 10]},
        # 'mlp': {'hidden_layer_sizes': [(5, 5, 5), (64, 64, 64), (16, 16, 16), (64, 32, 64), (100, 100, 100), (64, 32)], 'activation': ['logistic', 'tanh', 'relu'],
    #       'solver': ['sgd', 'adam', 'lbfgs'], 'batch_size':['auto', 32, 64], 'early_stopping': [True], 'max_iter': [1000],
    #       'learning_rate':['constant', 'adaptive', 'invscaling'], 'learning_rate_init': [0.001, 0.0001, 0.01]}
    # ,'lambda': {'n_estimators':[10,50,100,500,1000],'learning_rate': [0.1, 0.01, 0.001],
    #            'max_leaf_nodes':[None, 100, 350, 500, 1000], 'min_samples_leaf': [1, 2, 4]}
    # 'sgd':  {'loss':['squared_loss', 'huber'], 'penalty' : ['none', 'l2', 'l1',  'elasticnet'],
    #           'alpha' : 10.0**-np.arange(1,7),'l1_ratio':[.05, .15, .5, .7, .9, .95, .99, 1],
    #           'learning_rate': ['optimal', 'adaptive', 'invscaling'], 'early_stopping': [True]},
    param_grid = switcher.get(model_str.lower().strip(), "Invalid model choice")
    return  param_grid


#TODO: paralellism for NN learning curve:
# https://github.com/skorch-dev/skorch/blob/12842307309765b8a0caefa00441df88ec53ce0d/docs/user/parallelism.rst
def getNN(input_size):
    model = torch.nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    net = NeuralNetRegressor(
        model,
        max_epochs=5,
        lr=0.1,
        batch_size=64,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    return net


def getPolynomialRegressor(bagging=True):
    if bagging:
        polynomialRegressor = Pipeline([('Polynomial Features',
                                     PolynomialFeatures(degree=3, include_bias=True, interaction_only=True)),
                                    ('Linear Regressor', BaggingRegressor(base_estimator=Ridge(),
                                                                          n_estimators=100, max_samples=.8))])
    else:
        polynomialRegressor = Pipeline([('Polynomial Features',
                                         PolynomialFeatures(degree=3, include_bias=True, interaction_only=True)),
                                        ('Linear Regressor', Ridge())])
    return polynomialRegressor



def ndcg_scoring(y_true, y_predicted, ndcg_train, train_k):
    y_true_temp = copy.deepcopy(y_true)
    y_true_temp = sorted(range(len(y_true_temp)), reverse=True, key=lambda i: y_true_temp[i])
    topitems = y_true_temp[:train_k]
    y_true = np.array([1 if i in topitems else 0 for i in range(len(y_true))])
    return ndcg_train.calc_mean(np.ones(len(y_true)), y_true, y_predicted)

def precision_recall_k(y_pred, y, k, ids, X=None, writer=None):
    total, correct = 0, 0
    relevant_ids, topranked = [], []
    all_relevant = sum(y >= 1)
    ids, y_pred, y = (list(x) for x in zip(*sorted(zip(ids, y_pred, y), reverse=True, key=lambda pair: pair[1])))
    if X is None: X = ids #not the best practice
    for i, (id, x, pred, target) in enumerate(zip(ids, X, y_pred, y)):
        if i >= k: break
        total +=1
        if writer: writer.writerow([id, x, pred, target])
        if target >= 1:
            correct+=1
            relevant_ids.append(id)
            #if writer: print(id, x, pred, target)
    recall = correct/float(all_relevant)
    precision = correct/float(total)
    return precision, recall, correct, total, relevant_ids, topranked


def learning_curve_ltr(model, model_name, X_train, y_train, scoring, metric_name, save_dir, n_experiments = 20):
    # Learning curves: https://www.dataquest.io/blog/learning-curves-machine-learning
    max_size = int(X_train.shape[0]*0.5) #due to cv=5 ##
    increments = round(int(max_size/n_experiments), -1) #round up to next hundred
    train_sizes = list(range(10, max_size, increments-1))
    print('max_size', max_size)
    print('train_sizes', train_sizes)
    ndcg_scorer = make_scorer(scoring, greater_is_better=True)
    train_sizes, train_scores, validation_scores = learning_curve(scoring=ndcg_scorer, X=X_train, y=y_train, train_sizes=train_sizes, cv=5,
                                                                  estimator=model)
    train_scores_mean = train_scores.mean(axis = 1)
    validation_scores_mean = validation_scores.mean(axis = 1)
    print('Mean training scores\n', pd.Series(train_scores_mean, index = train_sizes))
    print('\nMean validation scores\n',pd.Series(validation_scores_mean, index = train_sizes))
    plt.clf()
    plt.cla()
    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
    plt.ylabel(metric_name, fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    title = 'Learning curve for {} model'.format(model_name)
    plt.title(title, fontsize = 14, y = 1.03)
    plt.legend()
    plt.savefig(os.path.join(save_dir, '.'.join(['LambdaMART', metric_name, 'png'])) , facecolor='white',edgecolor='none', bbox_inches="tight")


def learning_curves(args, save_dir, model, X_train, y_train, X_test, ndcg_train):
    for k in args.eval_k:
        ndcg_k = pyltr.metrics.NDCG(k=k)
        print('Proceed to learning curves for top-{}...'.format(k))
        def ndcg_scoring(y_test, y_pred):
            return ndcg_k.calc_mean(np.ones(len(y_test)), y_test, y_pred)
        def precision(y_test, y_pred):
            p, r, corr, tot, relids, topranked = precision_recall_k(y_pred, y_test, k, np.ones(len(y_test)), X_test)
            return p
        def recall(y_test, y_pred):
            p, r, corr, tot, relids, topranked = precision_recall_k(y_pred, y_test, k, np.ones(len(y_test)), X_test)
            return r
        start = time.time()
        learning_curve_ltr(copy.deepcopy(model), args.model, X_train, y_train, ndcg_scoring, metric_name='nDCG', save_dir=save_dir, n_experiments=20)
        print('{} for NDCG learning curve'.format(str(datetime.timedelta(seconds=int(time.time() - start)))))
        start = time.time()
        learning_curve_ltr(copy.deepcopy(model), args.model, X_train, y_train, precision, metric_name='precision', save_dir=save_dir, n_experiments=20)
        print('{} for precision learning curve'.format(str(datetime.timedelta(seconds=int(time.time() - start)))))
        start = time.time()
        learning_curve_ltr(copy.deepcopy(model), args.model, X_train, y_train, recall, metric_name='recall', save_dir=save_dir, n_experiments=20)
        print('{} for recall learning curve'.format(str(datetime.timedelta(seconds=int(time.time() - start)))))


def write_all_instances(X, y, dict, ids, filename):
    rs, ds, Xs, ys = (list(x) for x in zip(*sorted(zip(ids, dict.values(), X, y), reverse=True, key=lambda pair: pair[-1])))
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for i, (id, name, x, y) in enumerate(zip(rs, ds, Xs, ys)):
            writer.writerow([id, name, x, y])
    with open(filename + "_only_ids.txt", 'w') as f:
        ids = list(dict.values())
        for i,id in enumerate(ids):
            f.write(id + "\n")


def read_all_instances(filename):
    train_ids = {}
    with open(filename, newline='') as csvfile:
        train_examples = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in train_examples:
            if len(row) > 1:
                if "_" in row[1]:
                    name= row[1]
                    sorted_ids = "_".join(sorted(name.split("_")))
                    try:
                        train_ids[sorted_ids] += 1
                    except:
                        train_ids[sorted_ids] = 1
    for id in train_ids:
        if train_ids[id]>1:
            print ("Id: {} frequency: {}".format(id, train_ids[id]))
    print (len(train_ids))

def write_relevance_judgments_topranked(ids, y, rel_jud_file):
    rel_jud = csv.writer(open(rel_jud_file, 'w'), delimiter='\t')
    id_t, y_t = (list(x) for x in zip(*sorted(zip(ids, y), reverse=True, key=lambda pair: pair[1])))
    for i, (id, target) in enumerate(zip(id_t, y_t)):
        rel_jud.writerow(['Q1', id, target])

def post_processing_top_ranked(examples_folder,result_ranked_folder, result_ranked_prefix):
    test_ks = ["test10","test30","test100","test200"]
    x_tests = {}
    with open(os.path.join(examples_folder,"test_examples"), "r") as infile:
        for line in infile:
            items = line.strip().split("\t")
            id_test = items[0]
            if len(items) > 1:
                ds = items[1]
                if (len(ds.split("_")) == 3):
                   x_tests[id_test] = ds
    for k in test_ks:
        results_ranked = []
        with open(os.path.join(result_ranked_folder, result_ranked_prefix+k+".csv"), "r") as infile:
            for line in infile:
                q,id_test,score =line.strip().split("\t")
                results_ranked += [[id_test, x_tests[id_test].split("_")[0], x_tests[id_test].split("_")[1], x_tests[id_test].split("_")[2], score]]
        with open(os.path.join(result_ranked_folder, "top_ranked_" + k.split("test")[1] + ".txt"), "w") as outfile:
            for result in results_ranked:
                outfile.write(result[0] + "\t" + result[1] + "\t" + result[2] + "\t" + result[3] + "\t" + result[4] + "\n")

if __name__== "__main__":
    #post_processing_top_ranked("New_experiments/onto_exps_all_tune_train_200_2/RF/all_examples", "New_experiments/onto_exps_all_tune_train_200_2/RF/all/result_ranked/", "train200")
    read_all_instances("New_experiments/onto_exps_all_tune_train_200_2/RF/all_examples/train_examples")

