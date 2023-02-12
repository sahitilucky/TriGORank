import pandas as pd
import argparse
import copy
import os
import csv
import sys
import pickle
import random
import numpy as np
from format_data_2 import get_XY
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from lamdamart import Lambda
from collections import  defaultdict
from itertools import combinations, permutations, chain
import pyltr
from torch.backends import cudnn
from torch import cuda
import torch
import multiprocessing
from functools import partial
from scipy import misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.style.use('seaborn') #or plt.xkcd()
from sklearn.utils import shuffle
import time
pd.set_option('display.max_colwidth', -1)
from utils import ndcg_scoring, parameter_tuning, get_model, write_all_instances, write_relevance_judgments_topranked, precision_recall_k, learning_curves

#SOS: drop_duplicates(keep=False) removes any duplicate (keeps no version)
#I am using it when creating the triple_fitness.tsv and double_fitness.tsv but we have to discuss how to handle duplicates


def plot_allmodels(models_fig, pr_lines_permodel, dir, filename='allmodels'):
    #All models plots
    cm = plt.get_cmap('tab20')
    NUM_COLORS = len(models_fig)
    for test_k in pr_lines_permodel:
        count_colors = 0
        # ax.set_prop_cycle('color', colors)([cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
        plt.clf()
        plt.cla()
        for precision, recall, label in pr_lines_permodel[test_k]:
            if label in models_fig:
                plt.step(recall, precision, where='post', label=label, color=cm(1. * count_colors / NUM_COLORS))
                count_colors = count_colors+1
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall')
        plt.legend(loc="best")
        plt.savefig(os.path.join(dir,  '.'.join(['PRcurve', filename + str(test_k), 'png'])), facecolor='white',
                    edgecolor='none', bbox_inches="tight")


def get_features(dataset, dataset_double, save_dir, labeling, onto_features):
    (ids_train, X_train, y_train, train_dict), (ids_test, X_test, y_test, test_dict) = get_XY(dataset, dataset_double, test_size=0.3, labeling=labeling, onto_features=onto_features)
    # (ids_train, X_train, y_train, train_dict), (ids_val, X_val, y_val, val_dict), (ids_test, X_test, y_test, test_dict) = get_XY(args.dataset, val_size=0.3, test_size=0.3, labeling=labeling)
    print('Train relevant:{}, irrelevant:{}, percentage:{}'.format(sum(y_train >= 1), sum(y_train < 1), sum(y_train >= 1) / float(sum(y_train >= 1) + sum(y_train < 1))))
    # print('Val relevant:{}, irrelevant:{}, percentage:{}'.format(sum(y_val >= 1), sum(y_val < 1), sum(y_val >= 1)/float(sum(y_val >= 1)+sum(y_val < 1))))
    print('Test relevant:{}, irrelevant:{}, percentage:{}'.format(sum(y_test >= 1), sum(y_test < 1), sum(y_test >= 1) / float(sum(y_test >= 1) + sum(y_test < 1))))
    print('output min:%.3f - max:%.3f' % (y_train.min(), y_train.max()))
    # https://github.com/jma127/pyltr/issues/14 - we only have one session
    tids = np.ones(len(y_train))
    # vids = np.ones(len(y_val))
    qids = np.ones(len(y_test))
    if not os.path.exists(os.path.join(save_dir, 'relevance_judgments')): os.makedirs(os.path.join(save_dir, 'relevance_judgments'))
    if not os.path.exists(os.path.join(save_dir, 'result_ranked')): os.makedirs(os.path.join(save_dir, 'result_ranked'))
    if not os.path.exists(os.path.join(save_dir, 'topranked')): os.makedirs(os.path.join(save_dir, 'topranked'))
    if not os.path.exists(os.path.join(save_dir, 'all_examples')): os.makedirs(os.path.join(save_dir, 'all_examples'))
    write_all_instances(X_test, y_test, test_dict, ids_test, os.path.join(save_dir, 'all_examples', 'test_examples'))
    write_all_instances(X_train, y_train, train_dict, ids_train, os.path.join(save_dir, 'all_examples', 'train_examples'))
    # write_all_instances(X_val, y_val, val_dict, ids_val, os.path.join(save_dir, 'all_examples', 'val_examples'))
    return (ids_train, X_train, y_train, train_dict, tids), (ids_test, X_test, y_test, test_dict, qids)


def eval_curves(args, train_k, save_dir):
    pr_lines = []
    for k in args.eval_k:
        print('\nEvaluation for top-{} relevant items:'.format(k))
        rel_jud = os.path.join(save_dir, 'relevance_judgments', 'train'+str(train_k)+'test'+str(k)+'.csv')
        rel_res = os.path.join(save_dir, 'result_ranked', 'train'+str(train_k)+'test'+str(k)+'.csv')
        process_str = "perl ireval.pl -j "+rel_jud+" < "+rel_res
        print('\n')
        stream = os.popen(process_str).read()
        print(stream)
        start_reading = False
        prec, rec= [], []
        for line in stream.split('\n'):
            if line == 'Interpolated precsion at recalls:':
                start_reading = True
            elif line == 'Non-interpolated precsion at docs:':
                start_reading = False
            elif start_reading:
                pr = line.split('=')[-1].strip()
                point = line.split('=')[0].split('at')[-1].strip()
                prec.append(float(pr))
                rec.append(float(point))
        #y_pred_pr, y_test_pr = (list(x) for x in zip(*sorted(zip(y_pred, y_test), reverse=True, key=lambda pair: pair[0])))
        #prec, rec, threshold = precision_recall_curve(y_test_pr, y_pred_pr)
        print('precision', len(prec), prec)
        print('recall', len(rec), rec)
        pr_lines.append((prec, rec, 'test.top'+str(k)))
    return pr_lines


def train_model(args, model, train_set, test_set, train_k, labeling, save_dir):
    features = ['query1_score', 'query2_score', 'array_score', 'query1_array_score', 'query2_array_score', 'query1_query2_score']
    ids_train, X_train, y_train, train_dict, tids = train_set
    ids_test, X_test, y_test, test_dict, qids = test_set
    model_filename = os.path.join(save_dir, 'model'+str(train_k)+'.pkl')
    if args.model.lower().strip()  == "nn": y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

    if os.path.exists(model_filename):
        model = pickle.load(open(model_filename, 'rb'))
    elif args.tune_model and args.model.lower().strip() in ['sgd', 'ridge', 'gaussian', 'rf', 'svr', 'mlp']:#, 'lambda']:
        ndcg_k = pyltr.metrics.NDCG(k=train_k)
        modelopt = RandomizedSearchCV(model, param_distributions=parameter_tuning(args.model), cv=5, n_iter=10,
                verbose=True, n_jobs=-1, scoring= make_scorer(ndcg_scoring,greater_is_better=True, ndcg_train=ndcg_k), error_score=np.nan)
        modelopt.fit(X_train, y_train)
        print('RandomizedSearchCV best model', modelopt.best_estimator_)
        # fit the best model and save to disk
        modelopt.best_estimator_.fit(X_train, y_train)
        pickle.dump(modelopt.best_estimator_, open(model_filename, 'wb'))
        model = modelopt.best_estimator_
    else:
        # if isinstance(model, Lambda):
        # monitor = pyltr.models.monitors.ValidationMonitor(X_val, y_val, vids, metric=ndcg_train, stop_after=100)
        # model.fit(X_train, y_train, tids, monitor=monitor)
        model.fit(X_train, y_train)
        # save the model to disk
        pickle.dump(model, open(model_filename, 'wb'))

    #retrain a copy of the model on all data
    copy_model = copy.deepcopy(model)

    X_all = np.vstack((X_train, X_test))
    print(X_all.shape, y_train.shape, y_test.shape)
    y_all = np.concatenate((y_train, y_test), axis=0)
    X_all, y_all = shuffle(X_all, y_all)
    copy_model.fit(X_all, y_all)
    pickle.dump(copy_model, open(os.path.join(save_dir, 'model' + str(train_k) + 'all.pkl'), 'wb'))

    print('Training finished...')
    #model = pickle.load(open(model_filename, 'rb'))
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        print ('Number of features: ', len(importances))
        #print('feature_importances:', list(zip(features, importances)))
        print('feature_importances2:', list(zip(['feature'+str(i) for i in range(len(importances))] , importances)))

    if args.print_trees and isinstance(model, RandomForestRegressor):
        for count, tree_in_forest in enumerate(model.estimators_):
            export_graphviz(tree_in_forest, out_file=os.path.join(save_dir, str(count)+'tree.dot'),
                            feature_names=features, filled=True, rounded=True)
            os.system('dot -Tpng '+os.path.join(save_dir, str(count)+'tree.dot')+' -o '+os.path.join(save_dir, 'tree' + str(count) + '.png'))

    y_pred_train = model.predict(X_train)
    if labeling.lower().strip() != 'transfer':
        precision, recall,  correct, total, relids, topranked = precision_recall_k(y_pred_train, y_train, train_k, ids_train, X_train)
        print("Train correct:{} total:{} precision:{} recall:{}".format(correct, total, precision, recall))
    return model


def run(args, model, train_set, test_set, train_k, labeling, ndcg_train, save_dir):
    ids_train, X_train, y_train, train_dict, tids = train_set
    ids_test, X_test, y_test, test_dict, qids = test_set
    if args.learning_curves: copy_model = copy.deepcopy(model)
    model = train_model(args, model, train_set, test_set, train_k, labeling, save_dir)
    y_pred = model.predict(X_test)
    y_test_temp = copy.deepcopy(y_test)
    y_test_temp = sorted(range(len(y_test_temp)), reverse=True, key=lambda i: y_test_temp[i])

    pr_lines = []
    for k in args.eval_k:
        print('\nEvaluation for top-{} relevant items:'.format(k))
        topitems = y_test_temp[:k]
        y_test = np.array([1 if i in topitems else 0 for i in range(len(y_test_temp))])
        print('Test relevant:{}, irrelevant:{}, percentage:{}'.format(sum(y_test >= 1), sum(y_test < 1), sum(y_test >= 1) / float(sum(y_test >= 1) + sum(y_test < 1))))
        rel_jud = os.path.join(save_dir, 'relevance_judgments', 'train'+str(train_k)+'test'+str(k)+'.csv')
        rel_res = os.path.join(save_dir, 'result_ranked', 'train'+str(train_k)+'test'+str(k)+'.csv')
        write_relevance_judgments_topranked(ids_test, y_pred, rel_res)
        write_relevance_judgments_topranked(ids_test, y_test, rel_jud)

        top_file = os.path.join(save_dir, 'topranked', '.'.join(['train' + str(train_k), 'test' + str(k), 'csv']))
        writer = csv.writer(open(top_file, 'w'), delimiter='\t')
        precision, recall, correct, total, relids, topranked = precision_recall_k(y_pred, y_test, k, ids_test, X_test, writer=writer)
        print("Our model correct:{} total:{} precision:{} recall:{}".format(correct, total, precision, recall))

        ndcg_k = pyltr.metrics.NDCG(k=k)
        print('NDCG Random ranking:', ndcg_k.calc_mean_random(qids, y_test))
        print('NDCG Our model:', ndcg_k.calc_mean(qids, y_test, y_pred))

        process_str = "perl ireval.pl -j "+rel_jud+" < "+rel_res
        print('\n')
        stream = os.popen(process_str).read()
        print(stream)

        start_reading = False
        prec, rec= [], []
        for line in stream.split('\n'):
            if line == 'Interpolated precsion at recalls:':
                start_reading = True
            elif line == 'Non-interpolated precsion at docs:':
                start_reading = False
            elif start_reading:
                pr = line.split('=')[-1].strip()
                point = line.split('=')[0].split('at')[-1].strip()
                prec.append(float(pr))
                rec.append(float(point))

        #y_pred_pr, y_test_pr = (list(x) for x in zip(*sorted(zip(y_pred, y_test), reverse=True, key=lambda pair: pair[0])))
        #prec, rec, threshold = precision_recall_curve(y_test_pr, y_pred_pr)
        print('precision', len(prec), prec)
        print('recall', len(rec), rec)
        pr_lines.append((prec, rec, 'test.top'+str(k)))

    # Learning curves
    if args.learning_curves:
        learning_curves(args, save_dir, copy_model, X_train, y_train, X_test, ndcg_train)
    return pr_lines


if __name__== "__main__":
    parser = argparse.ArgumentParser("Genetic triple mutation ltr models")
    parser.add_argument('-onto_features', type=int, default=1, help="use ontology features")
    parser.add_argument('-dataset', default='data/triple_fitness.tsv', type=str, help="path to dataset")
    parser.add_argument('-dataset_double', default='data/double_fitness.tsv', type=str, help="path to dataset")
    parser.add_argument('-eval_k', default="10,30,100,200", type=str, help="list of k cutoff points e.g. 10,30,100,200 for evaluation. First one is also used for training") #10,30,100,200 #new200
    parser.add_argument('-seed', type=int, default=1, help="seed")
    parser.add_argument('-model', default="rf", type=str, help="model choice", choices=["sgd", "linear", "rf", "svr", "nn", "mlp", "lambda"])
    parser.add_argument('-labeling', type=str, default='top10,30,100,200', choices=['binarize', 'topk', 'transfer'], help="binarize: make targets 0/1, topk: top k as relevant, transfer: keep all above 1 to test")
    parser.add_argument('-learning_curves', type=int, default=0, help="make learning curves")
    parser.add_argument('-print_trees', type=int, default=0, help="visualize trees from random forest")
    parser.add_argument('-tune_model', type=int, default=0, help="perform hyper-parameter tuning")
    parser.add_argument("-save_dir", type=str, default='onto_exps', help="directory for saving results and models")
    parser.add_argument('-index_k', default=0, type=int, help="index of which eval_k to use, need serious restructuring to be able to do everything!")

    args = parser.parse_args()
    parent_dir = args.save_dir
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    # Set seed
    torch.backends.cudnn.enabled = False
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    temp_args = copy.deepcopy(args)
    #Run for all models at once
    #TODO: Check if issue with skopt and sklean is fixed https://github.com/scikit-learn/scikit-learn/issues/9857#issuecomment-515968932
    models = ["MLP", "RF",  "SGD", "Polynomial", "Gaussian", "Linear", "Ridge", "lambda"]#, "NN", 'w12_baseline', 'w13_baseline', 'w23_baseline', 'combo_baseline', "lambda"] # "SVR", "MLP", "RF",  "SGD", "Polynomial", "Polynomial_Bagging",
    pr_lines_permodel = defaultdict(list)
    for model in models:
        args.model = model
        args.save_dir = os.path.join(args.save_dir, args.model)
        fig_dir = copy.deepcopy(args.save_dir)
        if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
        args.eval_k = [int(i) for i in args.eval_k.split(',')]
        if args.model.lower().strip() == 'lambda' and args.labeling.lower().strip().startswith('top'): args.train_k = [args.eval_k[args.index_k]] #args.eval_k
        else: args.train_k = [args.eval_k[args.index_k]]

        pr_lines_pertest = defaultdict(list)
        for i in args.train_k:
            if args.model.lower().strip() != 'lambda': labeling = '' #regression models
            elif args.labeling.lower().strip().startswith('top'): labeling = 'top'+str(i)
            else: labeling = args.labeling
            save_dir = os.path.join(args.save_dir, labeling)
            if not os.path.exists(save_dir): os.makedirs(save_dir)

            if os.path.exists(os.path.join(save_dir, 'model'+str(i)+'.pkl')):
                print('Just creating precision recall curves...')
                sys.stdout = open(os.path.join(save_dir, 'out_eval.txt'), 'w')
                pr_lines = eval_curves(args, i, save_dir)
            else:
                sys.stdout = open(os.path.join(save_dir, 'out.txt'), 'w')
                train_set, test_set = get_features(args.dataset, args.dataset_double, save_dir, labeling, args.onto_features)
                ndcg_train = pyltr.metrics.NDCG(k=i)
                model = get_model(args.model, ndcg_train)
                pr_lines = run(args, model, train_set, test_set, i, labeling, ndcg_train, save_dir)

            plt.clf()
            for precision, recall, label in pr_lines:
                plt.step(recall, precision, where='post', label=label)
                k = label.split('test.top')[-1]
                pr_lines_pertest[k].append((precision, recall, 'train' + str(i) + 'test' + k))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.legend(loc="best")
            plt.savefig(os.path.join(save_dir, '.'.join(['PRcurve', 'train' + str(i), 'png'])), facecolor='white',
                        edgecolor='none', bbox_inches="tight")

        for test_k in pr_lines_pertest:
            plt.clf()
            plt.cla()
            for precision, recall, label in pr_lines_pertest[test_k]:
                plt.step(recall, precision, where='post', label=label)
                pr_lines_permodel[test_k].append((precision, recall, args.model+'_'+label.split('test')[0] if args.model.lower().strip() == 'lambda' else args.model))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Precision-Recall')
            plt.legend(loc="best")
            plt.savefig(os.path.join(fig_dir, '.'.join(['PRcurve', 'test' + str(test_k), 'png'])), facecolor='white',
                        edgecolor='none', bbox_inches="tight")
        args = copy.deepcopy(temp_args)


    #All models plots
    #models_fig = [m for m in models if 'baseline' not in m or 'lambda' not in m]
    plot_allmodels(models, pr_lines_permodel, parent_dir)

    #models_fig = [m for m in models if 'baseline' in m or 'lambda' in m]
    #plot_allmodels(models_fig, pr_lines_permodel, parent_dir, filename='baselines')