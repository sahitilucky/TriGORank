import pandas as pd
import argparse
import copy
import os
import csv
import sys
import pickle
import random
import numpy as np
from format_data import get_XY_train_test_data,get_top_k_GO_terms, get_XY_data
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


def get_features(dataset, dataset_double, save_dir, labeling, onto_features, train_ids_file, test_ids_file, test_size = 0.3):
    start_time = time.time()
    (ids_train, y_train, train_dict, train), (ids_test, y_test, test_dict, test),train_k_value,selected_GO_terms_names = get_XY_train_test_data(dataset, dataset_double, test_size=test_size, labeling=labeling, onto_features=onto_features, train_ids_file = train_ids_file, test_ids_file = test_ids_file)
    print ("Time taken to finish get features: ", time.time()-start_time)
    # (ids_train, X_train, y_train, train_dict), (ids_val, X_val, y_val, val_dict), (ids_test, X_test, y_test, test_dict) = get_XY(args.dataset, val_size=0.3, test_size=0.3, labeling=labeling)
    print('Train relevant:{}, irrelevant:{}, percentage:{}'.format(sum(y_train >= 1), sum(y_train < 1), sum(y_train >= 1) / float(sum(y_train >= 1) + sum(y_train < 1))))
    # print('Val relevant:{}, irrelevant:{}, percentage:{}'.format(sum(y_val >= 1), sum(y_val < 1), sum(y_val >= 1)/float(sum(y_val >= 1)+sum(y_val < 1))))
    print('Test relevant:{}, irrelevant:{}, percentage:{}'.format(sum(y_test >= 1), sum(y_test < 1), sum(y_test >= 1) / float(sum(y_test >= 1) + sum(y_test < 1))))
    print('output min:%.3f - max:%.3f' % (y_train.min(), y_train.max()))
    # https://github.com/jma127/pyltr/issues/14 - we only have one session
    #tids = np.ones(len(y_train))
    # vids = np.ones(len(y_val))
    qids = np.ones(len(y_test))
    if not os.path.exists(os.path.join(save_dir, 'relevance_judgments')): os.makedirs(os.path.join(save_dir, 'relevance_judgments'))
    if not os.path.exists(os.path.join(save_dir, 'result_ranked')): os.makedirs(os.path.join(save_dir, 'result_ranked'))
    if not os.path.exists(os.path.join(save_dir, 'topranked')): os.makedirs(os.path.join(save_dir, 'topranked'))
    if not os.path.exists(os.path.join(save_dir, 'all_examples')): os.makedirs(os.path.join(save_dir, 'all_examples'))
    #write_all_instances(X_test, y_test, test_dict, ids_test, os.path.join(save_dir, 'all_examples', 'test_examples'))
    #write_all_instances(X_train, y_train, train_dict, ids_train, os.path.join(save_dir, 'all_examples', 'train_examples'))
    #write_all_instances(X_val, y_val, val_dict, ids_val, os.path.join(save_dir, 'all_examples', 'val_examples'))
    return (ids_train, y_train, train_dict, train), (ids_test, y_test, test_dict, qids, test),train_k_value, selected_GO_terms_names



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


def train_model(args, model_name, model, train_set, test_set, train_k, labeling, save_dir, fitness_features, feature_names):
    '''

    Parameters
    ----------
    args
    model_name
    model
    train_set
    test_set
    train_k
    labeling
    save_dir
    fitness_features
    feature_names

    Returns
    -------

    '''
    #features = ['query1_score', 'query2_score', 'query3_score', 'query1_query3_score', 'query2_query3_score', 'query1_query2_score']
    ids_train, X_train, y_train, train_dict = train_set
    ids_test, X_test, y_test, qids, test_dict = test_set
    model_filename = os.path.join(save_dir, 'model'+str(train_k)+'.pkl')
    start_time = time.time()
    print ("Training...")
    if model_name == "nn": y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

    if os.path.exists(model_filename):
        model = pickle.load(open(model_filename, 'rb'))
    elif args.tune_model and model_name in ['sgd', 'ridge', 'gaussian', 'svr', 'mlp', 'lambda']:
        ndcg_k = pyltr.metrics.NDCG(k=train_k)
        modelopt = RandomizedSearchCV(model, param_distributions=parameter_tuning(model_name), cv=5, n_iter=10,
                verbose=True, n_jobs=-1, scoring= make_scorer(ndcg_scoring,greater_is_better=True, ndcg_train=ndcg_k,train_k=train_k), error_score=np.nan)
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

    if not os.path.exists(model_filename):
        #retrain a copy of the model on all data
        copy_model = copy.deepcopy(model)

        X_all = np.vstack((X_train, X_test))
        print(X_all.shape, y_train.shape, y_test.shape)
        y_all = np.concatenate((y_train, y_test), axis=0)
        X_all, y_all = shuffle(X_all, y_all)
        copy_model.fit(X_all, y_all)
        pickle.dump(copy_model, open(os.path.join(save_dir, 'model' + str(train_k) + 'all.pkl'), 'wb'))

    print('Training finished...')
    print ("Time Taken train model: ", (time.time()-start_time))
    #model = pickle.load(open(model_filename, 'rb'))
    start_time = time.time()
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        print ('Number of features: ', len(importances))
        #print('feature_importances:', list(zip(features, importances)))
        print('feature_importances2:', list(zip(['feature'+str(i) for i in range(len(importances))] , importances)))
        print('feature_importances2_with_names:', list(zip(feature_names, importances)))

    if args.print_trees and isinstance(model, RandomForestRegressor):
        for count, tree_in_forest in enumerate(model.estimators_):
            export_graphviz(tree_in_forest, out_file=os.path.join(save_dir, str(count)+'tree.dot'),
                            feature_names=fitness_features, filled=True, rounded=True)
            os.system('dot -Tpng '+os.path.join(save_dir, str(count)+'tree.dot')+' -o '+os.path.join(save_dir, 'tree' + str(count) + '.png'))

    y_pred_train = model.predict(X_train)
    if labeling.lower().strip() != 'transfer':
        precision, recall,  correct, total, relids, topranked = precision_recall_k(y_pred_train, y_train, train_k, ids_train, X_train)
        print("Train correct:{} total:{} precision:{} recall:{}".format(correct, total, precision, recall))
    print ("Time Taken after train model: ", (time.time()-start_time))
    return model


def run(args, model_name, model, train_set, test_set, train_k, labeling, ndcg_train, save_dir, fitness_features, feature_names):
    '''

    Parameters
    ----------
    args
    model_name
    model
    train_set
    test_set
    train_k
    labeling
    ndcg_train
    save_dir
    fitness_features
    feature_names

    Returns
    -------

    '''
    ids_train, X_train, y_train, train_dict = train_set
    ids_test, X_test, y_test, qids, test_dict= test_set
    if args.learning_curves: copy_model = copy.deepcopy(model)
    model = train_model(args, model_name, model, train_set, test_set, train_k, labeling, save_dir, fitness_features, feature_names)
    start_time = time.time()
    y_pred = model.predict(X_test)
    print ("Time taken to predict:", time.time()-start_time)
    y_test_temp = copy.deepcopy(y_test)
    y_test_temp = sorted(range(len(y_test_temp)), reverse=True, key=lambda i: y_test_temp[i])
    print ("Time Taken predict: ", (time.time()-start_time))
    pr_lines = []
    avg_precisions = []
    relevant_k = 40
    eval_k = [10,20,30,40]
    ndcgs = []
    precision_ks,recall_ks = [],[]
    start_time = time.time()
    for k in eval_k:
        print('\nEvaluation for top-{} relevant items:'.format(40))
        topitems = y_test_temp[:relevant_k]
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
        precision_ks += [(k,precision)]
        recall_ks += [(k,recall)]
        ndcg_k = pyltr.metrics.NDCG(k=k)
        print('NDCG Random ranking:', ndcg_k.calc_mean_random(qids, y_test))
        our_model_ndcg = ndcg_k.calc_mean(qids, y_test, y_pred)
        print('NDCG Our model:', our_model_ndcg)
        ndcgs += [(k,our_model_ndcg)]
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
            if "Average (non-interpolated) precision" in line:
                avg_pre = float(line.strip().split('= ')[1])
                avg_precisions += [(relevant_k,avg_pre)]
        #y_pred_pr, y_test_pr = (list(x) for x in zip(*sorted(zip(y_pred, y_test), reverse=True, key=lambda pair: pair[0])))
        #prec, rec, threshold = precision_recall_curve(y_test_pr, y_pred_pr)
        print('precision', len(prec), prec)
        print('recall', len(rec), rec)
    pr_lines.append((prec, rec, 'test.top'+str(relevant_k)))
    print ("Time Taken evaluation: ", (time.time()-start_time))
    with open(os.path.join(save_dir, "average_precision_train" + str(train_k) + ".txt"), "w") as outfile:
        for (k, avg_pre) in avg_precisions:
            outfile.write("Test" + str(k) + ": " + str(avg_pre) + "\n")
        for (k, ndcg) in ndcgs:
            outfile.write("NDCG" + str(k) + ": " + str(ndcg) + "\n")
        for (k, precision) in precision_ks:
            outfile.write("Precision" + str(k) + ": " + str(precision) + "\n")
        for (k, recall) in recall_ks:
            outfile.write("Recall" + str(k) + ": " + str(recall) + "\n")
    # Learning curves
    if args.learning_curves:
        learning_curves(args, save_dir, copy_model, X_train, y_train, X_test, ndcg_train)
    return pr_lines



if __name__== "__main__":
    parser = argparse.ArgumentParser("Genetic triple mutation ltr models")
    parser.add_argument('-onto_features', type=int, default=1, help="use ontology features")
    parser.add_argument('-dataset', default='../../../srv/local/work/DeepMutationsData/data/triple_fitness.tsv', type=str, help="path to dataset")
    parser.add_argument('-dataset_double', default='../../../srv/local/work/DeepMutationsData/data/double_fitness.tsv', type=str, help="path to dataset")
    parser.add_argument('-eval_k', default="10,30,100,200", type=str, help="list of k cutoff points e.g. 10,30,100,200 for evaluation. First one is also used for training") #10,30,100,200 #new200
    parser.add_argument('-seed', type=int, default=1, help="seed")
    parser.add_argument('-model', default="rf", type=str, help="string of models") #example="sgd_linear_rf_svr_nn_mlp_lambda"]
    parser.add_argument('-labeling', type=str, default='top10,30,100,200', choices=['above_1_topk', 'binarize', 'topk', 'transfer'], help="binarize: make targets 0/1, topk: top k as relevant, transfer: keep all above 1 to test")
    parser.add_argument('-learning_curves', type=int, default=0, help="make learning curves")
    parser.add_argument('-print_trees', type=int, default=0, help="visualize trees from random forest")
    parser.add_argument('-tune_model', type=int, default=1, help="perform hyper-parameter tuning")
    parser.add_argument("-save_dir", type=str, default='onto_exps', help="directory for saving results and models")
    #parser.add_argument('-index_k', default=0, type=int, help="index of which eval_k to use, need serious restructuring to be able to do everything!")
    parser.add_argument('-onto_filename', default='../../../srv/local/work/DeepMutationsData/data/goslim_yeast.obo', type=str, help="ontology features filename")
    #parser.add_argument('-expand_features', default=1, type=int, help="expand features with ontology features")
    parser.add_argument('-test_ids_file', default="data/Trigenic_smalldata_stratified_fold2_test_ids.txt", type=str, help="test_ids_file")
    parser.add_argument('-train_ids_file', default="data/Trigenic_smalldata_stratified_fold2_train_ids.txt", type=str, help="train ids file")
    parser.add_argument('-feature_sets', default= "baseline", type=str,help="string of feature_sets") #example = "baseline_baseline+intersct_ws"

    #parser.add_argument('-features_sets', default=, type=int, help="expand features with ontology features")

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

    train_ids_file = args.train_ids_file
    test_ids_file = args.test_ids_file

    #Run for all models at once
    #TODO: Check if issue with skopt and sklean is fixed https://github.com/scikit-learn/scikit-learn/issues/9857#issuecomment-515968932
    #models = ["RF"]#,"Ridge","Gaussian","lambda"] #["MLP", "RF",  "SGD", "Polynomial", "Gaussian", "Linear", "Ridge", "lambda", "SVR"]#, "NN", 'w12_baseline', 'w13_baseline', 'w23_baseline', 'combo_baseline', "lambda"] # "SVR", "MLP", "RF",  "SGD", "Polynomial", "Polynomial_Bagging",
    #models = ["RF", "Ridge", "Gaussian","lambda", "SVR", "Linear", "Polynomial_Bagging"]
    models = [m.lower().strip() for m in args.model.strip().split(",")]
    #models = ["RF"]
    if "triple_fitness.tsv" in args.dataset:
        fitness_features = ['query1_score', 'query2_score', 'array_score', 'query1_array_score', 'query2_array_score','query1_query2_score']
    elif "triple_fitness_large.tsv" in args.dataset:
        fitness_features = ['query1_score', 'query2_score', 'query3_score', 'query1_query3_score', 'query2_query3_score','query1_query2_score']
    onto_feature_names = ["regulates_12_intersection", "regulates_13_intersection", "regulates_23_intersection", "regulates_123_intersection"]
    onto_feature_names += ["partof_12_intersection", "partof_13_intersection", "partof_23_intersection", "partof_123_intersection"]
    onto_feature_names += ["isa_12_intersection", "isa_13_intersection", "isa_23_intersection","isa_123_intersection"]
    onto_feature_names += ["wusim_12_intersection", "wusim_13_intersection", "wusim_23_intersection", "wusim_123_intersection"]
    #gosemsim_feature_names += []
    #feature_sets = ['regulates', 'obsolete_features', 'part_of_features', 'is_a_features', 'part_of_lcs_similarity', 'all', 'none', "only_onto"]
    #feature_sets = ["sem_sims", "intersct"]
    #feature_sets = ["baseline", "GOGO", "intersct", "intersct+GOGO"]  #"Wang", "Jiang", "Lin", "Resnik", "Rel"
    #feature_sets = ["baseline", "intersct", "intersct+Rel+Resnik"]  #"Wang", "Jiang", "Lin", "Resnik", "Rel"
    #feature_sets = ["baseline", "intersct", "intersct+go_graph_f", "intersct+go_graph_f_only_onto"]
    #feature_sets = ["baseline", "baseline+go_graph_f_10", "baseline+go_graph_f_30", "baseline+go_graph_f_50", "baseline+go_graph_f_100"]
    #feature_sets = ["baseline"]# "baseline+intersct_ws", "baseline+go_terms_TS_10", "baseline+intersct_ws+go_terms_TS_10"]#"baseline+intersct+go_graph_f_30", "baseline+intersct+go_graph_f_50", "baseline+intersct+go_graph_f_100"]
    feature_sets = [s for s in args.feature_sets.split(",")]
    #feature_sets = ["baseline+intersct_ws" ,"baseline+intersct_ws+go_terms_TS_10"] #example = "baseline+go_terms_TS_30", "baseline+intersct_ws+go_terms_TS_30"] #"baseline+intersct_ws+go_terms_TS_50", "baseline+intersct_ws+go_terms_TS_100"]
    pr_lines_permodel = defaultdict(list)

    save_dir = args.save_dir
    sys.stdout = open(os.path.join(save_dir, 'out.txt'), 'w')
    args.eval_k = [int(i) for i in args.eval_k.split(',')]
    # args.train_k = [args.eval_k[args.index_k]]
    args.train_k = [160]
    if args.labeling == "above_1_topk":
        labeling = args.labeling
    else:
        labeling = ""
    #triple_set, selected_GO_terms_names = get_XY_data(args.dataset, args.dataset_double, args.onto_features)

    train_set, test_set, train_k_value, selected_GO_terms_names = get_features(args.dataset, args.dataset_double, save_dir, labeling, args.onto_features, train_ids_file, test_ids_file, test_size=0.0)

    if 'lambda' in models:
        lambda_labeling = 'top' + str(args.train_k[0])
        train_set_lambda, test_set_lambda, train_k_value, selected_GO_terms_names = get_features(args.dataset, args.dataset_double, save_dir, lambda_labeling, args.onto_features,train_ids_file, test_ids_file, test_size=0.0)

    if labeling == "above_1_topk":
        args.train_k = [train_k_value]
        print ("topk for above 1: ", train_k_value)
        #args.train_k += [100]


    for model in models:
        temp_args = copy.deepcopy(args)
        model_name = model
        #args.model = model
        model_dir = os.path.join(save_dir, model_name)
        fig_dir = copy.deepcopy(model_dir)
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        pr_lines_pertest = defaultdict(list)
        sys.stdout = open(os.path.join(model_dir, 'out.txt'), 'w')
        for i in temp_args.train_k:
            #save_dir = os.path.join(model_dir, labeling)
            #save_dir = model_dir
            for feature_idx,feature_set in enumerate(feature_sets):
                start_time = time.time()
                feature_dir = os.path.join(model_dir, feature_set)
                if not os.path.exists(feature_dir): os.makedirs(feature_dir)
                sys.stdout = open(os.path.join(feature_dir, 'out_train' + str(i) + '.txt'), 'w')
                if model == 'lambda':
                    (ids_train, y_train, train_dict, train) = train_set_lambda
                    (ids_test, y_test, test_dict, qids, test) = test_set_lambda
                else:
                    (ids_train, y_train, train_dict, train) = train_set
                    (ids_test, y_test, test_dict, qids, test) = test_set
                X_train_onto_agg = None
                feature_names = []
                for method_idx in range(len(feature_set.split("+"))):
                    if feature_set.split("+")[method_idx] == "baseline":
                        X_train_onto = np.array(train[fitness_features].values.tolist()).astype(np.float32)
                        feature_names += fitness_features
                    elif feature_set.split("+")[method_idx] == "intersct_ws":
                        X_train_onto = np.array(train["onto"].values.tolist()).astype(np.float32)
                        feature_names += onto_feature_names
                    else:
                        if "go_terms_TS_" in feature_set.split("+")[method_idx]:
                            topk = int(feature_set.split("+")[method_idx].split("go_terms_TS_")[1])
                            X_train_onto,_ = get_top_k_GO_terms(train, test, topk)
                            feature_names += selected_GO_terms_names[:topk]
                        else:
                            X_train_onto = np.array(train[feature_set.split("+")[method_idx]].values.tolist()).astype(np.float32)
                    if method_idx == 0:
                        X_train_onto_agg = X_train_onto
                    else:
                        X_train_onto_agg = np.hstack((X_train_onto_agg,X_train_onto))
                X_train = X_train_onto_agg

                X_test_onto_agg = None
                #X_test = np.array(test[fitness_features].values.tolist()).astype(np.float32)
                for method_idx in range(len(feature_set.split("+"))):
                    if feature_set.split("+")[method_idx] == "baseline":
                        X_test_onto = np.array(test[fitness_features].values.tolist()).astype(np.float32)
                    elif feature_set.split("+")[method_idx] == "intersct_ws":
                        X_test_onto = np.array(test["onto"].values.tolist()).astype(np.float32)
                    else:
                        if "go_terms_TS_" in feature_set.split("+")[method_idx]:
                            topk = int(feature_set.split("+")[method_idx].split("go_terms_TS_")[1])
                            _,X_test_onto = get_top_k_GO_terms(train, test, topk)
                        else:
                            X_test_onto = np.array(test[feature_set.split("+")[method_idx]].values.tolist()).astype(np.float32)
                    if method_idx == 0:
                        X_test_onto_agg = X_test_onto
                    else:
                        X_test_onto_agg = np.hstack((X_test_onto_agg,X_test_onto))
                X_test = X_test_onto_agg

                train_feature_set = ids_train, X_train, y_train, train_dict
                test_feature_set = ids_test, X_test, y_test, test_dict, qids
                if not os.path.exists(os.path.join(feature_dir, 'relevance_judgments')): os.makedirs(os.path.join(feature_dir, 'relevance_judgments'))
                if not os.path.exists(os.path.join(feature_dir, 'result_ranked')): os.makedirs(os.path.join(feature_dir, 'result_ranked'))
                if not os.path.exists(os.path.join(feature_dir, 'topranked')): os.makedirs(os.path.join(feature_dir, 'topranked'))
                ndcg_train = pyltr.metrics.NDCG(k=i)
                model = get_model(model_name, ndcg_train)
                pr_lines = run(temp_args, model_name, model, train_feature_set, test_feature_set, i, labeling, ndcg_train, feature_dir, fitness_features, feature_names)
                plt.clf()
                #feature_set = feature_set_long
                for (precision, recall, label1) in pr_lines:
                    plt.step(recall, precision, where = 'post', label=label1)
                    k = label1.split('test.top')[-1]
                    pr_lines_pertest[k].append((precision, recall, 'train' + str(i) + 'test' + k + feature_set))
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall')
                plt.ylim([0.0, 1.05])
                plt.xlim([0.0, 1.0])
                plt.legend(loc="best")
                plt.savefig(os.path.join(feature_dir, '.'.join(['PRcurve', 'train' + str(i), 'png'])), facecolor='white',
                            edgecolor='none', bbox_inches="tight")
                print ("Time taken feature set done: ", (time.time()-start_time))

        cm = plt.get_cmap('tab20')
        NUM_COLORS = len(temp_args.train_k)*len(feature_sets)
        markers = ['o', '^', '+', 's', 'D', '*', 'x','v']
        print(len(pr_lines_pertest))
        for test_k in pr_lines_pertest:
            plt.clf()
            plt.cla()
            count_colors = 0
            for precision, recall, label in pr_lines_pertest[test_k]:
                print(len(pr_lines_pertest[test_k]), len(markers), NUM_COLORS)
                plt.step(recall, precision, where='post', label=label, color=cm(1. * count_colors / NUM_COLORS), marker = markers[count_colors])
                #print (label, label.split("test"),test_k)
                #print (label.split("test" + test_k)[1]) 
                label = model_name + label.split("test" + test_k)[1] #+label.split('test')[0] if args.model.lower().strip() == 'lambda' else args.model
                #print (label)
                pr_lines_permodel[test_k].append((precision, recall, label))#args.model+'_'+label.split('test')[0] if args.model.lower().strip() == 'lambda' else args.model))
                count_colors = count_colors+1
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Precision-Recall')
            plt.legend(loc="best")
            plt.savefig(os.path.join(fig_dir, '.'.join(['PRcurve', 'test' + str(test_k), 'png'])), facecolor='white',
                        edgecolor='none', bbox_inches="tight")
        #args = copy.deepcopy(temp_args)


    #All models plots
    #models_fig = [m for m in models if 'baseline' not in m or 'lambda' not in m]
    #comparing features for all models
    '''
    fig_dir =  os.path.join(parent_dir, 'all_none')
    if not os.path.exists(fig_dir): os.makedirs(fig_dir)
    models_fig = [m+f for m in models for f in ['all', 'none']] 
    plot_allmodels(models_fig, pr_lines_permodel, fig_dir)
    fig_dir =  os.path.join(parent_dir, 'only_onto')
    if not os.path.exists(fig_dir): os.makedirs(fig_dir)
    models_fig = [m+f for m in models for f in ['only_onto']] 
    plot_allmodels(models_fig, pr_lines_permodel, fig_dir)
    '''
    #fig_dir = os.path.join(parent_dir, 'all_models_plots')
    #models_fig = [m + f for m in models for f in ['baseline']]
    #plot_allmodels(models_fig, pr_lines_permodel, fig_dir)
    # WRITE DOWN FEATURE WEIGHTS
    #model = pickle.load(open("New_experiments/onto_sem_sims_all_tune_train_10/RF/all/model10.pkl", "rb"))

    #models_fig = [m for m in models if 'baseline' in m or 'lambda' in m]
    #plot_allmodels(models_fig, pr_lines_permodel, parent_dir, filename='baselines')
    '''
    if os.path.exists(os.path.join(feature_dir, 'model'+str(i)+'.pkl')):
        print('Just creating precision recall curves...')
        #sys.stdout = open(os.path.join(save_dir, 'out_eval.txt'), 'w')
        pr_lines = eval_curves(args, i, feature_dir)
    else:
    '''


