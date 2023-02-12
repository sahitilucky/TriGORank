import pandas as pd
import argparse
import os
import sys
import re
from collections import Counter
pd.set_option('display.max_colwidth', -1)
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import GridSearchCV
import torch
import time
import copy
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
pd.options.mode.chained_assignment = None
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer
import pyltr
from ontology import *#make_features, return_list_genes, make_features_adv, return_onto_features,read_go_term_sims,gene_semantic_sims
from Genesemsims import *
#from tensorflow.keras.preprocessing.sequence import pad_sequences
#http://science.sciencemag.org/content/suppl/2018/04/18/360.6386.eaao1729.DC1
#http://boonelab.ccbr.utoronto.ca/supplement/costanzo2016/
from onto_path_graph import Ontology_graph
from format_data import get_top_k_GO_terms
from sklearn.model_selection import KFold
from format_data import get_data,get_data_large
from go_term_feature_selection import go_term_feature_slctn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import ndcg_scoring, parameter_tuning, get_model, write_all_instances, write_relevance_judgments_topranked, precision_recall_k, learning_curves
from utils import get_trigenic_data
import random
logger = logging.getLogger(__name__)

#Variable reliability of the labelled data
#Multiple ways to combine multiple measurements
#Consider both reliable and unreliable data for training and test on reliable data only
#Different ratios of reliable and unreliable data for training


def split_train_test_set_stratify(triple, k=5):
    #K for the K-fold split
    ids = ["query1", "query2", "arr"]
    filenames = ["data/Trigenic_smalldata_stratified_fold1_"]
    triple['sorted_ids'] = triple[ids].apply(lambda x: "_".join(list(sorted(x))), axis=1)
    top_200 = triple[triple["reverserank"]<=200]
    bottom_remaining = triple[triple["reverserank"]>200]
    X_positive = np.array(top_200["sorted_ids"].tolist())
    X_negative = np.array(bottom_remaining["sorted_ids"].tolist())
    #print (X_positive)
    #print (X_negative[:100])
    kf = KFold(n_splits=k, shuffle = False)
    kf.get_n_splits(X_positive)
    kf.get_n_splits(X_negative)
    #X_neg_trains, X_neg_tests = [],[]
    for train_index, test_index in kf.split(X_negative):
        X_neg_train, X_neg_test = X_negative[train_index], X_negative[test_index]
        break
        #X_neg_trains += [X_neg_train]
        #X_neg_tests += [X_neg_test]
    X_pos_trains, X_pos_tests = [], []
    for train_index, test_index in kf.split(X_positive):
        X_pos_train, X_pos_test = X_positive[train_index], X_positive[test_index]
        break
        #X_pos_trains += [X_pos_train]
        #X_pos_tests += [X_pos_test]
    X_train_ids = X_pos_train.tolist() + X_neg_train.tolist()
    X_test_ids = X_pos_test.tolist() + X_neg_test.tolist()
    X_train = triple[triple["sorted_ids"].isin(X_train_ids)]
    X_test = triple[triple["sorted_ids"].isin(X_test_ids)]
    return X_train, X_test

def get_XY_train_test_data_new(train, test, ids, onto_filename, onto_features=True):
    #features = ['query1_score', 'query2_score', 'query3_score', 'query1_query3_score', 'query2_query3_score','query1_query2_score']
    target = "triple_score"
    train['rank'] = train[target].rank(ascending=1, method='first')
    test['rank'] = test[target].rank(ascending=1, method='first')
    train['ids'] = train[ids].apply(lambda x: '_'.join(x), axis=1)
    test['ids'] = test[ids].apply(lambda x: '_'.join(x), axis=1)
    start_time = time.time()
    GO = Ontology(onto_filename)
    GO_graph = Ontology_graph(GO.go)
    triplets = np.hstack((train['ids'].to_numpy(copy="True"), test['ids'].to_numpy(copy="True")))
    GO_graph.make_triplet_graphs(GO, triplets)

    if onto_features:
        start_time = time.time()
        term2genes, gene_idx_features = make_features_adv(GO)
        go_terms_sims = read_go_term_sims()
        train['onto'] = train['ids'].apply(lambda x: return_onto_features(x, term2genes, gene_idx_features, GO.go))
        test['onto'] = test['ids'].apply(lambda x: return_onto_features(x, term2genes, gene_idx_features, GO.go))
        #print("Time taken to finish get features: ", time.time() - start_time)
        #train['onto'] = train['onto'].apply(lambda x: x[:-4])
        #test['onto'] = test['onto'].apply(lambda x: x[:-4])
        '''
        start_time = time.time()
        train['sem_sims'] = train['ids'].apply(lambda x: gene_semantic_sims(x, term2genes, go_terms_sims, "all"))
        test['sem_sims'] = test['ids'].apply(lambda x: gene_semantic_sims(x, term2genes, go_terms_sims, "all"))
        similarities = ["Wang", "Jiang", "Lin", "Resnik", "Rel", "GOGO"]
        for sim_name in similarities:
            train[sim_name] = train['ids'].apply(lambda x: gene_semantic_sims(x, term2genes, go_terms_sims, sim_name))
            test[sim_name] = test['ids'].apply(lambda x: gene_semantic_sims(x, term2genes, go_terms_sims, sim_name))
        print("Time taken to finish getting semsim features: ", time.time() - start_time)
        '''
        start_time = time.time()
        num_regulates = 4
        num_obsolete = 0
        num_part_of = 4
        num_is_a = 4
        num_lcs = 4
        train['regulates'] = train['onto'].apply(lambda x: x[:num_regulates])
        train['obsolete_features'] = train['onto'].apply(lambda x: x[num_regulates:num_regulates+num_obsolete])
        train['part_of_features'] = train['onto'].apply(lambda x: x[num_regulates+num_obsolete:num_regulates+num_obsolete+num_part_of])
        train['is_a_features'] = train['onto'].apply(lambda x: x[num_regulates+num_obsolete+num_part_of:num_regulates+num_obsolete+num_part_of+num_is_a])
        train['part_of_lcs_similarity'] = train['onto'].apply(lambda x: x[num_obsolete+num_part_of+num_is_a:num_obsolete+num_part_of+num_is_a+num_lcs])
        test['regulates'] = test['onto'].apply(lambda x: x[:num_regulates])
        test['obsolete_features'] = test['onto'].apply(lambda x: x[num_regulates:num_regulates+num_obsolete])
        test['part_of_features'] = test['onto'].apply(lambda x: x[num_regulates+num_obsolete:num_regulates+num_obsolete+num_part_of])
        test['is_a_features'] = test['onto'].apply(lambda x: x[num_regulates+num_obsolete+num_part_of:num_regulates+num_obsolete+num_part_of+num_is_a])
        test['is_a_features'] = test['onto'].apply(lambda x: x[num_regulates+num_obsolete+num_part_of:num_regulates+num_obsolete+num_part_of+num_is_a])
        test['part_of_lcs_similarity'] = test['onto'].apply(lambda x: x[num_obsolete+num_part_of+num_is_a:num_obsolete+num_part_of+num_is_a+num_lcs])
        #print("Time taken to finish get features: ", time.time() - start_time)
    y_train = train[target].to_numpy().astype(np.float32)
    ids_train = train['rank'].to_numpy().astype(np.int)
    train_dict = dict(zip(train['rank'], train['ids']))
    y_test = test[target].to_numpy().astype(np.float32)
    ids_test = test['rank'].to_numpy().astype(np.int)
    test_dict = dict(zip(test['rank'], test['ids']))

    start_time = time.time()
    train, test, selected_GO_term_names = go_term_feature_slctn(train, y_train, test, GO, GO_graph)
    #print("Time taken to finish get features: ", time.time() - start_time)

    return (ids_train, y_train, train_dict, train), (ids_test, y_test, test_dict, test),selected_GO_term_names




def testing_model(model, test_set, train_k, save_dir):
    ids_test, X_test, y_test, test_dict, qids = test_set
    #if args.learning_curves: copy_model = copy.deepcopy(model)
    start_time = time.time()
    y_pred = model.predict(X_test)
    print("Time taken to predict:", time.time() - start_time)
    y_test_temp = copy.deepcopy(y_test)
    y_test_temp = sorted(range(len(y_test_temp)), reverse=True, key=lambda i: y_test_temp[i])
    print("Time Taken predict: ", (time.time() - start_time))
    pr_lines = []
    avg_precisions = []
    relevant_k = 40
    eval_k = [10, 20, 30, 40]
    ndcgs = []
    precision_ks, recall_ks = [], []
    start_time = time.time()
    for k in eval_k:
        print('\nEvaluation for top-{} relevant items:'.format(40))
        topitems = y_test_temp[:relevant_k]
        y_test = np.array([1 if i in topitems else 0 for i in range(len(y_test_temp))])
        print('Test relevant:{}, irrelevant:{}, percentage:{}'.format(sum(y_test >= 1), sum(y_test < 1),
                                                                      sum(y_test >= 1) / float(
                                                                          sum(y_test >= 1) + sum(y_test < 1))))
        rel_jud = os.path.join(save_dir, 'relevance_judgments', 'train' + str(train_k) + 'test' + str(k) + '.csv')
        rel_res = os.path.join(save_dir, 'result_ranked', 'train' + str(train_k) + 'test' + str(k) + '.csv')
        write_relevance_judgments_topranked(ids_test, y_pred, rel_res)
        write_relevance_judgments_topranked(ids_test, y_test, rel_jud)

        top_file = os.path.join(save_dir, 'topranked', '.'.join(['train' + str(train_k), 'test' + str(k), 'csv']))
        writer = csv.writer(open(top_file, 'w'), delimiter='\t')
        precision, recall, correct, total, relids, topranked = precision_recall_k(y_pred, y_test, k, ids_test, X_test,
                                                                                  writer=writer)
        print("Our model correct:{} total:{} precision:{} recall:{}".format(correct, total, precision, recall))
        precision_ks += [(k, precision)]
        recall_ks += [(k, recall)]
        ndcg_k = pyltr.metrics.NDCG(k=k)
        print('NDCG Random ranking:', ndcg_k.calc_mean_random(qids, y_test))
        our_model_ndcg = ndcg_k.calc_mean(qids, y_test, y_pred)
        print('NDCG Our model:', our_model_ndcg)
        ndcgs += [(k, our_model_ndcg)]
        process_str = "perl ireval.pl -j " + rel_jud + " < " + rel_res
        print('\n')
        stream = os.popen(process_str).read()
        print(stream)
        start_reading = False
        prec, rec = [], []
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
                avg_precisions += [(relevant_k, avg_pre)]
        # y_pred_pr, y_test_pr = (list(x) for x in zip(*sorted(zip(y_pred, y_test), reverse=True, key=lambda pair: pair[0])))
        # prec, rec, threshold = precision_recall_curve(y_test_pr, y_pred_pr)
        print('precision', len(prec), prec)
        print('recall', len(rec), rec)
    pr_lines.append((prec, rec, 'test.top' + str(relevant_k)))
    print("Time Taken evaluation: ", (time.time() - start_time))
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
    #if args.learning_curves:
    #    learning_curves(args, save_dir, copy_model, X_train, y_train, X_test, ndcg_train)
    return pr_lines


def train_model(args, model_name, model, train_set, test_set, train_k, save_dir, feature_names):
    # features = ['query1_score', 'query2_score', 'query3_score', 'query1_query3_score', 'query2_query3_score', 'query1_query2_score']
    ids_train, X_train, y_train, train_dict = train_set
    ids_test, X_test, y_test, test_dict, qids = test_set
    model_filename = os.path.join(save_dir, 'model' + str(train_k) + '.pkl')
    start_time = time.time()
    print("Training...")
    if model_name == "nn": y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

    if os.path.exists(model_filename):
        model = pickle.load(open(model_filename, 'rb'))
    elif args.tune_model and model_name in ['sgd', 'ridge', 'gaussian', 'svr', 'mlp', 'lambda']:
        ndcg_k = pyltr.metrics.NDCG(k=train_k)
        #modelopt = RandomizedSearchCV(model, param_distributions=parameter_tuning(model_name), cv=5, n_iter=10,
        #                              verbose=True, n_jobs=-1,
        #                              scoring=make_scorer(ndcg_scoring, greater_is_better=True, ndcg_train=ndcg_k,
        #                                                  train_k=train_k), error_score=np.nan)
        modelopt = GridSearchCV(model, param_grid = parameter_tuning(model_name), cv=5, scoring=make_scorer(ndcg_scoring, greater_is_better=True, ndcg_train=ndcg_k,
                                                          train_k=train_k), n_jobs=-1, verbose=True, error_score=np.nan, )
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

    '''
    if not os.path.exists(model_filename):
        # retrain a copy of the model on all data
        copy_model = copy.deepcopy(model)

        X_all = np.vstack((X_train, X_test))
        print(X_all.shape, y_train.shape, y_test.shape)
        y_all = np.concatenate((y_train, y_test), axis=0)
        X_all, y_all = shuffle(X_all, y_all)
        copy_model.fit(X_all, y_all)
        pickle.dump(copy_model, open(os.path.join(save_dir, 'model' + str(train_k) + 'all.pkl'), 'wb'))
    '''
    print('Training finished...')
    print("Time Taken train model: ", (time.time() - start_time))
    # model = pickle.load(open(model_filename, 'rb'))
    start_time = time.time()
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        print('Number of features: ', len(importances))
        # print('feature_importances:', list(zip(features, importances)))
        #print('feature_importances2:', list(zip(['feature' + str(i) for i in range(len(importances))], importances)))
        print('feature_importances2_with_names:', list(zip(feature_names, importances)))

    '''
    if args.print_trees and isinstance(model, RandomForestRegressor):
        for count, tree_in_forest in enumerate(model.estimators_):
            export_graphviz(tree_in_forest, out_file=os.path.join(save_dir, str(count) + 'tree.dot'),
                            feature_names=fitness_features, filled=True, rounded=True)
            os.system('dot -Tpng ' + os.path.join(save_dir, str(count) + 'tree.dot') + ' -o ' + os.path.join(save_dir,
                                                                                                             'tree' + str(
                                                                                                                 count) + '.png'))
    '''
    y_pred_train = model.predict(X_train)
    #if labeling.lower().strip() != 'transfer':
    precision, recall, correct, total, relids, topranked = precision_recall_k(y_pred_train, y_train, train_k,
                                                                                  ids_train, X_train)
    print("Train correct:{} total:{} precision:{} recall:{}".format(correct, total, precision, recall))
    print("Time Taken after train model: ", (time.time() - start_time))
    return model

if __name__== "__main__":
    parser = argparse.ArgumentParser("Genetic triple mutation ltr models")
    parser.add_argument('-onto_features', type=int, default=1, help="use ontology features")
    parser.add_argument('-dataset', default='DeepMutationsData/data/triple_fitness.tsv',
                        type=str, help="path to dataset")
    parser.add_argument('-dataset_double', default='DeepMutationsData/data/double_fitness.tsv',
                        type=str, help="path to dataset")
    parser.add_argument('-eval_k', default="10,30,100,200", type=str,
                        help="list of k cutoff points e.g. 10,30,100,200 for evaluation. First one is also used for training")  # 10,30,100,200 #new200
    parser.add_argument('-seed', type=int, default=1, help="seed")
    parser.add_argument('-model', default="rf", type=str,
                        help="string of models")  # example="sgd_linear_rf_svr_nn_mlp_lambda"]
    parser.add_argument('-learning_curves', type=int, default=0, help="make learning curves")
    parser.add_argument('-print_trees', type=int, default=0, help="visualize trees from random forest")
    parser.add_argument('-tune_model', type=int, default=1, help="perform hyper-parameter tuning")
    parser.add_argument("-save_dir", type=str, default="New_experiments/unreliable_train/baseline_intersct_ws_go_terms_top10_stratify", help="directory for saving results and models")
    # parser.add_argument('-index_k', default=0, type=int, help="index of which eval_k to use, need serious restructuring to be able to do everything!")
    parser.add_argument('-onto_filename', default='DeepMutationsData/data/goslim_yeast.obo',
                        type=str, help="ontology features filename")
    parser.add_argument('-feature_sets', default="baseline", type=str,
                        help="string of feature_sets")  # example = "baseline_baseline+intersct_ws"

    # parser.add_argument('-features_sets', default=, type=int, help="expand features with ontology features")

    args = parser.parse_args()
    parent_dir = args.save_dir
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    save_dir = args.save_dir

    fname_triple = "DeepMutationsData/data/triple_fitness.tsv"
    fname_triple_large = "DeepMutationsData/data/triple_fitness_large.tsv"
    reliable_triple = get_trigenic_data(fname_triple)
    ids = ["query1", "query2", "arr"]
    reliable_triple['sorted_ids'] = reliable_triple[ids].apply(lambda x: "_".join(list(sorted(x))), axis=1)
    reliable_triplet_ids = reliable_triple['sorted_ids'].tolist()
    unreliable_triple = get_trigenic_data(fname_triple_large, ids = ['query1', 'query2','query3'])
    ids = ["query1", "query2", "query3"]
    unreliable_triple['sorted_ids'] = unreliable_triple[ids].apply(lambda x: "_".join(list(sorted(x))), axis=1)
    unreliable_triple["filter_reliable"] = unreliable_triple["sorted_ids"].apply(lambda x: 1 if x in reliable_triplet_ids else 0)
    unreliable_triple_filtered = unreliable_triple[unreliable_triple["filter_reliable"]==0]

    #split training and test data
    train, test = split_train_test_set_stratify(reliable_triple, k=5)
    # combine 50-50% reliable and unreliable
    length = train.shape[0]
    triplet_ids = unreliable_triple_filtered['sorted_ids'].tolist()
    unreliable_train_ids = random.sample(triplet_ids, length)
    unreliable_triple_filtered.drop(['filter_reliable'], axis=1, inplace=True)
    unreliable_train = unreliable_triple_filtered[unreliable_triple_filtered['sorted_ids'].isin(unreliable_train_ids)]
    print ("Reliable train, unreliable train, test: ", train.shape, unreliable_train.shape, test.shape)
    unreliable_train.rename(columns={'query2_query3_score': 'query2_array_score'}, inplace=True)
    unreliable_train.rename(columns={'query1_query3_score': 'query1_array_score'}, inplace=True)
    unreliable_train.rename(columns={'query3_score': 'array_score'}, inplace=True)
    unreliable_train.rename(columns={'query3': 'arr'}, inplace=True)
    train_small = train
    train = pd.concat([train,unreliable_train])

    train_triple_scores = train["triple_score"].tolist()
    ids = ["query1", "query2", "arr"]
    train_set, (ids_test, y_test, test_dict, test), selected_GO_terms_names = get_XY_train_test_data_new(train, test, ids, args.onto_filename, onto_features=True)

    train_set_small, (ids_test, y_test, test_dict, test), selected_GO_terms_names_2 = get_XY_train_test_data_new(train_small, test, ids,
                                                                                                 args.onto_filename,
                                                                                                 onto_features=True)

    #tids = np.ones(len(y_train))
    qids = np.ones(len(y_test))
    print(train.isnull().values.any())
    #train_set = (ids_train, y_train, train_dict, train)
    test_set = (ids_test, y_test, test_dict, qids, test)
    models = [m.lower().strip() for m in args.model.strip().split(",")]
    #models = ['ridge']
    feature_sets = [s for s in args.feature_sets.split(",")]
    onto_feature_names = ["regulates_12_intersection", "regulates_13_intersection", "regulates_23_intersection",
                          "regulates_123_intersection"]
    onto_feature_names += ["partof_12_intersection", "partof_13_intersection", "partof_23_intersection",
                           "partof_123_intersection"]
    onto_feature_names += ["isa_12_intersection", "isa_13_intersection", "isa_23_intersection", "isa_123_intersection"]
    onto_feature_names += ["wusim_12_intersection", "wusim_13_intersection", "wusim_23_intersection",
                           "wusim_123_intersection"]

    pr_lines_permodel = defaultdict(list)
    fitness_features = ['query1_score', 'query2_score', 'array_score', 'query1_array_score', 'query2_array_score',
                        'query1_query2_score']
    train_datas = ["small", "large"]
    train_k = 400
    print (train.isnull().values.any())
    print("train large shape: " , train_set[3].shape)
    print("train small shape: ", train_set_small[3].shape)

    for model in models:
        temp_args = copy.deepcopy(args)
        model_name = model
        #args.model = model
        model_dir = os.path.join(save_dir, model_name)
        fig_dir = copy.deepcopy(model_dir)
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        pr_lines_pertest = defaultdict(list)
        sys.stdout = open(os.path.join(model_dir, 'out.txt'), 'w')
        for train_data_sets in train_datas:
            if train_data_sets == "large":
                (ids_train, y_train, train_dict, train) = train_set
                (ids_test, y_test, test_dict, qids, test) = test_set
                train_label = "large"
            else:
                (ids_train, y_train, train_dict, train) = train_set_small
                (ids_test, y_test, test_dict, qids, test) = test_set
                train_label = "small"
            for feature_idx,feature_set in enumerate(feature_sets):
                start_time = time.time()
                feature_dir = os.path.join(model_dir, feature_set + "_train_" + train_label)
                if not os.path.exists(feature_dir): os.makedirs(feature_dir)
                sys.stdout = open(os.path.join(feature_dir, 'out_train' + str(train_k) + '.txt'), 'w')
                X_train_onto_agg = None
                X_test_onto_agg = None
                feature_names = []
                for method_idx in range(len(feature_set.split("+"))):
                    if feature_set.split("+")[method_idx] == "baseline":
                        X_train_onto = np.array(train[fitness_features].values.tolist()).astype(np.float32)
                        X_test_onto = np.array(test[fitness_features].values.tolist()).astype(np.float32)
                        feature_names += fitness_features
                    elif feature_set.split("+")[method_idx] == "intersct_ws":
                        X_train_onto = np.array(train["onto"].values.tolist()).astype(np.float32)
                        X_test_onto = np.array(test["onto"].values.tolist()).astype(np.float32)
                        feature_names += onto_feature_names
                    else:
                        if "go_terms_TS_" in feature_set.split("+")[method_idx]:
                            topk = int(feature_set.split("+")[method_idx].split("go_terms_TS_")[1])
                            X_train_onto,X_test_onto = get_top_k_GO_terms(train, test, topk)
                            feature_names += selected_GO_terms_names[:topk]
                        else:
                            X_train_onto = np.array(train[feature_set.split("+")[method_idx]].values.tolist()).astype(np.float32)
                            X_test_onto = np.array(test[feature_set.split("+")[method_idx]].values.tolist()).astype(
                                np.float32)
                    if method_idx == 0:
                        X_train_onto_agg = X_train_onto
                        X_test_onto_agg = X_test_onto
                    else:
                        X_train_onto_agg = np.hstack((X_train_onto_agg,X_train_onto))
                        X_test_onto_agg = np.hstack((X_test_onto_agg, X_test_onto))
                X_train = X_train_onto_agg
                X_test = X_test_onto_agg

                train_set_input = ids_train, X_train, y_train, train_dict
                test_set_input = ids_test, X_test, y_test, test_dict, qids
                if not os.path.exists(os.path.join(feature_dir, 'relevance_judgments')): os.makedirs(os.path.join(feature_dir, 'relevance_judgments'))
                if not os.path.exists(os.path.join(feature_dir, 'result_ranked')): os.makedirs(os.path.join(feature_dir, 'result_ranked'))
                if not os.path.exists(os.path.join(feature_dir, 'topranked')): os.makedirs(os.path.join(feature_dir, 'topranked'))
                ndcg_train = pyltr.metrics.NDCG(k=train_k)
                model = get_model(model_name, ndcg_train)
                model = train_model(args, model_name, model, train_set_input, test_set, train_k, feature_dir, feature_names)
                pr_lines = testing_model(model, test_set_input, train_k, feature_dir)
                #train_model(temp_args, model_name, model, train_feature_set, test_feature_set, train_k, labeling, ndcg_train, feature_dir, fitness_features, feature_names)
                #feature_set = feature_set_long
                for (precision, recall, label1) in pr_lines:
                    k = label1.split('test.top')[-1]
                    pr_lines_pertest[k].append((precision, recall, 'train_' + train_label + "_" + str(train_k) + 'test' + k + feature_set))
                print ("Time taken feature set done: ", (time.time()-start_time))


        cm = plt.get_cmap('tab20')
        NUM_COLORS = len(train_datas)*len(feature_sets)
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

    #train model

    #test model














